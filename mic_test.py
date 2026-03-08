import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import tensorflow as tf
from pymicro_features import MicroFrontend

_numpy_fromstring = np.fromstring


def _fromstring_compat(string, dtype=float, count=-1, sep="", **kwargs):
    if sep == "":
        return np.frombuffer(string, dtype=dtype, count=count)
    return _numpy_fromstring(string, dtype=dtype, count=count, sep=sep, **kwargs)


np.fromstring = _fromstring_compat

try:
    import soundcard as sc
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'soundcard'. Install it with: "
        ".venv\\Scripts\\python.exe -m pip install soundcard"
    ) from exc


MODEL_PATH = Path("models/hey_igor.tflite")
MANIFEST_PATH = Path("models/hey_igor.json")
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.25
RMS_THRESHOLD = 0.015
SILENCE_HOLD_SECONDS = 1.0


def quantize_input(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_output(x: np.ndarray, scale: float, zero_point: int) -> float:
    return float((x.astype(np.float32) - zero_point) * scale)


def make_frontend_features(audio_samples: np.ndarray) -> np.ndarray:
    audio_i16 = np.clip(audio_samples * 32768.0, -32768, 32767).astype(np.int16)
    frontend = MicroFrontend()
    features = []
    audio_bytes = audio_i16.tobytes()
    audio_idx = 0
    num_audio_bytes = len(audio_bytes)

    while audio_idx + 160 * 2 < num_audio_bytes:
        result = frontend.process_samples(audio_bytes[audio_idx : audio_idx + 160 * 2])
        audio_idx += result.samples_read * 2
        if result.features:
            features.append(result.features)

    return np.asarray(features, dtype=np.float32)


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cutoff = float(manifest["micro"]["probability_cutoff"])
    step_ms = int(manifest["micro"]["feature_step_size"])
    sliding_window_size = int(manifest["micro"]["sliding_window_size"])

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    input_slices = int(inp["shape"][1])
    feature_bins = int(inp["shape"][2])
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    mic = sc.default_microphone()
    if mic is None:
        raise RuntimeError("No default microphone found.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    block_frames = int(SAMPLE_RATE * BLOCK_SECONDS)
    block_bytes = block_frames * 2
    silence_hold_blocks = max(1, int(SILENCE_HOLD_SECONDS / BLOCK_SECONDS))
    feature_window = deque(maxlen=input_slices)
    score_window = deque(maxlen=sliding_window_size)
    analysis_index = 0
    silent_blocks = 0

    print(f"Microphone: {mic.name}")
    print("Model input:", inp["shape"], inp["dtype"], inp["quantization"])
    print("Model output:", out["shape"], out["dtype"], out["quantization"])
    print(
        f"Listening at {SAMPLE_RATE} Hz. RMS threshold={RMS_THRESHOLD:.3f}, "
        f"detection cutoff={cutoff:.2f}"
    )
    print("Press Ctrl+C to stop.")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wasapi",
        "-i",
        "default",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    time.sleep(0.5)
    if proc.poll() is not None:
        stderr = ""
        if proc.stderr is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="ignore").strip()
        raise RuntimeError(
            "ffmpeg microphone capture exited during startup. "
            f"Return code={proc.returncode}. stderr={stderr or '<empty>'}"
        )

    try:
        while True:
            if proc.poll() is not None:
                stderr = ""
                if proc.stderr is not None:
                    stderr = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                raise RuntimeError(
                    "ffmpeg microphone capture exited. "
                    f"Return code={proc.returncode}. stderr={stderr or '<empty>'}"
                )

            raw = proc.stdout.read(block_bytes)
            if len(raw) < block_bytes:
                stderr = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                if stderr:
                    raise RuntimeError(f"ffmpeg microphone capture stopped: {stderr}")
                raise RuntimeError("ffmpeg microphone capture stopped before a full block was read.")

            mono_i16 = np.frombuffer(raw, dtype=np.int16)
            mono = mono_i16.astype(np.float32) / 32768.0
            if mono.size == 0:
                continue

            try:
                rms = float(np.sqrt(np.mean(np.square(mono))))
                peak = float(np.max(np.abs(mono)))

                if rms < RMS_THRESHOLD:
                    silent_blocks += 1
                    if silent_blocks >= silence_hold_blocks:
                        feature_window.clear()
                        score_window.clear()
                    continue

                silent_blocks = 0
                features = make_frontend_features(mono)
                if features.size == 0:
                    continue
                if features.shape[1] != feature_bins:
                    raise RuntimeError(
                        f"Feature bin mismatch. Frontend produced {features.shape[1]}, "
                        f"model expects {feature_bins}."
                    )

                for frame in features:
                    feature_window.append(frame)

                if len(feature_window) < input_slices:
                    continue

                x = np.stack(feature_window, axis=0)
                x_q = quantize_input(x, in_scale, in_zero)[None, :, :]
                interpreter.set_tensor(inp["index"], x_q)
                interpreter.invoke()
                y_q = interpreter.get_tensor(out["index"])[0, 0]
                raw_score = dequantize_output(y_q, out_scale, out_zero)
                score_window.append(raw_score)
                smooth_score = float(np.mean(score_window))

                analysis_index += 1
                t_sec = analysis_index * BLOCK_SECONDS
                detected = smooth_score >= cutoff
                print(
                    f"{time.strftime('%H:%M:%S')} "
                    f"t={t_sec:7.2f}s rms={rms:.4f} peak={peak:.4f} "
                    f"frames={features.shape[0]:3d} raw={raw_score:.6f} "
                    f"smooth={smooth_score:.6f} detect={int(detected)}"
                )
            except Exception as exc:
                raise RuntimeError(f"Microphone analysis failed for current block: {exc}") from exc
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
