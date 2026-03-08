import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import tensorflow as tf
from pymicro_features import MicroFrontend


MODEL_PATH = Path("models/hey_igor.tflite")
MANIFEST_PATH = Path("models/hey_igor.json")
AUDIO_PATH = Path("sound.m4a")
SAMPLE_RATE = 16000


def load_audio_mono_16k(path: Path) -> np.ndarray:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def generate_features_for_clip(audio_samples: np.ndarray) -> np.ndarray:
    if audio_samples.dtype in (np.float32, np.float64):
        audio_samples = np.clip(audio_samples * 32768.0, -32768, 32767).astype(np.int16)

    frontend = MicroFrontend()
    features = []
    audio_bytes = audio_samples.tobytes()
    audio_idx = 0
    num_audio_bytes = len(audio_bytes)

    # The frontend consumes 10 ms / 160-sample chunks at 16 kHz.
    while audio_idx + 160 * 2 < num_audio_bytes:
        result = frontend.process_samples(audio_bytes[audio_idx : audio_idx + 160 * 2])
        audio_idx += result.samples_read * 2
        if result.features:
            features.append(result.features)

    return np.asarray(features, dtype=np.float32)


def quantize_input(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_output(x: np.ndarray, scale: float, zero_point: int) -> float:
    return float((x.astype(np.float32) - zero_point) * scale)


manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
cutoff = float(manifest["micro"]["probability_cutoff"])
step_ms = int(manifest["micro"]["feature_step_size"])
sliding_window_size = int(manifest["micro"]["sliding_window_size"])

itp = tf.lite.Interpreter(model_path=str(MODEL_PATH))
itp.allocate_tensors()

inp = itp.get_input_details()[0]
out = itp.get_output_details()[0]
input_slices = int(inp["shape"][1])
feature_bins = int(inp["shape"][2])
in_scale, in_zero = inp["quantization"]
out_scale, out_zero = out["quantization"]

print("Input:", inp["shape"], inp["dtype"], inp["quantization"])
print("Output:", out["shape"], out["dtype"], out["quantization"])

audio = load_audio_mono_16k(AUDIO_PATH)
features = generate_features_for_clip(audio)

if features.size == 0:
    raise RuntimeError("No frontend features were produced from the input audio.")
if features.shape[1] != feature_bins:
    raise RuntimeError(
        f"Feature bin mismatch. Frontend produced {features.shape[1]}, model expects {feature_bins}."
    )

# Pad so we can always feed the first chunk to the streaming model.
if features.shape[0] < input_slices:
    pad = input_slices - features.shape[0]
    features = np.pad(features, ((0, pad), (0, 0)), mode="constant")

scores = []
for end_idx in range(input_slices, features.shape[0] + 1, input_slices):
    chunk = features[end_idx - input_slices : end_idx]
    x = quantize_input(chunk, in_scale, in_zero)[None, :, :]
    itp.set_tensor(inp["index"], x)
    itp.invoke()
    y_q = itp.get_tensor(out["index"])[0, 0]
    y = dequantize_output(y_q, out_scale, out_zero)
    t_sec = end_idx * (step_ms / 1000.0)
    scores.append((t_sec, int(y_q), y))

if not scores:
    raise RuntimeError("No model scores were produced.")

raw_scores = np.array([s[2] for s in scores], dtype=np.float32)
if sliding_window_size > 1 and len(raw_scores) >= sliding_window_size:
    kernel = np.ones(sliding_window_size, dtype=np.float32) / sliding_window_size
    smoothed = np.convolve(raw_scores, kernel, mode="valid")
    smoothed_times = [scores[i + sliding_window_size - 1][0] for i in range(len(smoothed))]
else:
    smoothed = raw_scores
    smoothed_times = [s[0] for s in scores]

best_idx = int(np.argmax(smoothed))
best_score = float(smoothed[best_idx])
best_t = float(smoothed_times[best_idx])
detections = [(t, s) for t, s in zip(smoothed_times, smoothed) if s >= cutoff]

print(f"Clip: {AUDIO_PATH}")
print(f"Feature frames: {features.shape[0]}")
print(f"Chunks tested: {len(scores)}")
print(f"Best smoothed score: {best_score:.6f} at {best_t:.3f}s")
print(f"Detection cutoff: {cutoff:.2f}")
print(f"Detections: {len(detections)}")

print("Timeline:")
for t_sec, raw_score, smooth_score in zip(smoothed_times, raw_scores[-len(smoothed_times) :], smoothed):
    print(f"  {t_sec:6.3f}s raw={raw_score:.6f} smooth={smooth_score:.6f}")

for t_sec, score in detections[:10]:
    print(f"  detected at {t_sec:.3f}s score={score:.6f}")
