import numpy as np
import tensorflow as tf
import imageio_ffmpeg
import subprocess


MODEL_PATH = "models/hey_igor.tflite"
AUDIO_PATH = "sound3.m4a"
SAMPLE_RATE = 16000
NUM_MELS = 96
WINDOW_FRAMES = 16
FRAME_LENGTH = 480  # 30 ms @ 16 kHz
FRAME_STEP = 160  # 10 ms @ 16 kHz
FFT_LENGTH = 512
MEL_LOW_HZ = 20.0
MEL_HIGH_HZ = 7600.0


def load_audio_mono_16k(path: str) -> np.ndarray:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-i",
        path,
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


def compute_log_mel(audio: np.ndarray) -> np.ndarray:
    stft = tf.signal.stft(
        audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH
    )
    power_spec = tf.abs(stft) ** 2
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MELS,
        num_spectrogram_bins=(FFT_LENGTH // 2) + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=MEL_LOW_HZ,
        upper_edge_hertz=MEL_HIGH_HZ,
    )
    mel = tf.matmul(power_spec, mel_matrix)
    log_mel = tf.math.log(mel + 1e-6).numpy().astype(np.float32)
    # Lightweight normalization makes inference more stable across clips.
    return (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)


def quantize_to_int8(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


itp = tf.lite.Interpreter(model_path=MODEL_PATH)
itp.allocate_tensors()
inp = itp.get_input_details()[0]
out = itp.get_output_details()[0]
in_scale, in_zero = inp["quantization"]
out_scale, out_zero = out["quantization"]

print("Input:", inp["shape"], inp["dtype"], inp["quantization"])
print("Output:", out["shape"], out["dtype"], out["quantization"])

audio = load_audio_mono_16k(AUDIO_PATH)
features = compute_log_mel(audio)

if features.shape[0] < WINDOW_FRAMES:
    pad = WINDOW_FRAMES - features.shape[0]
    features = np.pad(features, ((0, pad), (0, 0)), mode="constant")

scores = []
for i in range(features.shape[0] - WINDOW_FRAMES + 1):
    window = features[i : i + WINDOW_FRAMES]
    x = quantize_to_int8(window, in_scale, in_zero)[None, :, :]
    itp.set_tensor(inp["index"], x)
    itp.invoke()
    y_q = int(itp.get_tensor(out["index"])[0, 0])
    y = (y_q - out_zero) * out_scale
    t_sec = (i + WINDOW_FRAMES / 2.0) * (FRAME_STEP / SAMPLE_RATE)
    scores.append((t_sec, y_q, y))

best_t, best_q, best_score = max(scores, key=lambda s: s[2])
mean_score = float(np.mean([s[2] for s in scores]))

print(f"Clip: {AUDIO_PATH}")
print(f"Windows tested: {len(scores)}")
print(f"Best score: {best_score:.6f} (quantized={best_q}) at {best_t:.3f}s")
print(f"Mean score: {mean_score:.6f}")
