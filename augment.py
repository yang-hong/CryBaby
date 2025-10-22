#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, math, argparse, random, uuid
import numpy as np
import soundfile as sf
import librosa

# -----------------------------
# Defaults
# -----------------------------
SR = 16000   # keep consistent with YAMNet pipeline

# -----------------------------
# Helpers: I/O
# -----------------------------
def load_mono(path, target_sr=SR):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y

def save_wav(path, y, sr=SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    sf.write(path, y, sr)

# -----------------------------
# Helpers: noise
# -----------------------------
def rms(x):
    return np.sqrt(np.mean(np.maximum(1e-12, x**2)))

def pink_noise(n):
    # Voss-McCartney pink noise (simple approx using filtering)
    # generate white noise then 1/f filter in frequency domain
    X = np.fft.rfft(np.random.randn(n))
    freqs = np.fft.rfftfreq(n, d=1.0/SR)
    w = np.where(freqs == 0, 0, 1 / np.sqrt(freqs))
    X = X * w
    y = np.fft.irfft(X, n=n)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)

def add_noise_snr(y, snr_db=10.0, mode="pink"):
    n = len(y)
    if mode == "white":
        noise = np.random.randn(n).astype(np.float32)
    else:
        noise = pink_noise(n)
    # scale noise to achieve target SNR
    rms_y = rms(y)
    rms_n_desired = rms_y / (10**(snr_db/20.0))
    rms_n = rms(noise)
    if rms_n < 1e-9:
        return y
    noise = noise * (rms_n_desired / rms_n)
    return np.clip(y + noise, -1.0, 1.0)

def add_external_noise(y, noise_paths, snr_db=10.0):
    if not noise_paths:
        return add_noise_snr(y, snr_db=snr_db, mode="pink")
    p = random.choice(noise_paths)
    n = load_mono(p, target_sr=SR)
    # tile or crop to match length
    if len(n) < len(y):
        reps = math.ceil(len(y) / len(n))
        n = np.tile(n, reps)[:len(y)]
    else:
        start = random.randint(0, len(n) - len(y))
        n = n[start:start+len(y)]
    # scale to target SNR
    rms_y = rms(y)
    rms_n_desired = rms_y / (10**(snr_db/20.0))
    rms_n = rms(n)
    if rms_n < 1e-9:
        return y
    n = n * (rms_n_desired / rms_n)
    return np.clip(y + n, -1.0, 1.0)

# -----------------------------
# Helpers: time-domain aug
# -----------------------------
def time_shift(y, max_ms=80):
    if max_ms <= 0:
        return y
    max_samples = int(SR * max_ms / 1000.0)
    shift = random.randint(-max_samples, max_samples)
    if shift == 0:
        return y
    if shift > 0:
        return np.r_[np.zeros(shift, dtype=np.float32), y[:-shift]]
    else:
        return np.r_[y[-shift:], np.zeros(-shift, dtype=np.float32)]

def speed_perturb(y, min_rate=0.95, max_rate=1.05):
    rate = random.uniform(min_rate, max_rate)
    if abs(rate - 1.0) < 1e-3:
        return y
    # librosa.effects.time_stretch expects STFT; use resample trick for speed+pitch change
    new_len = int(len(y) / rate)
    y_new = librosa.resample(y, orig_sr=SR, target_sr=int(SR * rate))
    # back to SR
    y_new = librosa.resample(y_new, orig_sr=int(SR * rate), target_sr=SR)
    # trim/pad to original length
    if len(y_new) >= len(y):
        return y_new[:len(y)]
    else:
        return np.r_[y_new, np.zeros(len(y)-len(y_new), dtype=np.float32)]

def gain(y, db_range=(-6, 6)):
    db = random.uniform(*db_range)
    factor = 10**(db/20.0)
    return np.clip(y * factor, -1.0, 1.0)

# -----------------------------
# SpecAugment-style (mel domain)
# -----------------------------
def spec_augment_waveform(y, time_mask_frac=0.08, freq_mask_frac=0.08, n_mels=64):
    """
    Apply time & freq masking in mel-spectrogram, then invert back with Griffin-Lim.
    This changes timbre/temporal cues while keeping duration.
    """
    # mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=1024, hop_length=160, win_length=400, n_mels=n_mels, power=2.0
    )  # shape [n_mels, T]
    S_db = librosa.power_to_db(S + 1e-12)

    n_mel, T = S_db.shape

    # time mask
    t_mask = int(T * time_mask_frac)
    if t_mask > 0:
        t0 = random.randint(0, max(0, T - t_mask))
        S_db[:, t0:t0+t_mask] = S_db[:, t0:t0+t_mask].min() - 10.0

    # freq mask
    f_mask = int(n_mel * freq_mask_frac)
    if f_mask > 0:
        f0 = random.randint(0, max(0, n_mel - f_mask))
        S_db[f0:f0+f_mask, :] = S_db[f0:f0+f_mask, :].min() - 10.0

    # back to waveform via Griffin-Lim
    S_mod = librosa.db_to_power(S_db)
    y_hat = librosa.feature.inverse.mel_to_audio(
        S_mod, sr=SR, n_fft=1024, hop_length=160, win_length=400, n_iter=32
    )
    # match length
    if len(y_hat) >= len(y):
        y_hat = y_hat[:len(y)]
    else:
        y_hat = np.r_[y_hat, np.zeros(len(y)-len(y_hat), dtype=np.float32)]
    return y_hat.astype(np.float32)

# -----------------------------
# Compose augmentations
# -----------------------------
def augment_once(
    y,
    speed_min=0.95, speed_max=1.05,
    shift_ms=80,
    snr_min=5, snr_max=20,
    use_ext_noise=False,
    noise_paths=None,
    use_specaug=True
):
    # order: speed -> shift -> gain -> (noise/specaug optionally)
    y1 = speed_perturb(y, min_rate=speed_min, max_rate=speed_max)
    y1 = time_shift(y1, max_ms=shift_ms)
    y1 = gain(y1, db_range=(-6, 6))

    # choose between external/pink noise or specaugment (random)
    if use_specaug and random.random() < 0.5:
        y1 = spec_augment_waveform(y1, time_mask_frac=0.08, freq_mask_frac=0.08)
    else:
        snr_db = random.uniform(snr_min, snr_max)
        if use_ext_noise:
            y1 = add_external_noise(y1, noise_paths or [], snr_db=snr_db)
        else:
            y1 = add_noise_snr(y1, snr_db=snr_db, mode="pink")
    return y1

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True, help="raw dataset root with class subfolders")
    ap.add_argument("--out-dir", type=str, required=True, help="output root for augmented dataset")
    ap.add_argument("--multiplier", type=int, default=5, help="how many augmented variants per source file")
    ap.add_argument("--speed-min", type=float, default=0.95)
    ap.add_argument("--speed-max", type=float, default=1.05)
    ap.add_argument("--shift-ms", type=int, default=80)
    ap.add_argument("--snr-min", type=float, default=5.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--use-ext-noise", type=str, default="false", help="true/false; mix external noise wavs")
    ap.add_argument("--noise-dir", type=str, default=None, help="directory with .wav noise files (optional)")
    ap.add_argument("--use-specaug", type=str, default="true", help="true/false; enable specaugment path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    use_ext_noise = str(args.use_ext_noise).lower() in ("1","true","yes","y")
    use_specaug = str(args.use_specaug).lower() in ("1","true","yes","y")

    noise_paths = []
    if use_ext_noise and args.noise_dir:
        noise_paths = sorted(glob.glob(os.path.join(args.noise_dir, "*.wav")))
        print(f"Found {len(noise_paths)} external noise files.")

    classes = sorted([d for d in os.listdir(args.data_root)
                      if os.path.isdir(os.path.join(args.data_root, d))])

    total_src = 0
    for cls in classes:
        src_dir = os.path.join(args.data_root, cls)
        dst_dir = os.path.join(args.out_dir, cls)
        os.makedirs(dst_dir, exist_ok=True)

        wavs = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
        print(f"[{cls}] {len(wavs)} source files.")
        total_src += len(wavs)

        for wp in wavs:
            y = load_mono(wp, target_sr=SR)
            base = os.path.splitext(os.path.basename(wp))[0]
            for k in range(args.multiplier):
                y_aug = augment_once(
                    y,
                    speed_min=args.speed_min, speed_max=args.speed_max,
                    shift_ms=args.shift_ms,
                    snr_min=args.snr_min, snr_max=args.snr_max,
                    use_ext_noise=use_ext_noise,
                    noise_paths=noise_paths,
                    use_specaug=use_specaug
                )
                out_name = f"{base}_aug{str(k+1)}_{uuid.uuid4().hex[:6]}.wav"
                out_path = os.path.join(dst_dir, out_name)
                save_wav(out_path, y_aug, sr=SR)

    print(f"Done. Augmented from {total_src} source files into folder: {args.out_dir}")

if __name__ == "__main__":
    main()

