#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

# -----------------------------
# Config
# -----------------------------
SR = 16000
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
LABELS = ["hungry", "uncomfortable"]
DATA_ROOT = "data/raw"

# -----------------------------
# Data Loading Helpers
# -----------------------------
def load_mono_16k(path):
    wav, sr = sf.read(path, dtype='float32')
    if wav.ndim > 1:  # stereo → mono
        wav = wav.mean(axis=1)
    if sr != SR:  # resample if not 16k
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)

def extract_emb(wav, model):
    waveform = tf.convert_to_tensor(wav, dtype=tf.float32)
    scores, emb, spec = model(waveform)
    arr = emb.numpy()
    return arr.mean(axis=0).astype(np.float32)  # average over time patches

def scan_dataset(data_root):
    items = []
    for lab in LABELS:
        for p in glob.glob(os.path.join(data_root, lab, "*.wav")):
            items.append((p, LABELS.index(lab)))
    items.sort()
    X_paths = [p for p,_ in items]
    y = np.array([c for _,c in items], dtype=np.int64)
    return X_paths, y

def load_embeddings(data_root=DATA_ROOT):
    print("Scanning dataset...")
    X_paths, y = scan_dataset(data_root)
    print("Found", len(X_paths), "files with class counts:",
          {LABELS[i]: int((y==i).sum()) for i in range(len(LABELS))})

    print("Loading YAMNet from TF Hub...")
    yamnet = hub.load(YAMNET_HANDLE)

    feats = []
    for p in tqdm(X_paths, desc="Extracting embeddings"):
        wav = load_mono_16k(p)
        emb = extract_emb(wav, yamnet)
        feats.append(emb)
    X = np.stack(feats, axis=0)
    return X, y

# -----------------------------
# Experiment
# -----------------------------
def run_experiment(X, y, hidden_layer_sizes, alpha=1e-3, dropout=0.2, seeds=[42,123,999]):
    """
    hidden_layer_sizes: tuple, e.g. (64,), (128,), (64,32)
    alpha: L2 regularization
    dropout: sklearn's MLP doesn't have dropout, but L2 acts similarly
    seeds: list of random seeds to repeat CV
    """
    results = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        f1s = []
        for tr, va in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])

            clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                alpha=alpha,
                                max_iter=2000,
                                random_state=seed)

            clf.fit(Xtr, y[tr])
            ypred = clf.predict(Xva)
            f1s.append(f1_score(y[va], ypred, average="macro"))
        results.append(np.mean(f1s))
    return np.mean(results), np.std(results)

def main():
    X, y = load_embeddings(DATA_ROOT)

    configs = [
        {"hidden": (32,), "alpha": 1e-3},
        {"hidden": (128,), "alpha": 1e-3},
        {"hidden": (64,32), "alpha": 1e-3},
        {"hidden": (64,), "alpha": 1e-2},  # stronger reg
    ]

    for cfg in configs:
        mean_f1, std_f1 = run_experiment(
            X, y,
            hidden_layer_sizes=cfg["hidden"],
            alpha=cfg["alpha"],
            seeds=[42,123,999]
        )
        print(f"MLP {cfg['hidden']} alpha={cfg['alpha']} -> mean F1={mean_f1:.3f} ± {std_f1:.3f}")

if __name__ == "__main__":
    main()

