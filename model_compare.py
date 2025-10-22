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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# -----------------------------
# Config
# -----------------------------
SR = 16000
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
LABELS = ["hungry", "uncomfortable"]
DATA_ROOT = "data/raw"

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

def main():
    X, y = load_embeddings(DATA_ROOT)

    models = {
        "lr":  LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        "svm": SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True),
        "mlp": MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=1000)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, clf in models.items():
        f1s = []
        for tr, va in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
            clf.fit(Xtr, y[tr])
            ypred = clf.predict(Xva)
            f1s.append(f1_score(y[va], ypred, average="macro"))
        print(name, "macro-F1:", sum(f1s)/len(f1s))

if __name__ == "__main__":
    main()

