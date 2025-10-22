#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, json
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

SR = 16000
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
LABELS = ["hungry", "uncomfortable"]

RAW_ROOT = "data/raw"
AUG_ROOT = "data/augmented"   # produced by augment.py

def load_mono_16k(path):
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.clip(y, -1.0, 1.0).astype(np.float32)

def yamnet_embed(wav, model):
    wf = tf.convert_to_tensor(wav, dtype=tf.float32)
    _, emb, _ = model(wf)
    arr = emb.numpy()
    return arr.mean(axis=0).astype(np.float32)

def scan_raw(root):
    items = []
    for lab in LABELS:
        for p in glob.glob(os.path.join(root, lab, "*.wav")):
            items.append((p, LABELS.index(lab)))
    items.sort()
    X_paths = [p for p,_ in items]
    y = np.array([c for _,c in items], dtype=np.int64)
    return X_paths, y

def index_augmented_by_source(aug_root):
    """
    Map raw base filename -> list of augmented paths for the same class.
    We assume augment.py produced files like <rawbase>_augX_*.wav
    and kept class subfolders identical.
    """
    mapping = {}  # key: (class, rawbase) -> [aug_paths]
    for lab in LABELS:
        for p in glob.glob(os.path.join(aug_root, lab, "*.wav")):
            fname = os.path.basename(p)
            # get raw base before '_aug'
            base = fname.split("_aug")[0]
            mapping.setdefault((lab, base), []).append(p)
    return mapping

def raw_base_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def main():
    # 1) list raw files / labels
    raw_paths, y = scan_raw(RAW_ROOT)
    print(f"RAW files: {len(raw_paths)} | class counts:",
          {LABELS[i]: int((y==i).sum()) for i in range(len(LABELS))})

    # 2) index augmented files by raw base
    aug_index = index_augmented_by_source(AUG_ROOT)
    print("Indexed augmented groups:", sum(len(v) for v in aug_index.values()))

    # 3) load YAMNet once, cache embeddings per path
    yamnet = hub.load(YAMNET_HANDLE)
    emb_cache = {}

    def embed_path(p):
        if p in emb_cache: return emb_cache[p]
        w = load_mono_16k(p)
        e = yamnet_embed(w, yamnet)
        emb_cache[p] = e
        return e

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1s = []
    fold = 0
    best_bundle = None
    best_f1 = -1

    for tr_idx, va_idx in skf.split(raw_paths, y):
        fold += 1
        raw_train = [raw_paths[i] for i in tr_idx]
        raw_val   = [raw_paths[i] for i in va_idx]
        y_train   = y[tr_idx]
        y_val     = y[va_idx]

        # 4) expand training set with ONLY aug of raw_train files
        aug_train = []
        for p in raw_train:
            lab = LABELS[y[raw_paths.index(p)]]  # class name
            base = raw_base_name(p)
            aug_train += aug_index.get((lab, base), [])

        train_paths = raw_train + aug_train
        val_paths   = raw_val  # strictly raw only

        # 5) build X, y
        Xtr = np.stack([embed_path(p) for p in tqdm(train_paths, desc=f"[Fold {fold}] embed train")])
        ytr = []
        for p in train_paths:
            # label from folder name
            lab = os.path.basename(os.path.dirname(p))
            ytr.append(LABELS.index(lab))
        ytr = np.array(ytr, dtype=np.int64)

        Xva = np.stack([embed_path(p) for p in tqdm(val_paths, desc=f"[Fold {fold}] embed val")])
        yva = y_val  # already from raw

        # 6) scale + train LR (or swap to your best MLP)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)

        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs")
        clf.fit(Xtr_s, ytr)
        ypred = clf.predict(Xva_s)

        print(f"\n[Fold {fold}]")
        print(classification_report(yva, ypred, target_names=LABELS, digits=4))
        print("Confusion matrix:\n", confusion_matrix(yva, ypred))
        f1 = f1_score(yva, ypred, average="macro")
        print(f"Fold {fold} macro-F1: {f1:.4f}")
        f1s.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_bundle = dict(scaler=scaler, clf=clf, labels=LABELS)

    print("\n==== Summary (RAW-only validation) ====")
    print("Macro-F1 per fold:", [f"{v:.4f}" for v in f1s])
    print("Macro-F1 mean:", f"{np.mean(f1s):.4f}", "±", f"{np.std(f1s):.4f}")

    os.makedirs("artifacts_augcv", exist_ok=True)
    joblib.dump(best_bundle, "artifacts_augcv/yamnet_lr_best.joblib")
    with open("artifacts_augcv/model_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "labels": LABELS,
            "samplerate": SR,
            "feature": "YAMNet-avg-embedding-1024",
            "cv_macro_f1_mean": float(np.mean(f1s)),
            "cv_macro_f1_std": float(np.std(f1s)),
            "note": "Trained with raw_train + augmented(raw_train); validated on raw only."
        }, f, indent=2)
    print("Saved best model to artifacts_augcv/yamnet_lr_best.joblib")

if __name__ == "__main__":
    main()

