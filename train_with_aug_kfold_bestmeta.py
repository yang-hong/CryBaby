#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, json, argparse
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
import random

# -----------------------------
# Config
# -----------------------------
SR = 16000
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
LABELS = ["hungry", "uncomfortable"]
RAW_ROOT = "data/raw"
AUG_ROOT = "data/augmented"   # produced by augment.py
ARTIFACT_DIR = "artifacts_augcv"

# -----------------------------
# Audio / Embedding utils
# -----------------------------
def load_mono_16k(path):
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.clip(y, -1.0, 1.0).astype(np.float32)

def yamnet_embed(wav, model):
    wf = tf.convert_to_tensor(wav, dtype=tf.float32)
    _, emb, _ = model(wf)                # emb: [T, 1024]
    arr = emb.numpy()
    return arr.mean(axis=0).astype(np.float32)  # (1024,)

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
    Map (class, rawbase) -> list of augmented paths.
    Assumes filenames like <rawbase>_augX_*.wav and same class subfolders.
    """
    mapping = {}
    for lab in LABELS:
        for p in glob.glob(os.path.join(aug_root, lab, "*.wav")):
            base = os.path.basename(p).split("_aug")[0]
            mapping.setdefault((lab, base), []).append(p)
    return mapping

def raw_base_name(path):
    return os.path.splitext(os.path.basename(path))[0]

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-root", type=str, default=RAW_ROOT)
    ap.add_argument("--aug-root", type=str, default=AUG_ROOT)
    args = ap.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # 1) list raw files / labels
    raw_paths, y = scan_raw(args.raw_root)
    print(f"RAW files: {len(raw_paths)} | class counts:",
          {LABELS[i]: int((y==i).sum()) for i in range(len(LABELS))})

    # 2) index augmented files by raw base
    aug_index = index_augmented_by_source(args.aug_root)
    total_aug = sum(len(v) for v in aug_index.values())
    print("Indexed augmented files:", total_aug)

    # 3) load YAMNet once, cache embeddings per path
    yamnet = hub.load(YAMNET_HANDLE)
    emb_cache = {}
    def embed_path(p):
        if p in emb_cache: return emb_cache[p]
        e = yamnet_embed(load_mono_16k(p), yamnet)
        emb_cache[p] = e
        return e

    # Helper for class name
    def class_of_path(p):
        return os.path.basename(os.path.dirname(p))  # 'hungry'/'uncomfortable'

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    best_bundle = None
    best_fold_meta = None
    best_f1 = -1.0
    fold = 0

    # (Optional) collect overall CV stats too (not required for your ask)
    cv_f1s = []

    for tr_idx, va_idx in skf.split(raw_paths, y):
        fold += 1
        raw_train = [raw_paths[i] for i in tr_idx]
        raw_val   = [raw_paths[i] for i in va_idx]
        y_val     = y[va_idx]

        # expand train with only aug of raw_train
        aug_train = []
        for p in raw_train:
            lab = class_of_path(p)
            base = raw_base_name(p)
            aug_train += aug_index.get((lab, base), [])

        train_paths = raw_train + aug_train
        val_paths   = raw_val  # raw only

        print(f"[Fold {fold}] counts -> train_raw={len(raw_train)}, train_aug={len(aug_train)}, val_raw={len(raw_val)}")

        # Embed
        Xtr = np.stack([embed_path(p) for p in tqdm(train_paths, desc=f"[Fold {fold}] embed train")])
        ytr = np.array([LABELS.index(class_of_path(p)) for p in train_paths], dtype=np.int64)
        Xva = np.stack([embed_path(p) for p in tqdm(val_paths, desc=f"[Fold {fold}] embed val")])

        # Scale + train LR
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs")
        clf.fit(Xtr_s, ytr)
        ypred = clf.predict(Xva_s)

        # Metrics for THIS FOLD
        report = classification_report(y_val, ypred, target_names=LABELS, digits=4, output_dict=True)
        cm = confusion_matrix(y_val, ypred).tolist()
        f1 = f1_score(y_val, ypred, average="macro")
        cv_f1s.append(f1)

        print(f"\n[Fold {fold}] macro-F1: {f1:.4f}")

        # Track best fold model + its metadata
        if f1 > best_f1:
            best_f1 = f1
            best_bundle = dict(scaler=scaler, clf=clf, labels=LABELS)

            # Save a clean, fold-specific metadata dict
            best_fold_meta = {
                "which_fold": fold,
                "seed": args.seed,
                "model": "lr",
                "labels": LABELS,
                "samplerate": SR,
                "feature": "YAMNet-avg-embedding-1024",
                "train_counts": {
                    "raw": len(raw_train),
                    "aug": len(aug_train),
                    "total": len(train_paths)
                },
                "val_counts": {
                    "raw": len(raw_val)
                },
                "metrics": {
                    "macro_f1": float(f1),
                    "per_class": {
                        lab: {
                            "precision": float(report[lab]["precision"]),
                            "recall": float(report[lab]["recall"]),
                            "f1": float(report[lab]["f1-score"]),
                            "support": int(report[lab]["support"])
                        } for lab in LABELS
                    },
                    "accuracy": float(report["accuracy"]),
                    "macro_avg": {
                        "precision": float(report["macro avg"]["precision"]),
                        "recall": float(report["macro avg"]["recall"]),
                        "f1": float(report["macro avg"]["f1-score"])
                    },
                    "weighted_avg": {
                        "precision": float(report["weighted avg"]["precision"]),
                        "recall": float(report["weighted avg"]["recall"]),
                        "f1": float(report["weighted avg"]["f1-score"])
                    },
                    "confusion_matrix": cm
                },
                "note": "This JSON describes ONLY the best fold that produced yamnet_lr_best.joblib. Train uses raw+aug; validation uses raw only."
            }

    # Save best fold model + its own metadata
    joblib.dump(best_bundle, os.path.join(ARTIFACT_DIR, "yamnet_lr_best.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "model_meta_bestfold_lr.json"), "w", encoding="utf-8") as f:
        json.dump(best_fold_meta, f, indent=2)
    print(f"\nSaved best fold model -> {os.path.join(ARTIFACT_DIR, 'yamnet_lr_best.joblib')}")
    print(f"Saved best fold metadata -> {os.path.join(ARTIFACT_DIR, 'model_meta_bestfold_lr.json')}")

if __name__ == "__main__":
    main()

