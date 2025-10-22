#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TFHUB_CACHE_DIR", "./.tfhub_cache")  # avoid corrupted global cache

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import joblib
np.set_printoptions(suppress=True)

# -----------------------
# Config
# -----------------------
SR = 16000
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
LABELS = ["hungry", "uncomfortable"]


# -----------------------
# I/O helpers
# -----------------------
def load_mono_16k(path, target_sr=SR):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return np.clip(y, -1.0, 1.0).astype(np.float32)

def yamnet_embed(wav, model):
    wf = tf.convert_to_tensor(wav, dtype=tf.float32)
    _, emb, _ = model(wf)
    # emb: (frames, 1024) -> average over time
    return emb.numpy().mean(axis=0).astype(np.float32)

def scan_raw(raw_root):
    items = []
    for lab in LABELS:
        for p in glob.glob(os.path.join(raw_root, lab, "*.wav")):
            items.append((p, LABELS.index(lab)))
    items.sort()
    X_paths = [p for p,_ in items]
    y = np.array([c for _,c in items], dtype=np.int64)
    return X_paths, y

def raw_base_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def index_augmented_by_source(aug_root):
    """
    Build mapping: (class_name, raw_base) -> [aug_paths...]
    Expects augmented files named like '<rawbase>_augK_*.wav' inside class subfolders.
    """
    mapping = {}
    for lab in LABELS:
        for p in glob.glob(os.path.join(aug_root, lab, "*.wav")):
            base = os.path.basename(p).split("_aug")[0]
            mapping.setdefault((lab, base), []).append(p)
    return mapping


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-raw", required=True, help="Folder: data/raw")
    ap.add_argument("--data-aug", required=True, help="Folder: data/augmented")
    ap.add_argument("--outdir", default="artifacts_augcv", help="Output dir")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--model", choices=["lr","mlp"], default="lr")
    ap.add_argument("--seed", type=int, default=42)
    # MLP knobs
    ap.add_argument("--mlp-hidden", default="64", help='e.g. "64", "128", "64,32"')
    ap.add_argument("--mlp-alpha", type=float, default=1e-2)
    args = ap.parse_args()

    np.random.seed(args.seed)

    # 1) Discover raw files
    raw_paths, y = scan_raw(args.data_raw)
    cls_counts = {LABELS[i]: int((y == i).sum()) for i in range(len(LABELS))}
    print(f"RAW files: {len(raw_paths)} | class counts: {cls_counts}")

    # 2) Index augmented by source raw base
    aug_index = index_augmented_by_source(args.data_aug)
    aug_count = sum(len(v) for v in aug_index.values())
    print(f"Indexed augmented files: {aug_count}")

    # 3) Load YAMNet once; use local TF Hub cache folder
    try:
        yamnet = hub.load(YAMNET_HANDLE)
    except Exception as e:
        print("Warning loading YAMNet from TF Hub:", e)
        print("If cache is corrupted, delete ./.tfhub_cache and retry.")
        raise

    emb_cache = {}
    def embed_path(p):
        if p in emb_cache:
            return emb_cache[p]
        w = load_mono_16k(p)
        e = yamnet_embed(w, yamnet)
        emb_cache[p] = e
        return e

    # 4) CV setup
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

    fold = 0
    fold_f1s = []
    fold_summaries = []
    best_f1 = -1.0
    best_bundle = None
    best_fold_summary = None

    # 5) Iterate folds
    for tr_idx, va_idx in skf.split(raw_paths, y):
        fold += 1
        raw_train = [raw_paths[i] for i in tr_idx]
        raw_val   = [raw_paths[i] for i in va_idx]
        y_train   = y[tr_idx]
        y_val     = y[va_idx]

        # expand training with aug of ONLY its raw members
        aug_train = []
        for p in raw_train:
            lab = os.path.basename(os.path.dirname(p))       # class from folder
            base = raw_base_name(p)
            aug_train += aug_index.get((lab, base), [])

        print(f"[Fold {fold}] counts -> train_raw={len(raw_train)}, train_aug={len(aug_train)}, val_raw={len(raw_val)}")

        train_paths = raw_train + aug_train
        val_paths   = raw_val  # raw only validation

        # Build X, y
        Xtr = np.stack([embed_path(p) for p in tqdm(train_paths, desc=f"[Fold {fold}] embed train")])
        ytr = np.array(
            [LABELS.index(os.path.basename(os.path.dirname(p))) for p in train_paths],
            dtype=np.int64
        )
        Xva = np.stack([embed_path(p) for p in tqdm(val_paths, desc=f"[Fold {fold}] embed val")])
        yva = y_val

        # Standardize
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        # Choose classifier
        if args.model == "lr":
            clf = LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=args.seed
            )
        else:
            hidden = tuple(int(h) for h in args.mlp_hidden.split(",") if h.strip())
            clf = MLPClassifier(
                hidden_layer_sizes=hidden,
                alpha=args.mlp_alpha,
                max_iter=2000,
                random_state=args.seed
            )

        # Train / predict / score
        clf.fit(Xtr_s, ytr)
        ypred = clf.predict(Xva_s)
        f1 = f1_score(yva, ypred, average="macro")
        cm = confusion_matrix(yva, ypred)

        # Optional: show per-fold report (kept brief)
        print(f"\n[Fold {fold}] macro-F1: {f1:.4f}")

        fold_f1s.append(float(f1))
        summary = {
            "fold": fold,
            "f1": float(f1),
            "n_train_raw": int(len(raw_train)),
            "n_train_aug": int(len(aug_train)),
            "n_val_raw": int(len(raw_val)),
            "confusion_matrix": cm.tolist()
        }
        fold_summaries.append(summary)

        # Track best fold
        if f1 > best_f1:
            best_f1 = float(f1)
            best_bundle = {"scaler": scaler, "clf": clf, "labels": LABELS}
            best_fold_summary = summary.copy()

    # 6) After CV: save best model + metadata files
    mean_f1 = float(np.mean(fold_f1s))
    std_f1  = float(np.std(fold_f1s))

    os.makedirs(args.outdir, exist_ok=True)

    # Best model
    best_model_path = os.path.join(args.outdir, f"yamnet_{args.model}_best.joblib")
    joblib.dump(best_bundle, best_model_path)
    print(f"\nSaved BEST fold model -> {best_model_path}")

    # Average-over-folds metadata
    avg_meta = {
        "model": args.model,
        "labels": LABELS,
        "samplerate": SR,
        "feature": "YAMNet-avg-embedding-1024",
        "kfold": args.kfold,
        "seed": args.seed,
        "f1_per_fold": [float(v) for v in fold_f1s],
        "fold_summaries": fold_summaries,
        "cv_macro_f1_mean": mean_f1,
        "cv_macro_f1_std": std_f1,
        "train_scheme": "train = raw + augmented(raw_train); val = raw only"
    }
    avg_meta_path = os.path.join(args.outdir, f"model_meta_avg_{args.model}.json")
    with open(avg_meta_path, "w", encoding="utf-8") as f:
        json.dump(avg_meta, f, indent=2)
    print(f"Saved AVERAGE-fold metadata -> {avg_meta_path}")

    # Best-fold metadata
    best_meta = {
        "model": args.model,
        "labels": LABELS,
        "samplerate": SR,
        "feature": "YAMNet-avg-embedding-1024",
        "kfold": args.kfold,
        "seed": args.seed,
        "best_fold": int(best_fold_summary["fold"]),
        "best_fold_f1": float(best_fold_summary["f1"]),
        "best_fold_details": best_fold_summary,
        "joblib_path": best_model_path,
        "note": "Single fold with highest macro-F1; model above was saved from this fold."
    }
    best_meta_path = os.path.join(args.outdir, f"model_meta_bestfold_{args.model}.json")
    with open(best_meta_path, "w", encoding="utf-8") as f:
        json.dump(best_meta, f, indent=2)
    print(f"Saved BEST-fold metadata -> {best_meta_path}")

    # Friendly recap
    print("\n==== Summary (RAW-only validation) ====")
    print("Macro-F1 per fold:", [f"{v:.4f}" for v in fold_f1s])
    print(f"Macro-F1 mean: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Best fold: {best_fold_summary['fold']} | Best macro-F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()

