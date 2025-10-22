#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, argparse, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# TF / YAMNet
import tensorflow as tf
import tensorflow_hub as hub

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# -----------------------------
# YAMNet config
# -----------------------------
SR = 16000  # YAMNet expects 16 kHz mono
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

LABELS = ["hungry", "uncomfortable"]  # binary classes

def load_mono_16k(path, target_sr=SR):
    """Load audio, convert to mono, resample to 16 kHz, return float32 np.array."""
    wav, sr = sf.read(path, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)  # mono
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    # clip to [-1,1]
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    return wav

def yamnet_embeddings(wav, model):
    """
    Run YAMNet, return averaged embedding per clip.
    model(waveform) -> scores, embeddings, spectrogram
    embeddings shape: [num_patches, 1024]
    """
    # model expects shape [num_samples], float32, 16k
    waveform = tf.convert_to_tensor(wav, dtype=tf.float32)
    scores, embeddings, spectrogram = model(waveform)
    emb = embeddings.numpy()
    # average pooling over time (patches)
    if emb.shape[0] == 0:
        # edge case: extremely short audio — pad zeros
        emb = np.zeros((1, 1024), dtype=np.float32)
    return emb.mean(axis=0).astype(np.float32)

def scan_dataset(data_root):
    """Return lists of (path, label_id)."""
    items = []
    for lab in LABELS:
        pat = os.path.join(data_root, lab, "*.wav")
        for p in glob.glob(pat):
            items.append((p, LABELS.index(lab)))
    if not items:
        raise RuntimeError(f"No .wav found under {data_root}/<label>/*.wav")
    # sort for determinism
    items.sort()
    X_paths = [p for p,_ in items]
    y = np.array([c for _,c in items], dtype=np.int64)
    return X_paths, y

def maybe_load_groups(groups_csv, X_paths):
    """
    Optional grouping for GroupKFold:
    CSV format: path,group_id
    path must match or be a suffix of actual path.
    """
    if not groups_csv:
        return None

    import csv
    mapping = {}
    with open(groups_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mapping[row["path"]] = row["group_id"]

    groups = []
    for p in X_paths:
        # match exact or suffix
        g = mapping.get(p)
        if g is None:
            # try by filename
            base = os.path.basename(p)
            g = mapping.get(base)
        if g is None:
            # last resort: parent folder name as group (not ideal, but prevents crash)
            g = os.path.basename(os.path.dirname(p))
        groups.append(g)
    return np.array(groups)

def extract_all_embeddings(X_paths, yamnet):
    """Compute embeddings (cached in-memory)."""
    feats = []
    for p in tqdm(X_paths, desc="Extracting YAMNet embeddings"):
        wav = load_mono_16k(p, SR)
        emb = yamnet_embeddings(wav, yamnet)
        feats.append(emb)
    X = np.stack(feats, axis=0)
    return X

def train_eval_kfold(X, y, k=5, groups=None, C=1.0, max_iter=2000, outdir="artifacts"):
    os.makedirs(outdir, exist_ok=True)

    if groups is not None:
        splitter = GroupKFold(n_splits=k)
        splits = splitter.split(X, y, groups=groups)
        print(f"Using GroupKFold with k={k}")
    else:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        splits = splitter.split(X, y)
        print(f"Using StratifiedKFold with k={k}")

    f1s = []
    fold = 0
    best_f1 = -1.0
    best_bundle = None

    for tr_idx, va_idx in splits:
        fold += 1
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        # scale
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)

        # class weights (handle imbalance)
        classes = np.unique(ytr)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
        cw = {int(c): w for c, w in zip(classes, class_weights)}

        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=cw,
            solver="lbfgs",  # robust for small-ish dims
            n_jobs=-1
        )
        clf.fit(Xtr_s, ytr)

        ypred = clf.predict(Xva_s)
        yprob = clf.predict_proba(Xva_s)[:, 1] if len(classes) == 2 else None

        print(f"\n[Fold {fold}]")
        print(classification_report(yva, ypred, target_names=LABELS, digits=4))
        print("Confusion matrix:\n", confusion_matrix(yva, ypred))
        f1 = f1_score(yva, ypred, average="macro")
        print(f"Fold {fold} macro-F1: {f1:.4f}")
        f1s.append(f1)

        # remember best fold (for quick prototype saving)
        if f1 > best_f1:
            best_f1 = f1
            best_bundle = dict(
                scaler=scaler, clf=clf, labels=LABELS
            )

    print("\n==== Summary over folds ====")
    print("Macro-F1 per fold:", [f"{v:.4f}" for v in f1s])
    print("Macro-F1 mean:", f"{np.mean(f1s):.4f}", "±", f"{np.std(f1s):.4f}")

    # save best scaler+model
    bundle_path = os.path.join(outdir, "yamnet_lr_best.joblib")
    joblib.dump(best_bundle, bundle_path)
    # save a tiny metadata.json
    with open(os.path.join(outdir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "labels": LABELS,
            "samplerate": SR,
            "feature": "YAMNet-avg-embedding-1024",
            "kfold_macro_f1_mean": float(np.mean(f1s)),
            "kfold_macro_f1_std": float(np.std(f1s))
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved best fold model to: {bundle_path}")
    print(f"Saved metadata to: {os.path.join(outdir, 'model_meta.json')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/raw",
                    help="root dir with class subfolders (hungry/, uncomfortable/)")
    ap.add_argument("--kfold", type=int, default=5, help="number of folds")
    ap.add_argument("--groups-csv", type=str, default=None,
                    help="optional CSV with columns: path,group_id (for GroupKFold)")
    ap.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization strength")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--outdir", type=str, default="artifacts")
    args = ap.parse_args()

    print("Loading file list...")
    X_paths, y = scan_dataset(args.data_root)
    print(f"Found {len(X_paths)} files. Class counts:",
          {LABELS[i]: int((y==i).sum()) for i in range(len(LABELS))})

    # optional groups
    groups = maybe_load_groups(args.groups_csv, X_paths)

    print("Loading YAMNet model from TF Hub...")
    yamnet = hub.load(YAMNET_HANDLE)

    print("Extracting embeddings...")
    X = extract_all_embeddings(X_paths, yamnet)

    print("Training & evaluating with K-Fold...")
    train_eval_kfold(
        X=X, y=y, k=args.kfold,
        groups=groups, C=args.C, max_iter=args.max_iter, outdir=args.outdir
    )

if __name__ == "__main__":
    # Force CPU/GPU memory growth (optional nicety)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    main()

