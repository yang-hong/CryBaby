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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import joblib
import random

SR = 16000
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
LABELS = ["hungry", "uncomfortable"]

# ------------------ audio & embedding utils ------------------

def load_mono_16k(path):
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        # consistent resampling path for reproducibility
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="kaiser_best")
    return np.clip(y, -1.0, 1.0).astype(np.float32)

def yamnet_embed(wav, model):
    wf = tf.convert_to_tensor(wav, dtype=tf.float32)
    _, emb, _ = model(wf)
    arr = emb.numpy()
    # average frame embeddings -> one 1024-d vector
    return arr.mean(axis=0).astype(np.float32)

def scan_raw(raw_root):
    items = []
    for lab in LABELS:
        for p in glob.glob(os.path.join(raw_root, lab, "*.wav")):
            items.append((p, LABELS.index(lab)))
    items.sort()
    X_paths = [p for p,_ in items]
    y = np.array([c for _,c in items], dtype=np.int64)
    return X_paths, y

def index_augmented_by_source(aug_root):
    """
    Build mapping: (class, raw_base_name) -> [aug_paths]
    Assumes filenames like <rawbase>_augX_*.wav inside class subfolders.
    """
    mapping = {}
    if not aug_root or not os.path.isdir(aug_root):
        return mapping
    for lab in LABELS:
        for p in glob.glob(os.path.join(aug_root, lab, "*.wav")):
            base = os.path.basename(p).split("_aug")[0]
            mapping.setdefault((lab, base), []).append(p)
    return mapping

def raw_base_name(path):
    return os.path.splitext(os.path.basename(path))[0]

# ------------------ models ------------------

def make_model(kind, seed):
    if kind == "lr":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=seed)
    elif kind == "svm":
        # linear-ish SVM; change to 'rbf' if you want, but keep it simple/robust
        return SVC(C=1.0, kernel="linear", class_weight="balanced", probability=False, random_state=seed)
    elif kind == "mlp":
        # small, regularized MLP; keep deterministic with seed
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=2000, random_state=seed)
    else:
        raise ValueError(f"Unknown model kind: {kind}")

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Option 2: CV for evaluation, then REFIT on 100% data (raw + aug).")
    ap.add_argument("--data-raw", required=True, help="root with class subfolders of RAW wavs")
    ap.add_argument("--data-aug", default="", help="root with class subfolders of AUGMENTED wavs (optional)")
    ap.add_argument("--outdir", required=True, help="output directory for models & metadata")
    ap.add_argument("--model", default="lr", choices=["lr","svm","mlp"], help="classifier type")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yamnet", default=YAMNET_HANDLE, help="TFHub handle or local path for YAMNet")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # 1) read raw set
    raw_paths, y = scan_raw(args.data_raw)
    print(f"RAW files: {len(raw_paths)} | class counts:",
          {LABELS[i]: int((y==i).sum()) for i in range(len(LABELS))})

    # 2) read augmented mapping
    aug_index = index_augmented_by_source(args.data_aug)
    aug_total = sum(len(v) for v in aug_index.values())
    print(f"Indexed augmented files: {aug_total}")

    # 3) load YAMNet once
    yamnet = hub.load(args.yamnet)
    emb_cache = {}

    def embed_path(p):
        if p in emb_cache:
            return emb_cache[p]
        w = load_mono_16k(p)
        e = yamnet_embed(w, yamnet)
        emb_cache[p] = e
        return e

    # 4) build folds (on RAW only)
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    folds = []
    for tr_idx, va_idx in skf.split(raw_paths, y):
        folds.append({"train_idx": tr_idx.tolist(), "val_idx": va_idx.tolist()})

    # Save fold indices for reproducibility
    with open(os.path.join(args.outdir, "folds.json"), "w") as f:
        json.dump(folds, f, indent=2)

    # 5) cross-val: train on raw_train + aug(raw_train), validate on raw_val
    f1s = []
    per_fold = []
    best_model_bundle = None
    best_f1 = -1.0

    for fold_id, fold in enumerate(folds, start=1):
        tr_idx = np.array(fold["train_idx"], dtype=int)
        va_idx = np.array(fold["val_idx"], dtype=int)

        raw_train = [raw_paths[i] for i in tr_idx]
        raw_val   = [raw_paths[i] for i in va_idx]
        y_train   = y[tr_idx]
        y_val     = y[va_idx]

        # expand with aug belonging to train raws
        aug_train = []
        for p in raw_train:
            lab = os.path.basename(os.path.dirname(p))
            base = raw_base_name(p)
            aug_train += aug_index.get((lab, base), [])

        train_paths = raw_train + aug_train
        val_paths   = raw_val

        print(f"[Fold {fold_id}] counts -> train_raw={len(raw_train)}, train_aug={len(aug_train)}, val_raw={len(raw_val)}")

        Xtr = np.stack([embed_path(p) for p in tqdm(train_paths, desc=f"[Fold {fold_id}] embed train")])
        ytr = np.array([LABELS.index(os.path.basename(os.path.dirname(p))) for p in train_paths], dtype=np.int64)

        Xva = np.stack([embed_path(p) for p in tqdm(val_paths, desc=f"[Fold {fold_id}] embed val")])
        yva = y_val

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)

        clf = make_model(args.model, seed=args.seed)
        clf.fit(Xtr_s, ytr)
        ypred = clf.predict(Xva_s)

        f1 = f1_score(yva, ypred, average="macro")
        f1s.append(float(f1))

        print(f"\n[Fold {fold_id}]")
        print(classification_report(yva, ypred, target_names=LABELS, digits=4))
        print("Confusion matrix:\n", confusion_matrix(yva, ypred))
        print(f"Fold {fold_id} macro-F1: {f1:.4f}")

        per_fold.append({
            "fold": fold_id,
            "macro_f1": float(f1),
            "val_size": int(len(yva))
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model_bundle = dict(scaler=scaler, clf=clf, labels=LABELS)

    # 6) summarize CV
    cv_mean = float(np.mean(f1s))
    cv_std  = float(np.std(f1s))
    print("\n==== Summary (RAW-only validation) ====")
    print("Macro-F1 per fold:", [f"{v:.4f}" for v in f1s])
    print("Macro-F1 mean:", f"{cv_mean:.4f}", "±", f"{cv_std:.4f}")

    # 7) REFIT on 100% of the data (Option 2) -> raw + ALL augmented
    print("\nRefitting final model on 100% training audio (raw + all augmented)...")

    # build full training list
    full_raw = list(raw_paths)
    full_aug = []
    # collect *all* aug that map to any raw in the dataset
    for p in full_raw:
        lab = os.path.basename(os.path.dirname(p))
        base = raw_base_name(p)
        full_aug += aug_index.get((lab, base), [])
    full_train_paths = full_raw + full_aug

    Xfull = np.stack([embed_path(p) for p in tqdm(full_train_paths, desc="[Full] embed all train")])
    yfull = np.array([LABELS.index(os.path.basename(os.path.dirname(p))) for p in full_train_paths], dtype=np.int64)

    scaler_full = StandardScaler()
    Xfull_s = scaler_full.fit_transform(Xfull)

    clf_full = make_model(args.model, seed=args.seed)
    clf_full.fit(Xfull_s, yfull)

    bundle_full = dict(scaler=scaler_full, clf=clf_full, labels=LABELS)
    model_name = f"yamnet_{args.model}_full.joblib"
    joblib.dump(bundle_full, os.path.join(args.outdir, model_name))
    print(f"Saved REFIT full-data model to: {os.path.join(args.outdir, model_name)}")

    # 8) metadata
    meta = {
        "option": "Option2_RefitFull",
        "labels": LABELS,
        "samplerate": SR,
        "feature": "YAMNet-avg-embedding-1024",
        "cv_macro_f1_mean": cv_mean,
        "cv_macro_f1_std": cv_std,
        "per_fold": per_fold,
        "kfold": int(args.kfold),
        "seed": int(args.seed),
        "model": args.model,
        "yamnet": args.yamnet,
        "data": {
            "raw_root": args.data_raw,
            "aug_root": args.data_aug,
            "n_raw": int(len(raw_paths)),
            "n_aug_total": int(aug_total),
            "full_train_raw": int(len(full_raw)),
            "full_train_aug": int(len(full_aug)),
            "full_train_total": int(len(full_train_paths))
        },
        "notes": "Evaluated with Stratified K-fold on RAW-only validation; training used raw_train + aug(raw_train). Final model refit on 100% raw + all corresponding augmentations."
    }
    with open(os.path.join(args.outdir, "model_meta_full.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {os.path.join(args.outdir, 'model_meta_full.json')}")

if __name__ == "__main__":
    main()

