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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

SR = 16000
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
LABELS = ["hungry", "uncomfortable"]

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
    mapping = {}
    for lab in LABELS:
        for p in glob.glob(os.path.join(aug_root, lab, "*.wav")):
            base = os.path.basename(p).split("_aug")[0]
            mapping.setdefault((lab, base), []).append(p)
    return mapping

def raw_base_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def get_head(model_name):
    if model_name == "lr":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs")
    elif model_name == "mlp":
        # your best MLP from experiments: (64,) with stronger L2
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=2000, random_state=42)
    else:
        raise ValueError("model_name must be 'lr' or 'mlp'")

def run_once(raw_root, aug_root, model_name="lr", seed=42, outdir="artifacts_augcv"):
    raw_paths, y = scan_raw(raw_root)
    print(f"RAW files: {len(raw_paths)} | class counts:",
          {LABELS[i]: int((y==i).sum()) for i in range(len(LABELS))})

    aug_index = index_augmented_by_source(aug_root)
    total_aug = sum(len(v) for v in aug_index.values())
    print("Indexed augmented files:", total_aug)

    yamnet = hub.load(YAMNET_HANDLE)
    emb_cache = {}

    def embed_path(p):
        if p in emb_cache: return emb_cache[p]
        e = yamnet_embed(load_mono_16k(p), yamnet)
        emb_cache[p] = e
        return e

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    f1s = []
    best_bundle, best_f1 = None, -1
    fold = 0

    # map raw path -> (label name, base)
    raw_info = {p: (LABELS[y[i]], raw_base_name(p)) for i, p in enumerate(raw_paths)}

    for tr_idx, va_idx in skf.split(raw_paths, y):
        fold += 1
        raw_train = [raw_paths[i] for i in tr_idx]
        raw_val   = [raw_paths[i] for i in va_idx]
        y_val     = y[va_idx]

        # collect aug only for raw_train
        aug_train = []
        for p in raw_train:
            lab, base = raw_info[p]
            aug_train += aug_index.get((lab, base), [])

        train_paths = raw_train + aug_train
        val_paths   = raw_val  # raw only

        print(f"[Fold {fold}] counts -> train_raw={len(raw_train)}, train_aug={len(aug_train)}, val_raw={len(raw_val)}")

        Xtr = np.stack([embed_path(p) for p in tqdm(train_paths, desc=f"[Fold {fold}] embed train")])
        ytr = np.array([LABELS.index(os.path.basename(os.path.dirname(p))) for p in train_paths], dtype=np.int64)

        Xva = np.stack([embed_path(p) for p in tqdm(val_paths, desc=f"[Fold {fold}] embed val")])

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)

        clf = get_head(model_name)
        clf.fit(Xtr_s, ytr)
        ypred = clf.predict(Xva_s)

        print(f"\n[Fold {fold}]")
        print(classification_report(y_val, ypred, target_names=LABELS, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_val, ypred))
        f1 = f1_score(y_val, ypred, average="macro")
        print(f"Fold {fold} macro-F1: {f1:.4f}")
        f1s.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_bundle = dict(scaler=scaler, clf=clf, labels=LABELS)

    print("\n==== Summary (RAW-only validation) ====")
    print("Macro-F1 per fold:", [f"{v:.4f}" for v in f1s])
    print("Macro-F1 mean:", f"{np.mean(f1s):.4f}", "±", f"{np.std(f1s):.4f}")

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(best_bundle, os.path.join(outdir, f"yamnet_{model_name}_best.joblib"))
    with open(os.path.join(outdir, f"model_meta_{model_name}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "labels": LABELS,
            "samplerate": SR,
            "feature": "YAMNet-avg-embedding-1024",
            "cv_macro_f1_mean": float(np.mean(f1s)),
            "cv_macro_f1_std": float(np.std(f1s)),
            "note": "Train: raw + aug(raw_train). Val: raw only.",
            "seed": seed,
            "model": model_name
        }, f, indent=2)
    print(f"Saved best model to {outdir}/yamnet_{model_name}_best.joblib")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, default="data/raw")
    ap.add_argument("--aug-root", type=str, default="data/augmented")
    ap.add_argument("--model", type=str, default="lr", choices=["lr","mlp"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[42])
    ap.add_argument("--outdir", type=str, default="artifacts_augcv")
    args = ap.parse_args()

    means = []
    for sd in args.seeds:
        run_once(args.raw_root, args.aug_root, model_name=args.model, seed=sd, outdir=args.outdir)
        # (Collecting summaries across seeds would require capturing returns; kept simple here.)

if __name__ == "__main__":
    main()

