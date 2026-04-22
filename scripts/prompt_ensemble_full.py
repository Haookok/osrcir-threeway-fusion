#!/usr/bin/env python3
"""Ensemble multi-prompt D1/D2 features on full FashionIQ dress evaluation."""
import argparse
import itertools
import json
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def l2(t):
    return F.normalize(t, dim=-1)


def compute_metrics(sim, target_names, gallery_names):
    name_to_idx = {n: i for i, n in enumerate(gallery_names)}
    metrics = {1: 0, 5: 0, 10: 0, 50: 0}
    for i, target in enumerate(target_names):
        if target not in name_to_idx:
            continue
        tgt = name_to_idx[target]
        rank = torch.argsort(sim[i], descending=True).tolist().index(tgt) + 1
        for k in metrics:
            if rank <= k:
                metrics[k] += 1
    n = len(target_names)
    return {f"R@{k}": v / n * 100 for k, v in metrics.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/full_ensemble_results.json")
    args = p.parse_args()

    short = pickle.load(open(PROJECT_ROOT / "precomputed_cache/eval_features/fashioniq_dress_gpt4o_minimax_garment_v2_preserveD2_full_eval_features.pkl", "rb"))
    rich = pickle.load(open(PROJECT_ROOT / "precomputed_cache/eval_features/fashioniq_dress_gpt4o_richD1_full_eval_features.pkl", "rb"))
    cot = pickle.load(open(PROJECT_ROOT / "precomputed_cache/eval_features/fashioniq_dress_gpt4o_cotD1_full_eval_features.pkl", "rb"))
    gallery = pickle.load(open(PROJECT_ROOT / "precomputed_cache/precomputed/fashioniq_dress_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl", "rb"))

    gf = l2(gallery["index_features"].float())
    gallery_names = gallery["index_names"]

    def pos_map(meta):
        return {m["index"]: i for i, m in enumerate(meta)}

    mshort = pos_map(short["meta"])
    mrich = pos_map(rich["meta"])
    mcot = pos_map(cot["meta"])

    common = sorted(set(mshort) & set(mrich) & set(mcot))
    print(f"Common samples: {len(common)}")

    target_names = [next(m["target_name"] for m in short["meta"] if m["index"] == idx) for idx in common]

    pshort = [mshort[i] for i in common]
    prich = [mrich[i] for i in common]
    pcot = [mcot[i] for i in common]

    d1_s = l2(short["d1_features"][pshort].float())
    d1_r = l2(rich["d1_features"][prich].float())
    d1_c = l2(cot["d1_features"][pcot].float())
    d2_s = l2(short["d2_features"][pshort].float())
    d2_r = l2(rich["d2_features"][prich].float())
    d2_c = l2(cot["d2_features"][pcot].float())
    proxy = l2(short["proxy_features"][pshort].float())

    results = []

    def eval_cfg(label, d1, d2, alpha=0.9, beta=0.7):
        text = l2(beta * d1 + (1 - beta) * d2)
        sim = alpha * (text @ gf.T) + (1 - alpha) * (proxy @ gf.T)
        m = compute_metrics(sim, target_names, gallery_names)
        m["label"] = label
        m["alpha"] = alpha
        m["beta"] = beta
        results.append(m)
        print(f"{label:75s} R@10={m['R@10']:.2f}  R@1={m['R@1']:.2f}  R@50={m['R@50']:.2f}")

    print(f"\n--- Single prompt (α=0.9, β=0.7) ---")
    eval_cfg("short(garment_only) only", d1_s, d2_s)
    eval_cfg("rich only", d1_r, d2_r)
    eval_cfg("cot only", d1_c, d2_c)

    print(f"\n--- D1 pairwise ensemble (D2=short) ---")
    eval_cfg("D1: short+rich (equal), D2: short", l2(d1_s + d1_r), d2_s)
    eval_cfg("D1: short+cot (equal), D2: short", l2(d1_s + d1_c), d2_s)
    eval_cfg("D1: rich+cot (equal), D2: short", l2(d1_r + d1_c), d2_s)
    eval_cfg("D1: short+rich+cot (equal), D2: short", l2(d1_s + d1_r + d1_c), d2_s)

    print(f"\n--- D1 weighted ensemble (D2=short, α=0.9, β=0.7) ---")
    weights = [
        (0.6, 0.3, 0.1), (0.5, 0.3, 0.2), (0.5, 0.4, 0.1), (0.4, 0.4, 0.2),
        (0.7, 0.2, 0.1), (0.5, 0.5, 0.0), (0.4, 0.5, 0.1), (0.5, 0.25, 0.25),
        (0.6, 0.2, 0.2), (0.33, 0.33, 0.33), (0.45, 0.45, 0.10),
    ]
    for w1, w2, w3 in weights:
        label = f"D1({w1:.2f}s+{w2:.2f}r+{w3:.2f}c), D2: short"
        eval_cfg(label, l2(w1 * d1_s + w2 * d1_r + w3 * d1_c), d2_s)

    print(f"\n--- D1 + D2 both ensemble (α=0.9, β=0.7) ---")
    eval_cfg("D1+D2: short+rich", l2(d1_s + d1_r), l2(d2_s + d2_r))
    eval_cfg("D1+D2: short+rich+cot", l2(d1_s + d1_r + d1_c), l2(d2_s + d2_r + d2_c))
    eval_cfg("D1: .6s+.3r+.1c, D2: .6s+.3r+.1c", l2(0.6*d1_s+0.3*d1_r+0.1*d1_c), l2(0.6*d2_s+0.3*d2_r+0.1*d2_c))

    # 找最佳
    best = max(results, key=lambda x: x["R@10"])
    print(f"\n=== BEST on {len(common)} samples ===")
    print(f"Config: {best['label']}")
    print(f"R@1={best['R@1']:.2f}  R@5={best['R@5']:.2f}  R@10={best['R@10']:.2f}  R@50={best['R@50']:.2f}")

    # α/β 微调最佳配置
    if best["R@10"] > 25:
        print(f"\n--- Fine-tune α/β on best config ---")
        if "rich+cot" in best["label"]:
            best_d1 = l2(d1_s + d1_r + d1_c)
            best_d2 = l2(d2_s + d2_r + d2_c)
        elif "short+rich" in best["label"]:
            best_d1 = l2(d1_s + d1_r)
            best_d2 = l2(d2_s + d2_r)
        else:
            best_d1 = l2(0.6*d1_s + 0.3*d1_r + 0.1*d1_c)
            best_d2 = d2_s
        for a in [0.85, 0.88, 0.9, 0.92, 0.95]:
            for b in [0.5, 0.6, 0.7, 0.8, 0.9]:
                eval_cfg(f"BEST: α={a:.2f} β={b:.2f}", best_d1, best_d2, alpha=a, beta=b)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
