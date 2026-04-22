#!/usr/bin/env python3
"""Multi-prompt feature ensemble + α sweep for OSrCIR.

Takes two feature pickles (old short prompt and new rich prompt),
fuses D1/D2/proxy features via weighted average, then does full α sweep.
"""
import argparse
import json
import pickle
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--old_feat", required=True)
    p.add_argument("--new_feat", required=True)
    p.add_argument("--gallery_pkl", required=True)
    p.add_argument("--dataset_name", default="dress")
    p.add_argument("--out_json", default=None)
    p.add_argument("--tag", default="ensemble")
    return p.parse_args()


def norm(x):
    return torch.nn.functional.normalize(x, dim=-1)


def evaluate(sim, target_names, gallery_name_to_idx, ks=(1, 5, 10, 50)):
    r = {f"R@{k}": 0 for k in ks}
    n = 0
    for i, t in enumerate(target_names):
        if t not in gallery_name_to_idx:
            continue
        n += 1
        sidx = torch.argsort(sim[i], descending=True).tolist()
        rank = sidx.index(gallery_name_to_idx[t]) + 1
        for k in ks:
            if rank <= k:
                r[f"R@{k}"] += 1
    return {k: v / n * 100 for k, v in r.items()}


def main():
    args = parse_args()
    old = pickle.load(open(args.old_feat, "rb"))
    new = pickle.load(open(args.new_feat, "rb"))

    oi = {m["index"]: i for i, m in enumerate(old["meta"])}
    ni = {m["index"]: i for i, m in enumerate(new["meta"])}
    common = sorted(set(oi) & set(ni))
    print(f"common samples: {len(common)}")

    def pick(src, src_idx):
        return torch.stack([src[src_idx[i]] for i in common])

    old_d1 = pick(old["d1_features"], oi)
    new_d1 = pick(new["d1_features"], ni)
    old_d2 = pick(old["d2_features"], oi)
    new_d2 = pick(new["d2_features"], ni)
    proxy = pick(old["proxy_features"], oi)  # same proxy

    sample_map = {m["index"]: m for m in old["meta"]}
    targets = [sample_map[i]["target_name"] for i in common]

    gd = pickle.load(open(args.gallery_pkl, "rb"))
    gallery = gd["index_features"]
    gnames = gd["index_names"]
    gi = {n: i for i, n in enumerate(gnames)}
    gn = norm(gallery)

    # normalize individual features
    old_d1_n = norm(old_d1)
    new_d1_n = norm(new_d1)
    old_d2_n = norm(old_d2)
    new_d2_n = norm(new_d2)
    proxy_n = norm(proxy)

    print(f"\n=== 单特征基线 ===")
    for name, f in [("old_d1", old_d1_n), ("new_d1", new_d1_n),
                    ("old_d2", old_d2_n), ("new_d2", new_d2_n),
                    ("proxy", proxy_n)]:
        r = evaluate(f @ gn.T, targets, gi)
        print(f"  {name}: {r}")

    # === 枚举所有 4-特征权重组合 + α ===
    # text = w1*old_d1 + w2*new_d1 + w3*old_d2 + w4*new_d2 (w1+w2+w3+w4 = 1)
    # score = alpha * text@gallery + (1-alpha) * proxy@gallery
    ws_grid = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    alphas = [0.80, 0.85, 0.90, 0.95, 1.00]

    best = {"R@10": 0, "R@1": 0}
    best_cfg = None
    scores = []
    for w1 in ws_grid:
        for w2 in ws_grid:
            for w3 in ws_grid:
                w4 = 1 - w1 - w2 - w3
                if w4 < -1e-6 or w4 > 1 + 1e-6:
                    continue
                if w4 < 0:
                    w4 = 0.0
                text = norm(w1 * old_d1_n + w2 * new_d1_n + w3 * old_d2_n + w4 * new_d2_n)
                tg = text @ gn.T
                pg = proxy_n @ gn.T
                for alpha in alphas:
                    sim = alpha * tg + (1 - alpha) * pg
                    r = evaluate(sim, targets, gi)
                    scores.append({
                        "w1": w1, "w2": w2, "w3": w3, "w4": round(w4, 3),
                        "alpha": alpha, **r
                    })
                    if r["R@10"] > best["R@10"]:
                        best = r
                        best_cfg = {"w1": w1, "w2": w2, "w3": w3, "w4": round(w4, 3), "alpha": alpha}

    print(f"\n=== 最佳 (by R@10): {best_cfg} ===")
    print(f"    {best}")

    # R@1 最优
    best_r1 = max(scores, key=lambda x: x["R@1"])
    print(f"\n=== 最佳 (by R@1): w1={best_r1['w1']} w2={best_r1['w2']} w3={best_r1['w3']} w4={best_r1['w4']} alpha={best_r1['alpha']} ===")
    print(f"    R@1={best_r1['R@1']:.2f} R@5={best_r1['R@5']:.2f} R@10={best_r1['R@10']:.2f} R@50={best_r1['R@50']:.2f}")

    # Top-10 by R@10
    top10 = sorted(scores, key=lambda x: -x["R@10"])[:10]
    print(f"\n=== Top 10 by R@10 ===")
    print(f"  {'w1':>4} {'w2':>4} {'w3':>4} {'w4':>4} {'α':>5} | R@1    R@5    R@10   R@50")
    for s in top10:
        print(f"  {s['w1']:>4.2f} {s['w2']:>4.2f} {s['w3']:>4.2f} {s['w4']:>4.2f} {s['alpha']:>5.2f} | "
              f"{s['R@1']:5.2f}  {s['R@5']:5.2f}  {s['R@10']:5.2f}  {s['R@50']:5.2f}")

    if args.out_json:
        json.dump({"best": best_cfg, "best_metric": best, "n_scan": len(scores),
                   "top10": top10}, open(args.out_json, "w"), indent=2)
        print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
