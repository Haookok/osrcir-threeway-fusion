#!/usr/bin/env python3
"""Ensemble multiple D1 prompt variants on the same 200 samples (dress)."""
import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def l2norm(t):
    return F.normalize(t, dim=-1)


def compute_recalls(query_feats, gallery_feats, gallery_names, target_names):
    sims = query_feats @ gallery_feats.T
    metrics = {1: 0, 5: 0, 10: 0, 50: 0}
    name_to_idx = {n: i for i, n in enumerate(gallery_names)}
    for i, target in enumerate(target_names):
        if target not in name_to_idx:
            continue
        tgt = name_to_idx[target]
        ranks = torch.argsort(sims[i], descending=True).tolist()
        rank = ranks.index(tgt) + 1
        for k in metrics:
            if rank <= k:
                metrics[k] += 1
    n = len(target_names)
    return {f"R@{k}": v / n * 100 for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fashioniq_dress")
    parser.add_argument("--output", default="/tmp/prompt_ensemble_results.json")
    args = parser.parse_args()

    # 三个 200 样本实验特征
    full_feats_path = PROJECT_ROOT / "precomputed_cache/eval_features/fashioniq_dress_gpt4o_minimax_garment_v2_preserveD2_full_eval_features.pkl"
    rich_path = PROJECT_ROOT / "precomputed_cache/eval_features/fashioniq_dress_gpt4o_richD1_200_seed42_eval_features.pkl"
    cot_path = PROJECT_ROOT / "precomputed_cache/eval_features/fashioniq_dress_gpt4o_cotD1_200_seed42_eval_features.pkl"
    gallery_path = PROJECT_ROOT / "precomputed_cache/precomputed/fashioniq_dress_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl"

    full_feats = pickle.load(open(full_feats_path, "rb"))
    rich_feats = pickle.load(open(rich_path, "rb"))
    cot_feats = pickle.load(open(cot_path, "rb"))
    gallery = pickle.load(open(gallery_path, "rb"))

    gf = l2norm(gallery["index_features"].float())
    gallery_names = gallery["index_names"]

    # 从 full（1847 样本）中筛选出与 rich/cot 相同的 200 样本
    full_meta = full_feats["meta"]
    rich_meta = rich_feats["meta"]
    cot_meta = cot_feats["meta"]

    rich_indices = [m["index"] for m in rich_meta]
    cot_indices = [m["index"] for m in cot_meta]
    full_indices = [m["index"] for m in full_meta]

    # 三个都共有的样本
    common = sorted(set(rich_indices) & set(cot_indices) & set(full_indices))
    print(f"Common samples: {len(common)}")

    def select(feats, meta, indices_list):
        pos = [meta_to_pos[i] for i in indices_list]
        return {
            "d1": feats["d1_features"][pos],
            "d2": feats["d2_features"][pos],
            "proxy": feats["proxy_features"][pos],
        }

    def build_pos_map(meta):
        return {m["index"]: i for i, m in enumerate(meta)}

    full_map = build_pos_map(full_meta)
    rich_map = build_pos_map(rich_meta)
    cot_map = build_pos_map(cot_meta)

    full_pos = [full_map[i] for i in common]
    rich_pos = [rich_map[i] for i in common]
    cot_pos = [cot_map[i] for i in common]

    # D1 特征（三个版本）
    d1_short = l2norm(full_feats["d1_features"][full_pos].float())  # garment_only
    d1_rich = l2norm(rich_feats["d1_features"][rich_pos].float())
    d1_cot = l2norm(cot_feats["d1_features"][cot_pos].float())

    # D2 特征（三个版本）
    d2_short = l2norm(full_feats["d2_features"][full_pos].float())
    d2_rich = l2norm(rich_feats["d2_features"][rich_pos].float())
    d2_cot = l2norm(cot_feats["d2_features"][cot_pos].float())

    # proxy 用 full 的
    proxy_feat = l2norm(full_feats["proxy_features"][full_pos].float())

    target_names = [m["target_name"] for m in full_meta]
    target_names = [target_names[full_map[i]] for i in common]

    results = {}

    def eval_config(name, d1, d2, alpha=0.9, beta=0.7):
        text = l2norm(beta * d1 + (1 - beta) * d2)
        sim = alpha * (text @ gf.T) + (1 - alpha) * (proxy_feat @ gf.T)
        # 直接用组合相似度排序
        metrics = {1: 0, 5: 0, 10: 0, 50: 0}
        name_to_idx = {n: i for i, n in enumerate(gallery_names)}
        for i, target in enumerate(target_names):
            if target not in name_to_idx:
                continue
            tgt = name_to_idx[target]
            rank = torch.argsort(sim[i], descending=True).tolist().index(tgt) + 1
            for k in metrics:
                if rank <= k:
                    metrics[k] += 1
        n = len(target_names)
        m = {f"R@{k}": v / n * 100 for k, v in metrics.items()}
        results[name] = m
        print(f"{name:65s} R@10={m['R@10']:.2f}  R@1={m['R@1']:.2f}  R@50={m['R@50']:.2f}")
        return m

    print(f"\nEvaluating on {len(common)} common samples\n")

    # 单 prompt 版本
    eval_config("short only", d1_short, d2_short)
    eval_config("rich only", d1_rich, d2_rich)
    eval_config("cot only", d1_cot, d2_cot)

    # D1 集成
    d1_sr = l2norm(d1_short + d1_rich)
    d1_sc = l2norm(d1_short + d1_cot)
    d1_rc = l2norm(d1_rich + d1_cot)
    d1_src = l2norm(d1_short + d1_rich + d1_cot)

    eval_config("D1: short+rich, D2: short", d1_sr, d2_short)
    eval_config("D1: short+cot, D2: short", d1_sc, d2_short)
    eval_config("D1: rich+cot, D2: short", d1_rc, d2_short)
    eval_config("D1: short+rich+cot, D2: short", d1_src, d2_short)

    # D1+D2 都集成
    d2_sr = l2norm(d2_short + d2_rich)
    d2_src = l2norm(d2_short + d2_rich + d2_cot)

    eval_config("D1: short+rich, D2: short+rich", d1_sr, d2_sr)
    eval_config("D1: short+rich+cot, D2: short+rich+cot", d1_src, d2_src)

    # 扫不同权重的 D1 集成
    print("\n--- D1 weighted ensemble (short w1 + rich w2 + cot w3), D2=short ---")
    for w1, w2, w3 in [(0.6, 0.3, 0.1), (0.5, 0.3, 0.2), (0.5, 0.5, 0.0),
                       (0.7, 0.2, 0.1), (0.4, 0.4, 0.2), (0.33, 0.33, 0.33)]:
        d1_w = l2norm(w1 * d1_short + w2 * d1_rich + w3 * d1_cot)
        name = f"D1({w1:.2f}s+{w2:.2f}r+{w3:.2f}c)"
        eval_config(name, d1_w, d2_short)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
