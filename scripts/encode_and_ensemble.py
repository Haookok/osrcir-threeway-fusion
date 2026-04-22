#!/usr/bin/env python3
"""Encode D1/D2 from caches and run prompt-ensemble evaluation."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import open_clip


def load_clip_model(device: str):
    local_weights = Path("/root/.cache/clip/ViT-L-14.pt")
    if local_weights.exists():
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None, device=device)
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        jit_sd = torch.jit.load(str(local_weights), map_location=device).state_dict()
        meta_keys = {"input_resolution", "context_length", "vocab_size"}
        filtered_sd = {k: v for k, v in jit_sd.items() if k not in meta_keys}
        model.load_state_dict(filtered_sd, strict=True)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    return model, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=32):
    outs = []
    for i in range(0, len(texts), batch_size):
        tok = tokenizer(texts[i:i + batch_size]).to(device)
        outs.append(model.encode_text(tok).float().cpu())
    return torch.cat(outs, dim=0)


def l2(t):
    return F.normalize(t, dim=-1)


def encode_cache(cache_path, model, tokenizer, device, fallback_to_d1=True):
    with open(cache_path) as f:
        cache = json.load(f)
    indices = sorted(int(k) for k in cache if cache[k].get("d1"))
    d1_texts = []
    d2_texts = []
    for i in indices:
        entry = cache[str(i)]
        d1 = entry.get("d1", "")
        d2 = entry.get("d2", "")
        if not d2 and fallback_to_d1:
            d2 = d1
        d1_texts.append(d1)
        d2_texts.append(d2)
    print(f"  Encoding {len(indices)} samples from {Path(cache_path).parent.name}")
    d1_feat = encode_texts(model, tokenizer, d1_texts, device)
    d2_feat = encode_texts(model, tokenizer, d2_texts, device)
    return indices, d1_feat, d2_feat


def compute_metrics(sim, target_names, gallery_names):
    name_to_idx = {n: i for i, n in enumerate(gallery_names)}
    metrics = {1: 0, 5: 0, 10: 0, 50: 0}
    valid = 0
    for i, target in enumerate(target_names):
        if target not in name_to_idx:
            continue
        valid += 1
        tgt = name_to_idx[target]
        rank = torch.argsort(sim[i], descending=True).tolist().index(tgt) + 1
        for k in metrics:
            if rank <= k:
                metrics[k] += 1
    return {f"R@{k}": v / valid * 100 for k, v in metrics.items()}, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/ensemble_full.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading CLIP model...")
    model, tokenizer = load_clip_model(device)

    # 加载现有的 short（garment_only 全量）特征
    print("Loading short (garment_only) features...")
    short = pickle.load(open(PROJECT_ROOT / "precomputed_cache/eval_features/fashioniq_dress_gpt4o_minimax_garment_v2_preserveD2_full_eval_features.pkl", "rb"))

    # 加载 Qwen 全量特征
    print("Loading Qwen features...")
    qwen = pickle.load(open(PROJECT_ROOT / "precomputed_cache/eval_features/fashioniq_dress_eval_features.pkl", "rb"))

    # Encode rich 和 cot 的 D1/D2
    print("Encoding rich experiment...")
    rich_indices, rich_d1, rich_d2 = encode_cache(
        PROJECT_ROOT / "outputs/provider_experiments/fashioniq_dress_gpt4o_richD1_full/sample_cache.json",
        model, tokenizer, device)

    print("Encoding cot experiment...")
    cot_indices, cot_d1, cot_d2 = encode_cache(
        PROJECT_ROOT / "outputs/provider_experiments/fashioniq_dress_gpt4o_cotD1_full/sample_cache.json",
        model, tokenizer, device)

    # Gallery
    print("Loading gallery...")
    gallery = pickle.load(open(PROJECT_ROOT / "precomputed_cache/precomputed/fashioniq_dress_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl", "rb"))
    gf = l2(gallery["index_features"].float())
    gallery_names = gallery["index_names"]

    # 找交集：short / rich / cot / qwen 都有的样本
    mshort = {m["index"]: i for i, m in enumerate(short["meta"])}
    mqwen = {m["index"]: i for i, m in enumerate(qwen["meta"])}
    mrich = {idx: i for i, idx in enumerate(rich_indices)}
    mcot = {idx: i for i, idx in enumerate(cot_indices)}

    common = sorted(set(mshort) & set(mrich) & set(mcot) & set(mqwen))
    print(f"\nCommon samples: {len(common)}")

    pshort = [mshort[i] for i in common]
    prich = [mrich[i] for i in common]
    pcot = [mcot[i] for i in common]
    pqwen = [mqwen[i] for i in common]

    target_names = [short["meta"][mshort[i]]["target_name"] for i in common]

    # 提取特征并 l2 归一化
    d1_s = l2(short["d1_features"][pshort].float())
    d2_s = l2(short["d2_features"][pshort].float())
    d1_r = l2(rich_d1[prich].float())
    d2_r = l2(rich_d2[prich].float())
    d1_c = l2(cot_d1[pcot].float())
    d2_c = l2(cot_d2[pcot].float())
    d1_q = l2(qwen["d1_features"][pqwen].float())
    d2_q = l2(qwen["d2_features"][pqwen].float())
    proxy = l2(short["proxy_features"][pshort].float())

    results = []

    def evcfg(label, d1, d2, alpha=0.9, beta=0.7):
        text = l2(beta * d1 + (1 - beta) * d2)
        sim = alpha * (text @ gf.T) + (1 - alpha) * (proxy @ gf.T)
        metrics, valid = compute_metrics(sim, target_names, gallery_names)
        metrics["label"] = label
        metrics["valid_samples"] = valid
        metrics["alpha"] = alpha
        metrics["beta"] = beta
        results.append(metrics)
        print(f"{label:70s} R@10={metrics['R@10']:6.2f}  R@1={metrics['R@1']:5.2f}  R@50={metrics['R@50']:6.2f}")

    print(f"\n=== Single prompt baselines (α=0.9 β=0.7) ===")
    evcfg("short (garment_only)", d1_s, d2_s)
    evcfg("rich", d1_r, d2_r)
    evcfg("cot", d1_c, d2_c)
    evcfg("qwen", d1_q, d2_q)

    print(f"\n=== GPT-4o D1 ensemble (2-way, D2=short) ===")
    evcfg("D1: short+rich", l2(d1_s + d1_r), d2_s)
    evcfg("D1: short+cot", l2(d1_s + d1_c), d2_s)
    evcfg("D1: rich+cot", l2(d1_r + d1_c), d2_s)
    evcfg("D1: short+rich+cot", l2(d1_s + d1_r + d1_c), d2_s)

    print(f"\n=== Weighted D1 ensemble ===")
    for w1, w2, w3 in [(0.6, 0.3, 0.1), (0.5, 0.3, 0.2), (0.7, 0.2, 0.1),
                       (0.4, 0.4, 0.2), (0.5, 0.5, 0.0), (0.5, 0.4, 0.1)]:
        label = f"D1({w1:.1f}s+{w2:.1f}r+{w3:.1f}c)"
        evcfg(label, l2(w1 * d1_s + w2 * d1_r + w3 * d1_c), d2_s)

    print(f"\n=== Cross-model ensemble (GPT + Qwen) ===")
    evcfg("D1: short+qwen", l2(d1_s + d1_q), d2_s)
    evcfg("D1: short+rich+qwen", l2(d1_s + d1_r + d1_q), d2_s)
    evcfg("D1: short+rich+cot+qwen", l2(d1_s + d1_r + d1_c + d1_q), d2_s)

    print(f"\n=== 4-way weighted ensemble ===")
    for ws, wr, wc, wq in [(0.4, 0.3, 0.1, 0.2), (0.5, 0.2, 0.1, 0.2),
                           (0.3, 0.3, 0.1, 0.3), (0.5, 0.25, 0.1, 0.15)]:
        label = f"D1({ws:.2f}s+{wr:.2f}r+{wc:.2f}c+{wq:.2f}q)"
        evcfg(label, l2(ws * d1_s + wr * d1_r + wc * d1_c + wq * d1_q), d2_s)

    print(f"\n=== D2 ensemble (D1=best) ===")
    d1_best = l2(d1_s + d1_r + d1_c)  # assume this
    evcfg("D1=s+r+c, D2=short", d1_best, d2_s)
    evcfg("D1=s+r+c, D2=short+rich", d1_best, l2(d2_s + d2_r))
    evcfg("D1=s+r+c, D2=short+rich+cot", d1_best, l2(d2_s + d2_r + d2_c))
    evcfg("D1=s+r+c, D2=short+qwen", d1_best, l2(d2_s + d2_q))

    # 找最佳
    best = max(results, key=lambda x: x["R@10"])
    print(f"\n=== BEST ===")
    print(f"Config: {best['label']}")
    print(f"Valid={best['valid_samples']}  R@1={best['R@1']:.2f}  R@5={best['R@5']:.2f}  R@10={best['R@10']:.2f}  R@50={best['R@50']:.2f}")

    # α/β 微调最佳
    print(f"\n=== α/β sweep on best config ({best['label']}) ===")
    # 从 label 里反推最佳组合
    if "s+r+c+q" in best["label"]:
        bd1 = l2(d1_s + d1_r + d1_c + d1_q)
    elif "s+r+c" in best["label"] or "short+rich+cot" in best["label"]:
        bd1 = l2(d1_s + d1_r + d1_c)
    elif "s+r+q" in best["label"] or "short+rich+qwen" in best["label"]:
        bd1 = l2(d1_s + d1_r + d1_q)
    elif "short+rich" in best["label"] or ("s+r" in best["label"] and "c" not in best["label"]):
        bd1 = l2(d1_s + d1_r)
    else:
        bd1 = d1_s
    if "D2=short+rich+cot" in best["label"]:
        bd2 = l2(d2_s + d2_r + d2_c)
    elif "D2=short+rich" in best["label"]:
        bd2 = l2(d2_s + d2_r)
    else:
        bd2 = d2_s
    for a in [0.85, 0.88, 0.9, 0.92, 0.95]:
        for b in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            evcfg(f"SWEEP α={a:.2f} β={b:.2f}", bd1, bd2, alpha=a, beta=b)

    best = max(results, key=lambda x: x["R@10"])
    print(f"\n=== FINAL BEST ===")
    print(f"Config: {best['label']}")
    print(f"R@1={best['R@1']:.2f}  R@5={best['R@5']:.2f}  R@10={best['R@10']:.2f}  R@50={best['R@50']:.2f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
