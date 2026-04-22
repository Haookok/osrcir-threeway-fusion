#!/usr/bin/env python3
"""Lightweight prompt A/B: reuse proxies from a previous full experiment,
only regenerate D1 and D2 with new prompts, then evaluate."""
import argparse
import json
import os
import pickle
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import open_clip
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cloudgpt_api
import prompts
import refine_prompts

print_lock = threading.Lock()

def sp(msg):
    with print_lock:
        print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fashioniq_dress")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--mllm_model", default="gpt-4o-2024-08-06")
    p.add_argument("--mllm_api_base", default=os.getenv("OPENAI_COMPAT_API_BASE", "https://yunwu.ai/v1"))
    p.add_argument("--mllm_api_key", default=os.getenv("OPENAI_COMPAT_API_KEY"))
    p.add_argument("--d1_prompt", required=True)
    p.add_argument("--d2_prompt", required=True)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--d1_workers", type=int, default=8)
    p.add_argument("--d2_workers", type=int, default=8)
    p.add_argument("--max_tokens_d1", type=int, default=1024)
    p.add_argument("--max_tokens_d2", type=int, default=1024)
    p.add_argument("--proxy_source_exp", required=True,
                   help="Experiment dir to borrow proxy_XXXXX.jpg from")
    return p.parse_args()


def parse_json_text(raw):
    resp = (raw or "").strip()
    if resp.startswith("<Response>"):
        resp = resp[len("<Response>"):].strip()
    for mk in ["```json", "```"]:
        if resp.startswith(mk):
            resp = resp[len(mk):].strip()
    if resp.endswith("```"):
        resp = resp[:-3].strip()
    try:
        return json.loads(resp)
    except Exception:
        return None


def extract_desc(raw, fallback=""):
    parsed = parse_json_text(raw)
    if isinstance(parsed, dict):
        for key in ["Target Image Description", "Refined Target Description"]:
            v = parsed.get(key, "")
            if isinstance(v, str) and v.strip():
                return v.strip()
    for line in (raw or "").splitlines():
        if ("Target Image Description" in line or "Refined Target Description" in line) and ":" in line:
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return fallback


def build_single(system_prompt, instruction, img_path):
    url = cloudgpt_api.encode_image(img_path)
    user = ('<Input>\n{\n    "Original Image": <image_url>,\n    "Manipulation text": "'
            + instruction + '"\n}')
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": url}},
        ]},
    ]


def build_dual(system_prompt, instruction, ref_path, proxy_path):
    r = cloudgpt_api.encode_image(ref_path)
    p = cloudgpt_api.encode_image(proxy_path)
    user = ('<Input>\n{\n    "Original Image": <image_1>,\n    "Proxy Image": <image_2>,\n'
            '    "Manipulation text": "' + instruction + '"\n}')
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": r}},
            {"type": "image_url", "image_url": {"url": p}},
        ]},
    ]


def call_mllm(messages, model, api_key, api_base, max_tokens, retries=3):
    for i in range(retries):
        try:
            r = cloudgpt_api.get_chat_completion(
                engine=model, messages=messages,
                max_tokens=max_tokens, timeout=120, temperature=0,
                api_key=api_key, api_base=api_base,
            )
            return r.choices[0].message.content
        except Exception as e:
            if i == retries - 1:
                sp(f"[FAIL] MLLM: {e}")
                return ""
            time.sleep(2 * (i + 1))
    return ""


def load_clip(device):
    local = Path("/root/.cache/clip/ViT-L-14.pt")
    if local.exists():
        m, _, pp = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None, device=device)
        tok = open_clip.get_tokenizer("ViT-L-14")
        sd = torch.jit.load(str(local), map_location=device).state_dict()
        meta_keys = {"input_resolution", "context_length", "vocab_size"}
        filtered = {k: v for k, v in sd.items() if k not in meta_keys}
        m.load_state_dict(filtered, strict=True)
    else:
        m, _, pp = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
        tok = open_clip.get_tokenizer("ViT-L-14")
    m.eval()
    return m, pp, tok


@torch.no_grad()
def encode_texts(m, tok, texts, device, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i+bs]).to(device)
        out.append(m.encode_text(t).float().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_images(m, pp, paths, device, bs=16):
    dummy = torch.zeros(3, 224, 224)
    out = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i+bs]:
            try:
                batch.append(pp(Image.open(p).convert("RGB")))
            except Exception:
                batch.append(dummy)
        x = torch.stack(batch).to(device)
        out.append(m.encode_image(x).float().cpu())
    return torch.cat(out, dim=0)


def norm(x):
    return torch.nn.functional.normalize(x, dim=-1)


def metrics(query_feat, gallery_feat, gallery_names, target_names):
    q = norm(query_feat); g = norm(gallery_feat)
    sim = q @ g.T
    gi = {n: i for i, n in enumerate(gallery_names)}
    r = {"R@1": 0, "R@5": 0, "R@10": 0, "R@50": 0}
    n = 0
    for i, t in enumerate(target_names):
        if t not in gi: continue
        n += 1
        sidx = torch.argsort(sim[i], descending=True).tolist()
        rank = sidx.index(gi[t]) + 1
        for k in [1, 5, 10, 50]:
            if rank <= k: r[f"R@{k}"] += 1
    return {k: v / n * 100 for k, v in r.items()}


def threeway(d1, d2, proxy, gallery, gallery_names, target_names, alpha, beta):
    tf = norm(beta * norm(d1) + (1 - beta) * norm(d2))
    pf = norm(proxy); gf = norm(gallery)
    sim = alpha * (tf @ gf.T) + (1 - alpha) * (pf @ gf.T)
    gi = {n: i for i, n in enumerate(gallery_names)}
    r = {"R@1": 0, "R@5": 0, "R@10": 0, "R@50": 0}
    n = 0
    for i, t in enumerate(target_names):
        if t not in gi: continue
        n += 1
        sidx = torch.argsort(sim[i], descending=True).tolist()
        rank = sidx.index(gi[t]) + 1
        for k in [1, 5, 10, 50]:
            if rank <= k: r[f"R@{k}"] += 1
    return {k: v / n * 100 for k, v in r.items()}


def resolve_ref(sample):
    p = sample.get("reference_image_path", "")
    if not p: return ""
    c = Path(p)
    if c.exists(): return str(c)
    c2 = PROJECT_ROOT / p
    return str(c2)


def main():
    args = parse_args()
    assert args.mllm_api_key, "Need OPENAI_COMPAT_API_KEY"

    d1_prompt = getattr(prompts, args.d1_prompt)
    d2_prompt = getattr(refine_prompts, args.d2_prompt)

    out_dir = PROJECT_ROOT / "outputs" / "prompt_ab_rich" / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = out_dir / "cache.json"
    summary_json = out_dir / "summary.json"

    # sample
    baseline_path = PROJECT_ROOT / "outputs" / f"{args.dataset}_full.json"
    all_samples = json.load(open(baseline_path))
    random.seed(args.seed)
    sidx = sorted(random.sample(range(len(all_samples)), args.num_samples))
    samples = [all_samples[i] for i in sidx]
    sp(f"Experiment: {args.experiment_name}, samples={len(samples)}")

    # proxy source
    proxy_src = PROJECT_ROOT / "outputs" / "provider_experiments" / args.proxy_source_exp / "proxy_images"
    assert proxy_src.exists(), f"proxy source {proxy_src} not found"

    cache = json.load(open(cache_json)) if cache_json.exists() else {}

    # Step 1: D1
    todo = []
    for idx, s in zip(sidx, samples):
        k = str(idx)
        rp = resolve_ref(s)
        if k in cache and cache[k].get("d1"): continue
        if not Path(rp).exists(): continue
        todo.append((idx, s, rp))
    sp(f"D1 cached={len(samples)-len(todo)}, todo={len(todo)}")

    def _d1(t):
        idx, s, rp = t
        msg = build_single(d1_prompt, s["instruction"], rp)
        raw = call_mllm(msg, args.mllm_model, args.mllm_api_key, args.mllm_api_base, args.max_tokens_d1)
        return idx, s, rp, raw, extract_desc(raw, "")

    if todo:
        with ThreadPoolExecutor(max_workers=args.d1_workers) as ex:
            fs = [ex.submit(_d1, t) for t in todo]
            for i, f in enumerate(as_completed(fs), 1):
                idx, s, rp, raw, desc = f.result()
                cache[str(idx)] = {
                    "index": idx, "reference_name": s["reference_name"],
                    "reference_image_path": rp, "instruction": s["instruction"],
                    "target_name": s["target_name"], "d1_raw": raw, "d1": desc,
                }
                if i % 20 == 0 or i == len(fs):
                    sp(f"  D1 {i}/{len(fs)}")
                    json.dump(cache, open(cache_json, "w"), ensure_ascii=False, indent=2)
        json.dump(cache, open(cache_json, "w"), ensure_ascii=False, indent=2)

    # Step 2: D2
    todo = []
    for idx, s in zip(sidx, samples):
        k = str(idx)
        rp = resolve_ref(s)
        px = proxy_src / f"proxy_{idx:05d}.jpg"
        if k in cache and cache[k].get("d2"): continue
        if cache.get(k, {}).get("d1") and Path(rp).exists() and px.exists():
            todo.append((idx, s, rp, str(px)))
    sp(f"D2 cached={len(samples)-len(todo)}, todo={len(todo)}")

    def _d2(t):
        idx, s, rp, px = t
        msg = build_dual(d2_prompt, s["instruction"], rp, px)
        raw = call_mllm(msg, args.mllm_model, args.mllm_api_key, args.mllm_api_base, args.max_tokens_d2)
        return idx, raw, extract_desc(raw, fallback=cache.get(str(idx), {}).get("d1", ""))

    if todo:
        with ThreadPoolExecutor(max_workers=args.d2_workers) as ex:
            fs = [ex.submit(_d2, t) for t in todo]
            for i, f in enumerate(as_completed(fs), 1):
                idx, raw, desc = f.result()
                cache.setdefault(str(idx), {})["d2_raw"] = raw
                cache[str(idx)]["d2"] = desc
                if i % 20 == 0 or i == len(fs):
                    sp(f"  D2 {i}/{len(fs)}")
                    json.dump(cache, open(cache_json, "w"), ensure_ascii=False, indent=2)
        json.dump(cache, open(cache_json, "w"), ensure_ascii=False, indent=2)

    # Step 3: evaluate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp(f"Using device: {device}")
    m, pp, tok = load_clip(device)

    gpath = PROJECT_ROOT / "precomputed_cache" / "precomputed" / f"{args.dataset}_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl"
    gd = pickle.load(open(gpath, "rb"))
    gallery = gd["index_features"].to(device)
    gnames = gd["index_names"]

    d1_texts, d2_texts, proxy_paths, meta = [], [], [], []
    for idx, s in zip(sidx, samples):
        e = cache.get(str(idx), {})
        px = proxy_src / f"proxy_{idx:05d}.jpg"
        if e.get("d1") and e.get("d2") and px.exists():
            d1_texts.append(e["d1"])
            d2_texts.append(e["d2"])
            proxy_paths.append(str(px))
            meta.append({
                "index": idx, "target_name": s["target_name"],
                "reference_name": s["reference_name"], "instruction": s["instruction"],
                "d1": e["d1"], "d2": e["d2"],
            })
    sp(f"Valid samples: {len(meta)}")
    targets = [m["target_name"] for m in meta]

    ef_d1 = encode_texts(m, tok, d1_texts, device).to(device)
    ef_d2 = encode_texts(m, tok, d2_texts, device).to(device)
    ef_px = encode_images(m, pp, proxy_paths, device).to(device)

    base = metrics(ef_d1, gallery, gnames, targets)
    ens_only = threeway(ef_d1, ef_d2, ef_px, gallery, gnames, targets, alpha=1.0, beta=args.beta)  # no proxy
    tw = threeway(ef_d1, ef_d2, ef_px, gallery, gnames, targets, alpha=args.alpha, beta=args.beta)

    sp("="*70)
    sp(f"Prompts: D1={args.d1_prompt}  D2={args.d2_prompt}")
    sp(f"D1 only (baseline):     {base}")
    sp(f"Ensemble only (β={args.beta}): {ens_only}")
    sp(f"3-way (α={args.alpha} β={args.beta}): {tw}")

    # feature cache for grid search
    feat_pkl = PROJECT_ROOT / "precomputed_cache" / "eval_features" / f"{args.experiment_name}_eval_features.pkl"
    feat_pkl.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({
        "d1_features": ef_d1.detach().cpu(),
        "d2_features": ef_d2.detach().cpu(),
        "proxy_features": ef_px.detach().cpu(),
        "meta": meta,
    }, open(feat_pkl, "wb"))
    sp(f"Saved features: {feat_pkl}")

    summary = {
        "config": vars(args),
        "n_valid": len(meta),
        "baseline_d1": base,
        "ensemble_only": ens_only,
        "threeway": tw,
    }
    json.dump(summary, open(summary_json, "w"), ensure_ascii=False, indent=2)
    sp(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
