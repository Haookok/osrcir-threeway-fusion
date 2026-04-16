#!/usr/bin/env python3
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

import numpy as np
import open_clip
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cloudgpt_api
import image_generation_api
import prompts
import refine_prompts

print_lock = threading.Lock()


def safe_print(msg: str):
    with print_lock:
        print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser("FashionIQ provider experiment")
    parser.add_argument("--dataset", default="fashioniq_dress",
                        choices=["fashioniq_dress", "fashioniq_shirt", "fashioniq_toptee"])
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--mllm_model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--mllm_api_base", type=str, default=os.getenv("OPENAI_COMPAT_API_BASE", ""))
    parser.add_argument("--mllm_api_key", type=str, default=os.getenv("OPENAI_COMPAT_API_KEY", ""))
    parser.add_argument("--image_backend", type=str, default=os.getenv("IMAGE_API_BACKEND", "minimax"))
    parser.add_argument("--image_api_base", type=str, default=os.getenv("IMAGE_API_BASE", ""))
    parser.add_argument("--image_api_key", type=str, default=os.getenv("IMAGE_API_KEY", ""))
    parser.add_argument("--image_model", type=str, default=os.getenv("IMAGE_API_MODEL", "image-01"))
    parser.add_argument("--d1_prompt", type=str, default="fashioniq_garment_only_prompt")
    parser.add_argument("--d2_prompt", type=str, default="V7_FASHIONIQ_GARMENT_ONLY")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--d1_workers", type=int, default=4)
    parser.add_argument("--d2_workers", type=int, default=4)
    parser.add_argument("--proxy_workers", type=int, default=3)
    parser.add_argument("--max_tokens_d1", type=int, default=512)
    parser.add_argument("--max_tokens_d2", type=int, default=512)
    parser.add_argument("--reuse_from_experiment", type=str, default=None)
    parser.add_argument("--reuse_d1", action="store_true")
    parser.add_argument("--reuse_proxy", action="store_true")
    parser.add_argument("--force_rerun_d2", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_json_text(raw_response: str):
    resp = (raw_response or "").strip()
    if resp.startswith("<Response>"):
        resp = resp[len("<Response>"):].strip()
    if resp.startswith("```json"):
        resp = resp.replace("```json", "", 1).strip()
    if resp.startswith("```"):
        resp = resp.replace("```", "", 1).strip()
    if resp.endswith("```"):
        resp = resp[:-3].strip()
    try:
        return json.loads(resp)
    except Exception:
        return None


def extract_target_description(raw_response: str, fallback: str = "") -> str:
    parsed = parse_json_text(raw_response)
    if isinstance(parsed, dict):
        for key in ["Target Image Description", "Refined Target Description"]:
            value = parsed.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()

    for line in (raw_response or "").splitlines():
        if "Target Image Description" in line and ":" in line:
            return line.split(":", 1)[1].strip().strip('"').strip("'")
        if "Refined Target Description" in line and ":" in line:
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return fallback


def build_single_image_messages(system_prompt: str, instruction: str, image_path: str):
    image_url = cloudgpt_api.encode_image(image_path)
    user_prompt = (
        "<Input>\n{\n"
        '    "Original Image": <image_url>,\n'
        f'    "Manipulation text": "{instruction}"\n'
        "}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]},
    ]


def build_dual_image_messages(system_prompt: str, instruction: str, ref_image_path: str, proxy_path: str):
    ref_url = cloudgpt_api.encode_image(ref_image_path)
    proxy_url = cloudgpt_api.encode_image(proxy_path)
    user_prompt = (
        "<Input>\n{\n"
        '    "Original Image": <image_1>,\n'
        '    "Proxy Image": <image_2>,\n'
        f'    "Manipulation text": "{instruction}"\n'
        "}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": ref_url}},
            {"type": "image_url", "image_url": {"url": proxy_url}},
        ]},
    ]


def call_mllm(messages, model: str, api_key: str, api_base: str, max_tokens: int, retries: int = 3):
    for attempt in range(retries):
        try:
            resp = cloudgpt_api.get_chat_completion(
                engine=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=120,
                temperature=0,
                api_key=api_key,
                api_base=api_base,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                safe_print(f"[FAIL] MLLM call failed after {retries} tries: {e}")
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def resolve_reference_image_path(sample):
    path = sample.get("reference_image_path", "")
    if not path:
        return ""
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return str(candidate)
    return str(candidate)


def get_experiment_paths(args):
    out_dir = PROJECT_ROOT / "outputs" / "provider_experiments" / args.experiment_name
    proxy_dir = out_dir / "proxy_images"
    cache_json = out_dir / "sample_cache.json"
    summary_json = out_dir / "comparison_results.json"
    feature_pkl = PROJECT_ROOT / "precomputed_cache" / "eval_features" / f"{args.experiment_name}_eval_features.pkl"
    return out_dir, proxy_dir, cache_json, summary_json, feature_pkl


def maybe_reuse_previous_artifacts(args, cache, proxy_dir):
    if not args.reuse_from_experiment:
        return cache, None

    prev_dir = PROJECT_ROOT / "outputs" / "provider_experiments" / args.reuse_from_experiment
    prev_cache_path = prev_dir / "sample_cache.json"
    prev_proxy_dir = prev_dir / "proxy_images"
    prev_cache = load_json(prev_cache_path) if prev_cache_path.exists() else {}

    for key, entry in prev_cache.items():
        cache.setdefault(key, {})
        for field in ["index", "reference_name", "reference_image_path", "instruction", "target_name"]:
            if field in entry and field not in cache[key]:
                cache[key][field] = entry[field]
        if args.reuse_d1:
            for field in ["d1_raw", "d1"]:
                if field in entry and field not in cache[key]:
                    cache[key][field] = entry[field]
        if args.reuse_proxy:
            src_proxy = prev_proxy_dir / f"proxy_{int(key):05d}.jpg"
            dst_proxy = proxy_dir / f"proxy_{int(key):05d}.jpg"
            if src_proxy.exists() and not dst_proxy.exists():
                try:
                    os.symlink(src_proxy, dst_proxy)
                except FileExistsError:
                    pass
                except OSError:
                    pass
            if dst_proxy.exists():
                cache[key]["proxy_status"] = str(dst_proxy)
    return cache, prev_dir


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
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=32):
    outputs = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i + batch_size]).to(device)
        outputs.append(model.encode_text(tokens).float().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_images(model, preprocess, image_paths, device, batch_size=16):
    from PIL import Image

    outputs = []
    dummy = torch.zeros(3, 224, 224)
    for i in range(0, len(image_paths), batch_size):
        batch = []
        for image_path in image_paths[i:i + batch_size]:
            try:
                batch.append(preprocess(Image.open(image_path).convert("RGB")))
            except Exception:
                batch.append(dummy)
        tensor = torch.stack(batch).to(device)
        outputs.append(model.encode_image(tensor).float().cpu())
    return torch.cat(outputs, dim=0)


def compute_metrics(query_features, gallery_features, gallery_names, target_names):
    query_features = torch.nn.functional.normalize(query_features, dim=-1)
    gallery_features = torch.nn.functional.normalize(gallery_features, dim=-1)
    sims = query_features @ gallery_features.T

    metrics = {"R@1": 0, "R@5": 0, "R@10": 0, "R@50": 0}
    for i, target in enumerate(target_names):
        if target not in gallery_names:
            continue
        target_idx = gallery_names.index(target)
        sorted_indices = torch.argsort(sims[i], descending=True).tolist()
        rank = sorted_indices.index(target_idx) + 1
        for k in [1, 5, 10, 50]:
            if rank <= k:
                metrics[f"R@{k}"] += 1

    n = len(target_names)
    return {k: v / n * 100 for k, v in metrics.items()}


def compute_threeway_metrics(d1_feat, d2_feat, proxy_feat, gallery_feat, gallery_names, target_names,
                             alpha=0.9, beta=0.7):
    text_feat = torch.nn.functional.normalize(
        beta * torch.nn.functional.normalize(d1_feat, dim=-1)
        + (1 - beta) * torch.nn.functional.normalize(d2_feat, dim=-1),
        dim=-1,
    )
    proxy_feat = torch.nn.functional.normalize(proxy_feat, dim=-1)
    gallery_feat = torch.nn.functional.normalize(gallery_feat, dim=-1)

    sims = alpha * (text_feat @ gallery_feat.T) + (1 - alpha) * (proxy_feat @ gallery_feat.T)
    metrics = {"R@1": 0, "R@5": 0, "R@10": 0, "R@50": 0}
    for i, target in enumerate(target_names):
        if target not in gallery_names:
            continue
        target_idx = gallery_names.index(target)
        sorted_indices = torch.argsort(sims[i], descending=True).tolist()
        rank = sorted_indices.index(target_idx) + 1
        for k in [1, 5, 10, 50]:
            if rank <= k:
                metrics[f"R@{k}"] += 1
    n = len(target_names)
    return {k: v / n * 100 for k, v in metrics.items()}


def main():
    args = parse_args()
    if not args.mllm_api_key:
        raise RuntimeError("Missing MLLM API key. Use --mllm_api_key or OPENAI_COMPAT_API_KEY.")

    d1_prompt = getattr(prompts, args.d1_prompt)
    d2_prompt = getattr(refine_prompts, args.d2_prompt)
    out_dir, proxy_dir, cache_json, summary_json, feature_pkl = get_experiment_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    proxy_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = PROJECT_ROOT / "outputs" / f"{args.dataset}_full.json"
    if args.dataset == "fashioniq_shirt" or args.dataset == "fashioniq_toptee":
        baseline_path = PROJECT_ROOT / "outputs" / f"{args.dataset}_full.json"
    all_samples = load_json(baseline_path)

    random.seed(args.seed)
    sample_indices = sorted(random.sample(range(len(all_samples)), args.num_samples))
    samples = [all_samples[i] for i in sample_indices]

    safe_print(f"Experiment: {args.experiment_name}")
    safe_print(f"Dataset: {args.dataset}, samples={len(samples)}, seed={args.seed}")
    safe_print(f"MLLM: {args.mllm_model} @ {args.mllm_api_base}")
    safe_print(f"Image backend: {args.image_backend}, model={args.image_model}")
    safe_print(f"D1 prompt: {args.d1_prompt}, D2 prompt: {args.d2_prompt}")

    cache = load_json(cache_json) if cache_json.exists() else {}
    cache, prev_dir = maybe_reuse_previous_artifacts(args, cache, proxy_dir)
    if prev_dir:
        safe_print(f"Reusing cache from: {prev_dir}")
        save_json(cache_json, cache)

    # Step 1: D1 generation
    todo_d1 = []
    for idx, sample in zip(sample_indices, samples):
        key = str(idx)
        ref_path = resolve_reference_image_path(sample)
        if key in cache and cache[key].get("d1"):
            continue
        if not Path(ref_path).exists():
            continue
        todo_d1.append((idx, sample, ref_path))

    safe_print(f"D1 cached={len(samples) - len(todo_d1)}, to_generate={len(todo_d1)}")

    def _run_d1(task):
        idx, sample, ref_path = task
        messages = build_single_image_messages(d1_prompt, sample["instruction"], ref_path)
        raw = call_mllm(messages, args.mllm_model, args.mllm_api_key, args.mllm_api_base, args.max_tokens_d1)
        desc = extract_target_description(raw, fallback="")
        return idx, sample, ref_path, raw, desc

    if todo_d1:
        with ThreadPoolExecutor(max_workers=args.d1_workers) as ex:
            futures = [ex.submit(_run_d1, task) for task in todo_d1]
            for i, future in enumerate(as_completed(futures), start=1):
                idx, sample, ref_path, raw, desc = future.result()
                cache[str(idx)] = {
                    "index": idx,
                    "reference_name": sample["reference_name"],
                    "reference_image_path": ref_path,
                    "instruction": sample["instruction"],
                    "target_name": sample["target_name"],
                    "d1_raw": raw,
                    "d1": desc,
                }
                if i % 10 == 0 or i == len(futures):
                    safe_print(f"  D1 {i}/{len(futures)}")
                    save_json(cache_json, cache)
        save_json(cache_json, cache)

    # Step 2: proxy generation
    todo_proxy = []
    for idx, sample in zip(sample_indices, samples):
        key = str(idx)
        d1 = cache.get(key, {}).get("d1", "")
        proxy_path = proxy_dir / f"proxy_{idx:05d}.jpg"
        if proxy_path.exists():
            continue
        if d1:
            todo_proxy.append((idx, d1, proxy_path))

    safe_print(f"Proxy cached={len(samples) - len(todo_proxy)}, to_generate={len(todo_proxy)}")

    def _run_proxy(task):
        idx, d1, proxy_path = task
        result = image_generation_api.generate_image(
            d1,
            str(proxy_path),
            backend=args.image_backend,
            api_key=args.image_api_key,
            api_base=args.image_api_base,
            model=args.image_model,
        )
        return idx, result

    if todo_proxy:
        with ThreadPoolExecutor(max_workers=args.proxy_workers) as ex:
            futures = [ex.submit(_run_proxy, task) for task in todo_proxy]
            for i, future in enumerate(as_completed(futures), start=1):
                idx, result = future.result()
                cache.setdefault(str(idx), {})["proxy_status"] = result or "FAILED"
                if i % 10 == 0 or i == len(futures):
                    safe_print(f"  Proxy {i}/{len(futures)}")
                    save_json(cache_json, cache)
        save_json(cache_json, cache)

    # Step 3: D2 generation
    todo_d2 = []
    for idx, sample in zip(sample_indices, samples):
        key = str(idx)
        ref_path = resolve_reference_image_path(sample)
        proxy_path = proxy_dir / f"proxy_{idx:05d}.jpg"
        if (not args.force_rerun_d2) and key in cache and cache[key].get("d2"):
            continue
        if cache.get(key, {}).get("d1") and Path(ref_path).exists() and proxy_path.exists():
            todo_d2.append((idx, sample, ref_path, str(proxy_path)))

    safe_print(f"D2 cached={len(samples) - len(todo_d2)}, to_generate={len(todo_d2)}")

    def _run_d2(task):
        idx, sample, ref_path, proxy_path = task
        messages = build_dual_image_messages(d2_prompt, sample["instruction"], ref_path, proxy_path)
        raw = call_mllm(messages, args.mllm_model, args.mllm_api_key, args.mllm_api_base, args.max_tokens_d2)
        desc = extract_target_description(raw, fallback=cache.get(str(idx), {}).get("d1", ""))
        return idx, raw, desc

    if todo_d2:
        with ThreadPoolExecutor(max_workers=args.d2_workers) as ex:
            futures = [ex.submit(_run_d2, task) for task in todo_d2]
            for i, future in enumerate(as_completed(futures), start=1):
                idx, raw, desc = future.result()
                cache.setdefault(str(idx), {})["d2_raw"] = raw
                cache.setdefault(str(idx), {})["d2"] = desc
                if i % 10 == 0 or i == len(futures):
                    safe_print(f"  D2 {i}/{len(futures)}")
                    save_json(cache_json, cache)
        save_json(cache_json, cache)

    # Step 4: evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess, tokenizer = load_clip_model(device)

    gallery_path = PROJECT_ROOT / "precomputed_cache" / "precomputed" / f"{args.dataset}_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl"
    gallery_data = pickle.load(open(gallery_path, "rb"))
    gallery_features = gallery_data["index_features"].to(device)
    gallery_names = gallery_data["index_names"]

    qwen_eval_path = PROJECT_ROOT / "precomputed_cache" / "eval_features" / f"{args.dataset}_eval_features.pkl"
    qwen_feats = pickle.load(open(qwen_eval_path, "rb"))
    qwen_d1 = qwen_feats["d1_features"][sample_indices].to(device)
    qwen_d2 = qwen_feats["d2_features"][sample_indices].to(device)
    qwen_proxy = qwen_feats["proxy_features"][sample_indices].to(device)

    valid_sample_indices = []
    d1_texts = []
    d2_texts = []
    proxy_paths = []
    meta = []
    for idx, sample in zip(sample_indices, samples):
        entry = cache.get(str(idx), {})
        proxy_path = proxy_dir / f"proxy_{idx:05d}.jpg"
        if entry.get("d1") and entry.get("d2") and proxy_path.exists():
            valid_sample_indices.append(idx)
            d1_texts.append(entry["d1"])
            d2_texts.append(entry["d2"])
            proxy_paths.append(str(proxy_path))
            meta.append({
                "index": idx,
                "target_name": sample["target_name"],
                "reference_name": sample["reference_name"],
                "instruction": sample["instruction"],
            })

    subset_positions = [sample_indices.index(idx) for idx in valid_sample_indices]
    qwen_d1 = qwen_d1[subset_positions]
    qwen_d2 = qwen_d2[subset_positions]
    qwen_proxy = qwen_proxy[subset_positions]
    target_names = [m["target_name"] for m in meta]

    exp_d1 = encode_texts(clip_model, tokenizer, d1_texts, device).to(device)
    exp_d2 = encode_texts(clip_model, tokenizer, d2_texts, device).to(device)
    exp_proxy = encode_images(clip_model, preprocess, proxy_paths, device).to(device)

    qwen_baseline = compute_metrics(qwen_d1, gallery_features, gallery_names, target_names)
    qwen_threeway = compute_threeway_metrics(qwen_d1, qwen_d2, qwen_proxy, gallery_features, gallery_names, target_names)
    exp_baseline = compute_metrics(exp_d1, gallery_features, gallery_names, target_names)
    exp_threeway = compute_threeway_metrics(
        exp_d1, exp_d2, exp_proxy, gallery_features, gallery_names, target_names,
        alpha=args.alpha, beta=args.beta,
    )

    summary = {
        "config": {
            "dataset": args.dataset,
            "num_samples_requested": args.num_samples,
            "num_samples_valid": len(meta),
            "seed": args.seed,
            "mllm_model": args.mllm_model,
            "mllm_api_base": args.mllm_api_base,
            "image_backend": args.image_backend,
            "image_model": args.image_model,
            "d1_prompt": args.d1_prompt,
            "d2_prompt": args.d2_prompt,
            "alpha": args.alpha,
            "beta": args.beta,
            "sample_indices": valid_sample_indices,
            "feature_pkl": str(feature_pkl),
            "gallery_pkl": str(gallery_path),
        },
        "qwen_baseline": qwen_baseline,
        "qwen_threeway": qwen_threeway,
        "experiment_baseline": exp_baseline,
        "experiment_threeway": exp_threeway,
    }
    save_json(summary_json, summary)

    feature_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_pkl, "wb") as f:
        pickle.dump({
            "d1_features": exp_d1.detach().cpu(),
            "d2_features": exp_d2.detach().cpu(),
            "proxy_features": exp_proxy.detach().cpu(),
            "meta": meta,
        }, f)

    safe_print("=" * 72)
    safe_print(f"Results for {args.experiment_name} ({len(meta)} valid samples)")
    safe_print(f"Qwen baseline:    {qwen_baseline}")
    safe_print(f"Qwen three-way:   {qwen_threeway}")
    safe_print(f"Experiment base:  {exp_baseline}")
    safe_print(f"Experiment 3-way: {exp_threeway}")
    safe_print(f"Saved summary:    {summary_json}")
    safe_print(f"Saved features:   {feature_pkl}")
    safe_print("=" * 72)


if __name__ == "__main__":
    main()
