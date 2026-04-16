#!/usr/bin/env python3
"""
GPT-4o vs Qwen-VL-Max Comparison on FashionIQ dress (200 samples).

Full pipeline: D₁ generation + Proxy image + D₂ refinement + CLIP evaluation.
"""
import argparse
import json
import os
import sys
import time
import random
import base64
import pickle
import urllib.request
import urllib.error
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
import open_clip
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from refine_prompts import V7_ANTI_HALLUCINATION
from prompts import mllm_structural_predictor_prompt_CoT

# ======================== Config ========================
GPT4O_API_KEY = os.getenv("GPT4O_API_KEY", "")
GPT4O_API_BASE = os.getenv("GPT4O_API_BASE", "https://api.zhizengzeng.com/v1")
GPT4O_MODEL = os.getenv("GPT4O_MODEL", "gpt-4o-2024-08-06")

MINIMAX_KEY = os.getenv("MINIMAX_API_KEY", "")

PROXY_LIST = [
    "10.66.28.231:11080",
    "10.68.41.218:11080",
    "10.68.24.160:11080",
    "10.68.8.218:11080",
    "10.66.29.113:11080",
    "10.66.72.150:11080",
    "10.66.37.111:11080",
    "10.66.22.211:11080",
    "10.68.24.18:11080",
    "10.68.25.38:11080",
    "10.68.58.44:11080",
]
_proxy_idx = 0
_proxy_lock = threading.Lock()

def _next_proxy():
    """Round-robin proxy selection for load balancing."""
    global _proxy_idx
    with _proxy_lock:
        proxy = PROXY_LIST[_proxy_idx % len(PROXY_LIST)]
        _proxy_idx += 1
    return proxy

NUM_SAMPLES = 200
SEED = 42
WORKERS = 4

IMAGE_CACHE_DIR = os.path.join(PROJECT_ROOT, "cache_fiq_images")
GPT4O_CACHE_DIR = os.path.join(PROJECT_ROOT, "outputs", "gpt4o_comparison")
PROXY_DIR = os.path.join(GPT4O_CACHE_DIR, "proxy_images")

print_lock = threading.Lock()


def safe_print(msg):
    with print_lock:
        print(msg, flush=True)


# ======================== Image Download ========================
def download_fiq_image(asin, save_dir):
    """Download a FashionIQ image from Amazon by ASIN."""
    save_path = os.path.join(save_dir, f"{asin}.jpg")
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return save_path

    url = f"https://m.media-amazon.com/images/P/{asin}.01._SCLZZZZZZZ_.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = resp.read()
        if len(data) < 500:
            return None
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(data)
        return save_path
    except Exception:
        return None


def ensure_images(samples, save_dir):
    """Download all needed reference images, return success count."""
    os.makedirs(save_dir, exist_ok=True)
    asins = [s["reference_name"] for s in samples]
    success = 0
    fail_list = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_fiq_image, asin, save_dir): asin for asin in asins}
        for i, future in enumerate(as_completed(futures)):
            asin = futures[future]
            result = future.result()
            if result:
                success += 1
            else:
                fail_list.append(asin)
            if (i + 1) % 50 == 0:
                safe_print(f"  Downloaded {i+1}/{len(asins)} images...")

    print(f"  Images: {success}/{len(asins)} downloaded, {len(fail_list)} failed")
    if fail_list:
        print(f"  Failed ASINs (first 10): {fail_list[:10]}")
    return success, fail_list


# ======================== GPT-4o API ========================
def gpt4o_chat_completion(messages, max_tokens=1024, temperature=0):
    """Call GPT-4o via zhizengzeng, with round-robin proxy for speed."""
    import urllib.request
    import json as _json
    if not GPT4O_API_KEY:
        raise RuntimeError("Missing GPT4O_API_KEY environment variable.")

    url = f"{GPT4O_API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GPT4O_API_KEY}",
    }
    body = {
        "model": GPT4O_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    proxy_addr = _next_proxy()
    proxy_handler = urllib.request.ProxyHandler({
        "https": f"http://{proxy_addr}",
        "http": f"http://{proxy_addr}",
    })
    opener = urllib.request.build_opener(proxy_handler)
    resp = opener.open(req, timeout=120)
    result = _json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def encode_image_b64(image_path):
    """Encode image to base64 data URL."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ======================== D₁ Generation ========================
def generate_d1(image_path, instruction, max_retries=3):
    """Generate D₁ (initial description) using GPT-4o with the paper's CoT prompt."""
    image_url = encode_image_b64(image_path)
    user_prompt = f'''<Input>\n{{\n    "Original Image": <image_url>,\n    "Manipulation text": "{instruction}"\n}}'''

    messages = [
        {"role": "system", "content": mllm_structural_predictor_prompt_CoT},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]},
    ]

    for attempt in range(max_retries):
        try:
            raw = gpt4o_chat_completion(messages, max_tokens=4096)
            return raw
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                safe_print(f"  D₁ FAIL: {e}")
                return ""


def parse_d1_response(raw_response):
    """Extract target description from D₁ response."""
    resp = raw_response.strip()
    if resp.startswith("```json"):
        resp = resp.replace("```json", "").replace("```", "").strip()
    if resp.startswith("```"):
        resp = resp.replace("```", "").strip()
    try:
        d = json.loads(resp)
        return d.get("Target Image Description", "")
    except Exception:
        for line in resp.split("\n"):
            if "Target Image Description" in line and ":" in line:
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return ""


# ======================== Proxy Generation ========================
def generate_proxy_image(prompt, save_path, max_retries=3):
    """Generate proxy image via MiniMax."""
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return save_path
    if not MINIMAX_KEY:
        raise RuntimeError("Missing MINIMAX_API_KEY environment variable.")

    import requests
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.minimax.chat/v1/image_generation",
                headers={"Authorization": f"Bearer {MINIMAX_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "image-01", "prompt": prompt,
                      "aspect_ratio": "1:1", "response_format": "url", "n": 1},
                timeout=60)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}")

            result = resp.json()
            status = result.get("base_resp", {}).get("status_code", -1)
            if status == 1026:
                return "SENSITIVE"
            if status != 0:
                if status == 1002:
                    time.sleep(5 + attempt * 3)
                    continue
                raise Exception(f"API {status}")

            img_url = result["data"]["image_urls"][0]
            img_data = requests.get(img_url, timeout=20).content
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(img_data)
            return save_path
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return None
    return None


# ======================== D₂ Refinement ========================
def generate_d2(ref_image_path, proxy_image_path, instruction, max_retries=3):
    """Generate D₂ (refined description) using GPT-4o with V7 anti-hallucination prompt."""
    ref_url = encode_image_b64(ref_image_path)
    proxy_url = encode_image_b64(proxy_image_path)

    user_prompt = f'''<Input>\n{{\n    "Original Image": <image_1>,\n    "Proxy Image": <image_2>,\n    "Manipulation text": "{instruction}"\n}}'''

    messages = [
        {"role": "system", "content": V7_ANTI_HALLUCINATION},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": ref_url}},
            {"type": "image_url", "image_url": {"url": proxy_url}},
        ]},
    ]

    for attempt in range(max_retries):
        try:
            raw = gpt4o_chat_completion(messages, max_tokens=1024)
            return raw
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                safe_print(f"  D₂ FAIL: {e}")
                return ""


def parse_d2_response(raw_response, fallback=""):
    """Extract target description from D₂ V7 response."""
    resp = raw_response.strip()
    if resp.startswith("```json"):
        resp = resp.replace("```json", "").replace("```", "").strip()
    if resp.startswith("```"):
        resp = resp.replace("```", "").strip()
    try:
        d = json.loads(resp)
        return d.get("Target Image Description", fallback)
    except Exception:
        for line in resp.split("\n"):
            if "Target Image Description" in line and ":" in line:
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return fallback


# ======================== CLIP Evaluation ========================
def load_clip_model(device):
    """Load open_clip ViT-L/14 (openai pretrained)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=32):
    """Encode texts to CLIP features."""
    all_feats = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i+batch_size]).to(device)
        feats = model.encode_text(tokens).float().cpu()
        all_feats.append(feats)
    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def encode_images(model, preprocess, image_paths, device, batch_size=16):
    """Encode images to CLIP features."""
    from PIL import Image
    all_feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                imgs.append(img)
            except Exception:
                imgs.append(torch.zeros(1, 3, 224, 224))
        batch = torch.cat(imgs, dim=0).to(device)
        feats = model.encode_image(batch).float().cpu()
        all_feats.append(feats)
    return torch.cat(all_feats, dim=0)


def compute_metrics(query_features, gallery_features, gallery_names, target_names):
    """Compute R@1/5/10/50."""
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


def compute_threeway_metrics(d1_feat, d2_feat, proxy_feat, gallery_feat, gallery_names,
                              target_names, alpha=0.9, beta=0.7):
    """Compute three-way fusion metrics."""
    text_feat = torch.nn.functional.normalize(
        beta * torch.nn.functional.normalize(d1_feat, dim=-1)
        + (1 - beta) * torch.nn.functional.normalize(d2_feat, dim=-1),
        dim=-1)
    proxy_feat_n = torch.nn.functional.normalize(proxy_feat, dim=-1)
    gallery_feat_n = torch.nn.functional.normalize(gallery_feat, dim=-1)

    text_sim = text_feat @ gallery_feat_n.T
    proxy_sim = proxy_feat_n @ gallery_feat_n.T
    combined_sim = alpha * text_sim + (1 - alpha) * proxy_sim

    metrics = {"R@1": 0, "R@5": 0, "R@10": 0, "R@50": 0}
    for i, target in enumerate(target_names):
        if target not in gallery_names:
            continue
        target_idx = gallery_names.index(target)
        sorted_indices = torch.argsort(combined_sim[i], descending=True).tolist()
        rank = sorted_indices.index(target_idx) + 1
        for k in [1, 5, 10, 50]:
            if rank <= k:
                metrics[f"R@{k}"] += 1

    n = len(target_names)
    return {k: v / n * 100 for k, v in metrics.items()}


# ======================== Main Pipeline ========================
def main():
    os.makedirs(GPT4O_CACHE_DIR, exist_ok=True)
    os.makedirs(PROXY_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  GPT-4o vs Qwen-VL-Max Comparison — FashionIQ Dress")
    print(f"  Samples: {NUM_SAMPLES}, Seed: {SEED}, Device: {device}")
    print(f"  GPT-4o model: {GPT4O_MODEL}")
    print(f"{'='*70}\n")

    # Load baseline data (Qwen results)
    baseline_path = os.path.join(PROJECT_ROOT, "outputs", "fashioniq_dress_full.json")
    all_samples = json.load(open(baseline_path))
    print(f"Loaded {len(all_samples)} baseline samples")

    # Step 0: find which samples have images available
    print(f"\n{'='*70}")
    print(f"  Step 0: Checking available FashionIQ reference images")
    print(f"{'='*70}")
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

    available_indices = []
    for i, s in enumerate(all_samples):
        img_path = os.path.join(IMAGE_CACHE_DIR, f"{s['reference_name']}.jpg")
        if os.path.exists(img_path) and os.path.getsize(img_path) > 500:
            available_indices.append(i)
    print(f"  Images on disk: {len(available_indices)}/{len(all_samples)}")

    if len(available_indices) < NUM_SAMPLES:
        print("  Downloading missing images...")
        success_count, _ = ensure_images(all_samples, IMAGE_CACHE_DIR)
        available_indices = []
        for i, s in enumerate(all_samples):
            img_path = os.path.join(IMAGE_CACHE_DIR, f"{s['reference_name']}.jpg")
            if os.path.exists(img_path) and os.path.getsize(img_path) > 500:
                available_indices.append(i)
        print(f"  Available after download: {len(available_indices)}")

    # Select 200 from available samples
    random.seed(SEED)
    indices = random.sample(available_indices, min(NUM_SAMPLES, len(available_indices)))
    indices.sort()
    samples = [all_samples[i] for i in indices]
    print(f"  Selected {len(samples)} samples (seed={SEED})")

    # Cache file for GPT-4o results
    cache_path = os.path.join(GPT4O_CACHE_DIR, "gpt4o_fiq_dress_200.json")
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path))
        print(f"  Loaded {len(cached)} cached GPT-4o results")
    else:
        cached = {}

    # ===== Step 1: Generate D₁ with GPT-4o (parallel) =====
    print(f"\n{'='*70}")
    print(f"  Step 1: Generating D₁ with GPT-4o (parallel, {len(PROXY_LIST)} proxies)")
    print(f"{'='*70}")

    todo_d1 = []
    for i, sample in enumerate(samples):
        key = str(indices[i])
        if key in cached and cached[key].get("gpt4o_d1"):
            continue
        asin = sample["reference_name"]
        img_path = os.path.join(IMAGE_CACHE_DIR, f"{asin}.jpg")
        if not os.path.exists(img_path):
            continue
        todo_d1.append((i, key, asin, sample["instruction"], img_path))

    print(f"  Cached: {len(samples) - len(todo_d1)}, to generate: {len(todo_d1)}")
    d1_done = [0]
    d1_lock = threading.Lock()
    start_time = time.time()

    def _do_d1(args):
        i, key, asin, instruction, img_path = args
        raw = generate_d1(img_path, instruction)
        desc = parse_d1_response(raw)
        with d1_lock:
            if key not in cached:
                cached[key] = {"index": indices[i], "reference_name": asin, "instruction": instruction}
            cached[key]["gpt4o_d1_raw"] = raw
            cached[key]["gpt4o_d1"] = desc
            d1_done[0] += 1
            if d1_done[0] % 10 == 0:
                rate = d1_done[0] / (time.time() - start_time)
                safe_print(f"  D₁ done={d1_done[0]}/{len(todo_d1)} rate={rate:.2f}/s")
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cached, f, ensure_ascii=False, indent=2)

    if todo_d1:
        with ThreadPoolExecutor(max_workers=min(6, len(PROXY_LIST))) as ex:
            list(ex.map(_do_d1, todo_d1))
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached, f, ensure_ascii=False, indent=2)

    d1_total = sum(1 for i in range(len(samples))
                   if cached.get(str(indices[i]), {}).get("gpt4o_d1"))
    print(f"  D₁ complete: {d1_total}/{len(samples)}")

    # ===== Step 2: Generate proxy images =====
    print(f"\n{'='*70}")
    print(f"  Step 2: Generating proxy images from GPT-4o D₁")
    print(f"{'='*70}")
    proxy_count = 0
    proxy_fail = 0
    for i, sample in enumerate(samples):
        key = str(indices[i])
        d1 = cached.get(key, {}).get("gpt4o_d1", "")
        if not d1:
            continue

        asin = sample["reference_name"]
        save_path = os.path.join(PROXY_DIR, f"proxy_{asin}.jpg")
        if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
            proxy_count += 1
            continue

        result = generate_proxy_image(d1, save_path)
        if result and result != "SENSITIVE":
            proxy_count += 1
        else:
            proxy_fail += 1

        if (i + 1) % 20 == 0:
            safe_print(f"  [{i+1}/{len(samples)}] proxy={proxy_count} fail={proxy_fail}")

    print(f"  Proxy complete: {proxy_count} generated, {proxy_fail} failed")

    # ===== Step 3: Generate D₂ with GPT-4o (parallel) =====
    print(f"\n{'='*70}")
    print(f"  Step 3: Generating D₂ with GPT-4o (parallel, V7 anti-hallucination)")
    print(f"{'='*70}")

    todo_d2 = []
    for i, sample in enumerate(samples):
        key = str(indices[i])
        if key in cached and cached[key].get("gpt4o_d2"):
            continue
        d1 = cached.get(key, {}).get("gpt4o_d1", "")
        if not d1:
            continue
        asin = sample["reference_name"]
        img_path = os.path.join(IMAGE_CACHE_DIR, f"{asin}.jpg")
        proxy_path = os.path.join(PROXY_DIR, f"proxy_{asin}.jpg")
        if not os.path.exists(img_path) or not os.path.exists(proxy_path):
            continue
        todo_d2.append((i, key, asin, sample["instruction"], img_path, proxy_path, d1))

    d2_cached = len(samples) - len(todo_d2)
    print(f"  Cached: {d2_cached}, to generate: {len(todo_d2)}")
    d2_done = [0]
    d2_lock = threading.Lock()
    start_time = time.time()

    def _do_d2(args):
        i, key, asin, instruction, img_path, proxy_path, d1_fallback = args
        raw = generate_d2(img_path, proxy_path, instruction)
        desc = parse_d2_response(raw, fallback=d1_fallback)
        with d2_lock:
            cached[key]["gpt4o_d2_raw"] = raw
            cached[key]["gpt4o_d2"] = desc
            d2_done[0] += 1
            if d2_done[0] % 10 == 0:
                rate = d2_done[0] / (time.time() - start_time)
                safe_print(f"  D₂ done={d2_done[0]}/{len(todo_d2)} rate={rate:.2f}/s")
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cached, f, ensure_ascii=False, indent=2)

    if todo_d2:
        with ThreadPoolExecutor(max_workers=min(6, len(PROXY_LIST))) as ex:
            list(ex.map(_do_d2, todo_d2))
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached, f, ensure_ascii=False, indent=2)

    d2_total = sum(1 for i, s in enumerate(samples)
                   if cached.get(str(indices[i]), {}).get("gpt4o_d2"))
    print(f"  D₂ complete: {d2_total}/{len(samples)}")

    # ===== Step 4: CLIP Evaluation =====
    print(f"\n{'='*70}")
    print(f"  Step 4: CLIP Evaluation")
    print(f"{'='*70}")

    print("  Loading CLIP ViT-L/14...")
    clip_model, preprocess, tokenizer = load_clip_model(device)

    # Load precomputed gallery features
    gallery_path = os.path.join(PROJECT_ROOT, "precomputed_cache", "precomputed",
                                "fashioniq_dress_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl")
    gallery_data = pickle.load(open(gallery_path, "rb"))
    gallery_features = gallery_data["index_features"].to(device)
    gallery_names = gallery_data["index_names"]
    print(f"  Gallery: {gallery_features.shape[0]} images")

    # Get target names for our 200 samples
    target_names = [s["target_name"] for s in samples]

    # --- Qwen features (from precomputed) ---
    eval_feat_path = os.path.join(PROJECT_ROOT, "precomputed_cache", "eval_features",
                                  "fashioniq_dress_eval_features.pkl")
    eval_feats = pickle.load(open(eval_feat_path, "rb"))
    qwen_d1_all = eval_feats["d1_features"]
    qwen_d2_all = eval_feats["d2_features"]
    qwen_proxy_all = eval_feats["proxy_features"]
    qwen_d1 = qwen_d1_all[indices].to(device)
    qwen_d2 = qwen_d2_all[indices].to(device)
    qwen_proxy = qwen_proxy_all[indices].to(device)
    print(f"  Qwen features loaded (precomputed)")

    # --- GPT-4o features (encode fresh) ---
    gpt4o_d1_texts = []
    gpt4o_d2_texts = []
    gpt4o_proxy_paths = []
    valid_indices = []

    for i, sample in enumerate(samples):
        key = str(indices[i])
        d1 = cached.get(key, {}).get("gpt4o_d1", "")
        d2 = cached.get(key, {}).get("gpt4o_d2", "")
        proxy_path = os.path.join(PROXY_DIR, f"proxy_{sample['reference_name']}.jpg")

        if d1 and d2 and os.path.exists(proxy_path):
            gpt4o_d1_texts.append(d1)
            gpt4o_d2_texts.append(d2)
            gpt4o_proxy_paths.append(proxy_path)
            valid_indices.append(i)

    print(f"  Valid GPT-4o samples: {len(valid_indices)}/{len(samples)}")

    print("  Encoding GPT-4o D₁ texts...")
    gpt4o_d1_feat = encode_texts(clip_model, tokenizer, gpt4o_d1_texts, device)
    print("  Encoding GPT-4o D₂ texts...")
    gpt4o_d2_feat = encode_texts(clip_model, tokenizer, gpt4o_d2_texts, device)
    print("  Encoding GPT-4o proxy images...")
    gpt4o_proxy_feat = encode_images(clip_model, preprocess, gpt4o_proxy_paths, device)

    # Move to device
    gpt4o_d1_feat = gpt4o_d1_feat.to(device)
    gpt4o_d2_feat = gpt4o_d2_feat.to(device)
    gpt4o_proxy_feat = gpt4o_proxy_feat.to(device)

    # Filter Qwen features to match valid GPT-4o indices
    qwen_d1_valid = qwen_d1[valid_indices]
    qwen_d2_valid = qwen_d2[valid_indices]
    qwen_proxy_valid = qwen_proxy[valid_indices]
    valid_targets = [target_names[i] for i in valid_indices]

    gallery_features_d = gallery_features.to(device)

    # ===== Compute all metrics =====
    print("\n  Computing metrics...")

    # Qwen Baseline (D₁ only)
    qwen_baseline = compute_metrics(qwen_d1_valid, gallery_features_d, gallery_names, valid_targets)
    # Qwen Three-Way
    qwen_3way = compute_threeway_metrics(qwen_d1_valid, qwen_d2_valid, qwen_proxy_valid,
                                          gallery_features_d, gallery_names, valid_targets)

    # GPT-4o Baseline (D₁ only)
    gpt4o_baseline = compute_metrics(gpt4o_d1_feat, gallery_features_d, gallery_names, valid_targets)
    # GPT-4o Three-Way
    gpt4o_3way = compute_threeway_metrics(gpt4o_d1_feat, gpt4o_d2_feat, gpt4o_proxy_feat,
                                           gallery_features_d, gallery_names, valid_targets)

    # ===== Print Results =====
    print(f"\n{'='*70}")
    print(f"  RESULTS: GPT-4o vs Qwen-VL-Max — FashionIQ Dress ({len(valid_indices)} samples)")
    print(f"{'='*70}")
    print(f"\n  {'Method':<30} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8}")
    print(f"  {'-'*62}")
    print(f"  {'Qwen Baseline (D₁ only)':<30} {qwen_baseline['R@1']:>8.2f} {qwen_baseline['R@5']:>8.2f} {qwen_baseline['R@10']:>8.2f} {qwen_baseline['R@50']:>8.2f}")
    print(f"  {'Qwen 3-Way Fusion':<30} {qwen_3way['R@1']:>8.2f} {qwen_3way['R@5']:>8.2f} {qwen_3way['R@10']:>8.2f} {qwen_3way['R@50']:>8.2f}")
    print(f"  {'GPT-4o Baseline (D₁ only)':<30} {gpt4o_baseline['R@1']:>8.2f} {gpt4o_baseline['R@5']:>8.2f} {gpt4o_baseline['R@10']:>8.2f} {gpt4o_baseline['R@50']:>8.2f}")
    print(f"  {'GPT-4o 3-Way Fusion':<30} {gpt4o_3way['R@1']:>8.2f} {gpt4o_3way['R@5']:>8.2f} {gpt4o_3way['R@10']:>8.2f} {gpt4o_3way['R@50']:>8.2f}")

    # Delta rows
    print(f"\n  {'--- Delta vs Qwen Baseline ---'}")
    for name, m in [("Qwen 3-Way", qwen_3way), ("GPT-4o Baseline", gpt4o_baseline), ("GPT-4o 3-Way", gpt4o_3way)]:
        deltas = {k: m[k] - qwen_baseline[k] for k in m}
        signs = {k: "+" if v >= 0 else "" for k, v in deltas.items()}
        print(f"  {'Δ ' + name:<30} {signs['R@1']}{deltas['R@1']:>7.2f} {signs['R@5']}{deltas['R@5']:>7.2f} {signs['R@10']}{deltas['R@10']:>7.2f} {signs['R@50']}{deltas['R@50']:>7.2f}")

    # Save results
    results = {
        "config": {
            "num_samples": len(valid_indices),
            "seed": SEED,
            "gpt4o_model": GPT4O_MODEL,
            "alpha": 0.9,
            "beta": 0.7,
        },
        "qwen_baseline": qwen_baseline,
        "qwen_threeway": qwen_3way,
        "gpt4o_baseline": gpt4o_baseline,
        "gpt4o_threeway": gpt4o_3way,
    }
    results_path = os.path.join(GPT4O_CACHE_DIR, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
