"""
Full-scale Three-Way Fusion Pipeline for OSrCIR (v2 - concurrent).

Runs on the Linux server (API calls only, no GPU needed):
  Step 1: Generate proxy images via MiniMax (concurrent workers, skips cache)
  Step 2: Generate V7 refined descriptions via MLLM (concurrent workers, skips cache)

Usage:
  python run_full_pipeline.py --dataset fashioniq_dress --dataset_path ../datasets/FASHIONIQ
  python run_full_pipeline.py --dataset circo --dataset_path ../datasets/CIRCO
"""
import argparse
import json
import os
import sys
import time
import random
import requests
import base64
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cloudgpt_api
from refine_prompts import V7_ANTI_HALLUCINATION

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROXY_CACHE_DIR = os.path.join(PROJECT_ROOT, 'proxy_cache')
MINIMAX_KEY = os.getenv('MINIMAX_API_KEY', '')

COST_PER_PROXY = 0.025
COST_PER_REFINE = 0.007

PROXY_TIMEOUT = 60
PROXY_DOWNLOAD_TIMEOUT = 20
PROXY_WORKERS = 3
REFINE_WORKERS = 3

print_lock = threading.Lock()


def safe_print(msg):
    with print_lock:
        print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser('Full Pipeline v2')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fashioniq_dress', 'fashioniq_shirt', 'fashioniq_toptee',
                                 'circo', 'cirr',
                                 'genecis_change_object', 'genecis_focus_object',
                                 'genecis_change_attribute', 'genecis_focus_attribute'])
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--baseline_json', type=str, default=None)
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'full_pipeline'))
    parser.add_argument('--skip_proxy', action='store_true')
    parser.add_argument('--skip_refine', action='store_true')
    parser.add_argument('--engine', type=str, default='qwen-vl-max-latest')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--proxy_workers', type=int, default=PROXY_WORKERS)
    parser.add_argument('--refine_workers', type=int, default=REFINE_WORKERS)
    return parser.parse_args()


def find_baseline_json(dataset):
    base = os.path.join(PROJECT_ROOT, 'outputs')
    candidates = [
        os.path.join(base, f'{dataset}_full.json'),
        os.path.join(base, dataset.split("_", 1)[0], f'{dataset.replace("fashioniq_", "")}_full.json') if 'fashioniq' in dataset else None,
        os.path.join(base, 'circo', 'circo_full.json') if dataset == 'circo' else None,
        os.path.join(base, 'cirr', 'cirr_full.json') if dataset == 'cirr' else None,
        os.path.join(base, f'{dataset}.json'),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


# ==================== Proxy Generation ====================

def generate_proxy_image(prompt, save_path, max_retries=4):
    """Generate one proxy image with rate-limit backoff."""
    if os.path.exists(save_path):
        return save_path
    if not MINIMAX_KEY:
        raise RuntimeError('Missing MINIMAX_API_KEY environment variable.')

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                'https://api.minimax.chat/v1/image_generation',
                headers={'Authorization': f'Bearer {MINIMAX_KEY}',
                         'Content-Type': 'application/json'},
                json={'model': 'image-01', 'prompt': prompt,
                      'aspect_ratio': '1:1', 'response_format': 'url', 'n': 1},
                timeout=PROXY_TIMEOUT)

            if resp.status_code != 200:
                raise Exception(f'HTTP {resp.status_code}')

            result = resp.json()
            status = result.get('base_resp', {}).get('status_code', -1)
            if status != 0:
                msg = result.get('base_resp', {}).get('status_msg', '')
                if status == 1026:
                    return 'SENSITIVE'
                if status == 1002:
                    time.sleep(5 + attempt * 3)
                    continue
                raise Exception(f'API {status}: {msg}')

            img_url = result['data']['image_urls'][0]
            img_data = requests.get(img_url, timeout=PROXY_DOWNLOAD_TIMEOUT).content
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(img_data)
            return save_path

        except Exception as e:
            if 'SENSITIVE' in str(type(e)) or 'SENSITIVE' == save_path:
                return 'SENSITIVE'
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return None
    return None


def _proxy_worker(task):
    """Worker function for concurrent proxy generation."""
    idx, desc, save_path = task
    return idx, generate_proxy_image(desc, save_path)


def generate_proxies_concurrent(all_samples, proxy_dir, num_workers):
    """Generate proxy images with concurrent workers."""
    tasks = []
    existing = 0
    for idx, sample in enumerate(all_samples):
        save_path = os.path.join(proxy_dir, f'proxy_{idx:05d}.jpg')
        if os.path.exists(save_path):
            existing += 1
            continue
        desc = sample.get('target_description', '')
        if desc:
            tasks.append((idx, desc, save_path))

    total_need = len(tasks)
    print(f"  Existing: {existing}/{len(all_samples)}")
    print(f"  Need to generate: {total_need}")
    print(f"  Workers: {num_workers}")
    print(f"  Estimated cost: {total_need * COST_PER_PROXY:.1f} CNY")
    print(flush=True)

    if not tasks:
        return existing, 0, 0

    gen_count = 0
    fail_count = 0
    sensitive_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_proxy_worker, t): t for t in tasks}

        for i, future in enumerate(as_completed(futures)):
            idx, result = future.result()
            if result == 'SENSITIVE':
                sensitive_count += 1
            elif result and result != 'SENSITIVE':
                gen_count += 1
            else:
                fail_count += 1

            done = i + 1
            if done % 50 == 0 or done == total_need:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total_need - done) / rate if rate > 0 else 0
                cost = gen_count * COST_PER_PROXY
                safe_print(f"  [{done}/{total_need}] gen={gen_count} "
                           f"sensitive={sensitive_count} fail={fail_count} "
                           f"cost={cost:.1f}¥ rate={rate:.2f}/s "
                           f"ETA={format_eta(remaining)}")

    total_cost = gen_count * COST_PER_PROXY
    print(f"\n  Proxy complete: {gen_count} generated, "
          f"{sensitive_count} sensitive, {fail_count} failed", flush=True)
    print(f"  Cost: {total_cost:.1f} CNY", flush=True)
    return existing + gen_count, sensitive_count, fail_count


# ==================== V7 Refinement ====================

def call_v7_refine(ref_image_path, proxy_image_path, manipulation_text, engine):
    ref_url = cloudgpt_api.encode_image(ref_image_path)
    proxy_url = cloudgpt_api.encode_image(proxy_image_path)
    if not ref_url or not proxy_url:
        return ""

    user_prompt = f'''<Input>
{{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation text": "{manipulation_text}"
}}'''

    messages = [
        {"role": "system", "content": V7_ANTI_HALLUCINATION},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": ref_url}},
            {"type": "image_url", "image_url": {"url": proxy_url}},
        ]}
    ]

    for attempt in range(3):
        try:
            resp = cloudgpt_api.get_chat_completion(
                engine=engine, messages=messages,
                max_tokens=1024, timeout=60, temperature=0)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                return ""


def parse_v7_response(raw_response, fallback=""):
    resp = raw_response.strip()
    if resp.startswith('```json'):
        resp = resp.replace('```json', '').replace('```', '').strip()
    if resp.startswith('```'):
        resp = resp.replace('```', '').strip()
    try:
        d = json.loads(resp)
        return d.get('Target Image Description', fallback)
    except Exception:
        for line in resp.split('\n'):
            if 'Target Image Description' in line and ':' in line:
                return line.split(':', 1)[1].strip().strip('"').strip("'")
        return fallback


def get_reference_image_path(sample, dataset, dataset_path):
    ref_path = sample.get('reference_image_path', '')

    if ref_path and ref_path.startswith('data:image'):
        return ref_path

    if ref_path and os.path.exists(ref_path):
        return ref_path

    if ref_path and not os.path.isabs(ref_path):
        abs_path = os.path.join(PROJECT_ROOT, ref_path)
        if os.path.exists(abs_path):
            return abs_path
        stem, _ = os.path.splitext(abs_path)
        for ext in ['.jpg', '.png', '.jpeg']:
            if os.path.exists(stem + ext):
                return stem + ext

    ref_name = sample.get('reference_name', '')

    if 'genecis' in dataset and ref_name:
        for base in [os.path.join(dataset_path, 'coco2017', 'val2017'),
                     os.path.join(dataset_path, 'Visual_Genome', 'VG_100K'),
                     os.path.join(dataset_path, 'Visual_Genome', 'VG_100K_2')]:
            for ext in ['.jpg', '.png', '.jpeg']:
                p = os.path.join(base, ref_name + ext)
                if os.path.exists(p):
                    return p
    elif 'fashioniq' in dataset and ref_name:
        for ext in ['.jpg', '.png', '.jpeg']:
            p = os.path.join(dataset_path, 'images', ref_name + ext)
            if os.path.exists(p):
                return p
    elif dataset == 'circo' and ref_name:
        coco_dir = os.path.join(dataset_path, 'COCO2017_unlabeled', 'unlabeled2017')
        for ext in ['.jpg', '.png']:
            p = os.path.join(coco_dir, f'{int(ref_name):012d}{ext}')
            if os.path.exists(p):
                return p
    elif dataset == 'cirr' and ref_name:
        for subdir in ['dev', 'test1', 'train']:
            for ext in ['.png', '.jpg', '.jpeg']:
                p = os.path.join(dataset_path, subdir, ref_name + ext)
                if os.path.exists(p):
                    return p

    return ref_path


def _refine_worker(task):
    """Worker for concurrent V7 refinement."""
    idx, ref_path, proxy_path, instruction, original_desc, engine = task
    raw_resp = call_v7_refine(ref_path, proxy_path, instruction, engine)
    refined_desc = parse_v7_response(raw_resp, fallback=original_desc)
    return {
        'index': idx,
        'instruction': instruction,
        'original_description': original_desc,
        'refined_description': refined_desc,
        'raw_response': raw_resp,
    }


def run_refine_concurrent(all_samples, proxy_dir, existing_refine, refine_cache_path,
                          dataset, dataset_path, engine, num_workers):
    """Run V7 refinement with concurrent workers."""
    tasks = []
    skip_count = 0
    for idx in range(len(all_samples)):
        if str(idx) in existing_refine:
            continue
        proxy_path = os.path.join(proxy_dir, f'proxy_{idx:05d}.jpg')
        if not os.path.exists(proxy_path):
            continue
        sample = all_samples[idx]
        ref_path = get_reference_image_path(sample, dataset, dataset_path)
        if not ref_path or (not ref_path.startswith('data:image') and not os.path.exists(ref_path)):
            skip_count += 1
            continue
        instruction = sample.get('instruction', '')
        original_desc = sample.get('target_description', instruction)
        tasks.append((idx, ref_path, proxy_path, instruction, original_desc, engine))

    total_need = len(tasks)
    print(f"  Existing refinements: {len(existing_refine)}/{len(all_samples)}")
    print(f"  Need to refine: {total_need} (skipped {skip_count} missing refs)")
    print(f"  Workers: {num_workers}")
    print(f"  Estimated cost: {total_need * COST_PER_REFINE:.1f} CNY")
    print(flush=True)

    if not tasks:
        return

    refine_results = list(existing_refine.values())
    refine_count = 0
    fail_count = 0
    start_time = time.time()
    save_interval = 100

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_refine_worker, t): t for t in tasks}

        for i, future in enumerate(as_completed(futures)):
            entry = future.result()
            if entry['raw_response']:
                refine_results.append(entry)
                existing_refine[str(entry['index'])] = entry
                refine_count += 1
            else:
                fail_count += 1

            done = i + 1
            if done % 50 == 0 or done == total_need:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total_need - done) / rate if rate > 0 else 0
                cost = refine_count * COST_PER_REFINE
                safe_print(f"  [{done}/{total_need}] refined={refine_count} fail={fail_count} "
                           f"cost={cost:.1f}¥ rate={rate:.2f}/s "
                           f"ETA={format_eta(remaining)}")

            if done % save_interval == 0:
                sorted_r = sorted(refine_results, key=lambda x: x['index'])
                with open(refine_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(sorted_r, f, ensure_ascii=False, indent=2)

    sorted_r = sorted(refine_results, key=lambda x: x['index'])
    with open(refine_cache_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_r, f, ensure_ascii=False, indent=2)

    cost = refine_count * COST_PER_REFINE
    print(f"\n  V7 refinement complete: {refine_count} refined, {fail_count} failed", flush=True)
    print(f"  Cost: {cost:.1f} CNY", flush=True)
    print(f"  Saved to: {refine_cache_path}", flush=True)


# ==================== Utilities ====================

def format_eta(seconds):
    if seconds <= 0:
        return "done"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def load_baseline_results(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected baseline format in {path}")


# ==================== Main ====================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    baseline_path = args.baseline_json or find_baseline_json(args.dataset)
    if not baseline_path or not os.path.exists(baseline_path):
        print(f"[ERROR] Baseline JSON not found for {args.dataset}.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Three-Way Fusion Pipeline v2 (concurrent)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Baseline: {baseline_path}")
    print(f"  Proxy workers: {args.proxy_workers}, Refine workers: {args.refine_workers}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n", flush=True)

    all_samples = load_baseline_results(baseline_path)
    if args.max_samples:
        all_samples = all_samples[:args.max_samples]
    total = len(all_samples)
    print(f"Loaded {total} baseline samples", flush=True)

    proxy_dir = os.path.join(PROXY_CACHE_DIR, args.dataset)
    os.makedirs(proxy_dir, exist_ok=True)

    refine_cache_path = os.path.join(args.output_dir, f'{args.dataset}_v7_refine_cache.json')

    existing_refine = {}
    if os.path.exists(refine_cache_path):
        try:
            existing_refine = {str(r['index']): r for r in json.load(open(refine_cache_path))}
            print(f"Loaded {len(existing_refine)} existing V7 refinements from cache", flush=True)
        except Exception:
            pass

    # ===== STEP 1: Proxy images =====
    if not args.skip_proxy:
        print(f"\n{'='*70}")
        print(f"  STEP 1: Generating proxy images ({args.dataset})")
        print(f"{'='*70}", flush=True)
        generate_proxies_concurrent(all_samples, proxy_dir, args.proxy_workers)

    # ===== STEP 2: V7 Refinement =====
    if not args.skip_refine:
        print(f"\n{'='*70}")
        print(f"  STEP 2: V7 Anti-Hallucination Refinement ({args.dataset})")
        print(f"{'='*70}", flush=True)
        run_refine_concurrent(all_samples, proxy_dir, existing_refine, refine_cache_path,
                              args.dataset, args.dataset_path, args.engine, args.refine_workers)

    # ===== Summary =====
    proxy_count = sum(1 for idx in range(total)
                      if os.path.exists(os.path.join(proxy_dir, f'proxy_{idx:05d}.jpg')))
    refine_count = len(existing_refine)

    print(f"\n{'='*70}")
    print(f"  Pipeline Summary for {args.dataset}")
    print(f"{'='*70}")
    print(f"  Total samples:     {total}")
    print(f"  Proxy images:      {proxy_count}/{total}")
    print(f"  V7 refinements:    {refine_count}/{total}")
    print(f"  Ready for eval:    {'YES' if proxy_count >= total * 0.95 and refine_count >= total * 0.95 else 'PARTIAL'}")
    print(f"  Refine cache:      {refine_cache_path}")
    print(f"  Completed at:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n", flush=True)


if __name__ == '__main__':
    main()
