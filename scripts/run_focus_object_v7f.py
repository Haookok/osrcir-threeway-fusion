"""
Run V7-Focus prompt on 200 random focus_object samples.
Only calls Qwen-VL API — proxy images already exist.
"""
import json
import os
import sys
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import cloudgpt_api
from refine_prompts import V7_FOCUS

BASELINE_JSON = 'outputs/genecis_focus_object_full.json'
PROXY_DIR = 'proxy_cache/genecis_focus_object'
DATASET_PATH = 'datasets/GENECIS'
OUTPUT_PATH = 'outputs/full_pipeline/genecis_focus_object_v7focus_refine_cache.json'
ENGINE = 'qwen-vl-max-latest'
N_SAMPLES = 200
SEED = 42
WORKERS = 3

print_lock = threading.Lock()


def safe_print(msg):
    with print_lock:
        print(msg, flush=True)


def call_refine(ref_image_path, proxy_image_path, manipulation_text):
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
        {"role": "system", "content": V7_FOCUS},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": ref_url}},
            {"type": "image_url", "image_url": {"url": proxy_url}},
        ]}
    ]

    for attempt in range(3):
        try:
            resp = cloudgpt_api.get_chat_completion(
                engine=ENGINE, messages=messages,
                max_tokens=1024, timeout=60, temperature=0)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                safe_print(f"  [FAIL] {e}")
                return ""


def parse_response(raw, fallback=""):
    resp = raw.strip()
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


def resolve_ref_path(sample):
    ref_path = sample.get('reference_image_path', '')
    if ref_path and ref_path.startswith('data:image'):
        return ref_path
    if ref_path and os.path.exists(ref_path):
        return ref_path
    if ref_path and not os.path.isabs(ref_path):
        abs_path = os.path.join(os.path.dirname(__file__), ref_path)
        if os.path.exists(abs_path):
            return abs_path
    return ref_path


def worker(task):
    idx, ref_path, proxy_path, instruction, original_desc = task
    raw = call_refine(ref_path, proxy_path, instruction)
    refined = parse_response(raw, fallback=original_desc)
    return {
        'index': idx,
        'instruction': instruction,
        'original_description': original_desc,
        'refined_description': refined,
        'raw_response': raw,
        'prompt_version': 'v7_focus',
    }


def main():
    print(f"Loading baseline: {BASELINE_JSON}")
    baseline = json.load(open(BASELINE_JSON, encoding='utf-8'))
    print(f"  Total: {len(baseline)} samples")

    random.seed(SEED)
    indices = sorted(random.sample(range(len(baseline)), min(N_SAMPLES, len(baseline))))
    print(f"  Selected {len(indices)} samples (seed={SEED})")

    # Also load old V7 cache for comparison
    old_cache_path = 'outputs/full_pipeline/genecis_focus_object_v7_refine_cache.json'
    old_refine = {}
    if os.path.exists(old_cache_path):
        old_data = json.load(open(old_cache_path, encoding='utf-8'))
        old_refine = {r['index']: r for r in old_data}
        print(f"  Old V7 cache: {len(old_refine)} entries")

    tasks = []
    for idx in indices:
        sample = baseline[idx]
        ref_path = resolve_ref_path(sample)
        proxy_path = os.path.join(PROXY_DIR, f'proxy_{idx:05d}.jpg')
        instruction = sample.get('instruction', '')
        original_desc = sample.get('target_description', '')

        if not os.path.exists(proxy_path):
            print(f"  [SKIP] no proxy for {idx}")
            continue
        tasks.append((idx, ref_path, proxy_path, instruction, original_desc))

    print(f"\n  Tasks: {len(tasks)}")
    est_cost = len(tasks) * 0.007
    print(f"  Estimated cost: {est_cost:.1f} CNY")
    print(f"  Workers: {WORKERS}")
    print()

    results = []
    done_count = 0
    fail_count = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}

        for future in as_completed(futures):
            entry = future.result()
            done_count += 1

            if entry['raw_response']:
                results.append(entry)
                idx = entry['index']
                d1 = entry['original_description']
                d2 = entry['refined_description']
                old_d2 = old_refine.get(idx, {}).get('refined_description', '')

                if done_count <= 5 or done_count % 50 == 0:
                    safe_print(f"\n[{done_count}/{len(tasks)}] idx={idx}")
                    safe_print(f"  instruction: {entry['instruction']}")
                    safe_print(f"  D1:     {d1[:100]}")
                    safe_print(f"  old D2: {old_d2[:100]}")
                    safe_print(f"  new D2: {d2[:100]}")
                    safe_print(f"  lengths: D1={len(d1.split())}w old={len(old_d2.split())}w new={len(d2.split())}w")
            else:
                fail_count += 1

            if done_count % 20 == 0 or done_count == len(tasks):
                elapsed = time.time() - start
                rate = done_count / elapsed if elapsed > 0 else 0
                safe_print(f"  progress: {done_count}/{len(tasks)} "
                           f"ok={len(results)} fail={fail_count} "
                           f"rate={rate:.1f}/s cost={len(results)*0.007:.1f}¥")

    results.sort(key=lambda x: x['index'])
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Stats comparison
    old_lens, new_lens = [], []
    ultra_short_old, ultra_short_new = 0, 0
    for r in results:
        idx = r['index']
        new_d2 = r['refined_description']
        old_d2 = old_refine.get(idx, {}).get('refined_description', '')
        nw = len(new_d2.split())
        ow = len(old_d2.split())
        new_lens.append(nw)
        old_lens.append(ow)
        if ow <= 3:
            ultra_short_old += 1
        if nw <= 3:
            ultra_short_new += 1

    print(f"\n{'='*60}")
    print(f"  DONE — {len(results)} refined, {fail_count} failed")
    print(f"  Cost: {len(results)*0.007:.1f} CNY")
    print(f"  Saved: {OUTPUT_PATH}")
    print(f"\n  D2 length comparison (200 samples):")
    print(f"    Old V7: avg={sum(old_lens)/len(old_lens):.1f}w, <=3w: {ultra_short_old} ({ultra_short_old/len(old_lens)*100:.1f}%)")
    print(f"    New V7F: avg={sum(new_lens)/len(new_lens):.1f}w, <=3w: {ultra_short_new} ({ultra_short_new/len(new_lens)*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
