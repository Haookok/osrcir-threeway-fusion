"""
Test GeneCIS-specific prompts on random samples or full data.

For each sub-dataset, generates new D2 with the GeneCIS-specific prompt,
then evaluates with various α/β and compares to V7 baseline.

Usage:
  python3 -u scripts/eval/test_genecis_prompt.py --datasets genecis_change_object
  python3 -u scripts/eval/test_genecis_prompt.py --sample_size 200
  python3 -u scripts/eval/test_genecis_prompt.py --sample_size 0  # full data
"""
import argparse
import json
import os
import sys
import time
import random
import pickle
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import cloudgpt_api
from refine_prompts import PROMPT_VARIANTS

GENECIS_PATH = os.path.join(ROOT, 'datasets', 'GENECIS')
ENGINE = 'qwen-vl-max-latest'
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 8

DATASET_CONFIGS = {
    'genecis_change_object': {
        'prompt_key': 'genecis_change_object',
        'baseline_pkl': f'precomputed_cache/precomputed/genecis_change_object_val_mods_mllm_structural_predictor_prompt_CoT_qwen-vl-max-latest.pkl',
        'v7_cache': f'outputs/full_pipeline/genecis_change_object_v7_refine_cache.json',
        'proxy_dir': f'proxy_cache/genecis_change_object',
        'annotation': f'datasets/GENECIS/genecis/change_object.json',
        'gallery_cache': f'precomputed_cache/genecis/genecis_change_object_gallery.pkl',
        'image_type': 'coco',
    },
    'genecis_focus_object': {
        'prompt_key': 'genecis_focus_object',
        'baseline_pkl': f'precomputed_cache/precomputed/genecis_focus_object_val_mods_mllm_structural_predictor_prompt_CoT_qwen-vl-max-latest.pkl',
        'v7_cache': f'outputs/full_pipeline/genecis_focus_object_v7_refine_cache.json',
        'proxy_dir': f'proxy_cache/genecis_focus_object',
        'annotation': f'datasets/GENECIS/genecis/focus_object.json',
        'gallery_cache': f'precomputed_cache/genecis/genecis_focus_object_gallery.pkl',
        'image_type': 'coco',
    },
    'genecis_change_attribute': {
        'prompt_key': 'genecis_change_attribute',
        'baseline_pkl': f'precomputed_cache/precomputed/genecis_change_attribute_val_mods_mllm_structural_predictor_prompt_CoT_qwen-vl-max-latest.pkl',
        'v7_cache': f'outputs/full_pipeline/genecis_change_attribute_v7_refine_cache.json',
        'proxy_dir': f'proxy_cache/genecis_change_attribute',
        'annotation': f'datasets/GENECIS/genecis/change_attribute.json',
        'gallery_cache': f'precomputed_cache/genecis/genecis_change_attribute_gallery.pkl',
        'image_type': 'vg',
    },
    'genecis_focus_attribute': {
        'prompt_key': 'genecis_focus_attribute',
        'baseline_pkl': f'precomputed_cache/precomputed/genecis_focus_attribute_val_mods_mllm_structural_predictor_prompt_CoT_qwen-vl-max-latest.pkl',
        'v7_cache': f'outputs/full_pipeline/genecis_focus_attribute_v7_refine_cache.json',
        'proxy_dir': f'proxy_cache/genecis_focus_attribute',
        'annotation': f'datasets/GENECIS/genecis/focus_attribute.json',
        'gallery_cache': f'precomputed_cache/genecis/genecis_focus_attribute_gallery.pkl',
        'image_type': 'vg',
    },
}


def call_refine(ref_image_path, proxy_image_path, manipulation_text, system_prompt):
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
        {"role": "system", "content": system_prompt},
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
                max_tokens=512, timeout=60, temperature=0)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"  FAIL: {e}", flush=True)
                return ""


def parse_response(raw_response, fallback=""):
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


def get_ref_path(sample):
    ref = sample.get('reference_image_path', '')
    if ref.startswith('data:image'):
        return ref
    if ref and not os.path.isabs(ref):
        ref = os.path.join(ROOT, ref)
    if os.path.exists(ref):
        return ref
    return ref


def refine_worker(args):
    idx, ref_path, proxy_path, instruction, original_desc, system_prompt = args
    raw = call_refine(ref_path, proxy_path, instruction, system_prompt)
    desc = parse_response(raw, fallback=original_desc)
    return {
        'index': idx,
        'instruction': instruction,
        'original_description': original_desc,
        'refined_description': desc,
        'raw_response': raw,
    }


def generate_refinements(name, config, sample_indices, all_samples, num_workers):
    prompt_key = config['prompt_key']
    system_prompt = PROMPT_VARIANTS[prompt_key]
    proxy_dir = os.path.join(ROOT, config['proxy_dir'])

    cache_path = os.path.join(ROOT, 'outputs', 'full_pipeline',
                              f'{name}_genecis_prompt_cache.json')
    existing = {}
    if os.path.exists(cache_path):
        for entry in json.load(open(cache_path)):
            existing[entry['index']] = entry

    tasks = []
    for idx in sample_indices:
        if idx in existing:
            continue
        sample = all_samples[idx]
        ref_path = get_ref_path(sample)
        proxy_path = os.path.join(proxy_dir, f'proxy_{idx:05d}.jpg')
        if not os.path.exists(proxy_path):
            continue
        instruction = sample.get('instruction', '')
        original_desc = sample.get('target_description', instruction)
        tasks.append((idx, ref_path, proxy_path, instruction, original_desc, system_prompt))

    cached = sum(1 for idx in sample_indices if idx in existing)
    print(f"  Cached: {cached}, Need API calls: {len(tasks)}", flush=True)
    if not tasks:
        return {idx: existing[idx] for idx in sample_indices if idx in existing}

    est_cost = len(tasks) * 0.007
    print(f"  Estimated cost: {est_cost:.2f} CNY", flush=True)

    results = dict(existing)
    done_count = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(refine_worker, t): t for t in tasks}
        for future in as_completed(futures):
            entry = future.result()
            results[entry['index']] = entry
            done_count += 1
            if done_count % 20 == 0 or done_count == len(tasks):
                elapsed = time.time() - t0
                rate = done_count / max(elapsed, 1)
                print(f"    [{done_count}/{len(tasks)}] {rate:.1f}/s", flush=True)

    all_results = sorted(results.values(), key=lambda x: x['index'])
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved cache: {cache_path} ({len(all_results)} entries)", flush=True)

    return {idx: results[idx] for idx in sample_indices if idx in results}


def evaluate_with_clip(name, config, sample_indices, all_samples,
                       new_refine_map, v7_refine_map, model, preprocess, tokenizer):
    import torch
    import torch.nn.functional as F
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    annotation = json.load(open(os.path.join(ROOT, config['annotation'])))
    gallery_data = pickle.load(open(os.path.join(ROOT, config['gallery_cache']), 'rb'))
    gallery_feats = gallery_data['feats'].to(DEVICE)
    gallery_feats = F.normalize(gallery_feats, dim=-1)
    gid_to_idx = {gid: i for i, gid in enumerate(gallery_data['ids'])}
    image_type = config['image_type']

    results = {'baseline': [], 'v7': [], 'new_prompt': []}

    for qi, sample_idx in enumerate(sample_indices):
        sample = all_samples[sample_idx]
        ann = annotation[sample_idx]
        d1_text = sample.get('target_description', sample.get('instruction', ''))

        v7_entry = v7_refine_map.get(sample_idx, {})
        d2_v7 = v7_entry.get('refined_description', d1_text)

        new_entry = new_refine_map.get(sample_idx, {})
        d2_new = new_entry.get('refined_description', d1_text)

        gallery_items = ann.get('gallery', [])
        target = ann.get('target', {})
        id_key = 'val_image_id' if image_type == 'coco' else 'image_id'
        target_id = target.get(id_key)
        gallery_ids = [g[id_key] for g in gallery_items]

        if target_id not in gallery_ids:
            gallery_ids.append(target_id)
        target_pos = gallery_ids.index(target_id)

        local_indices = []
        skip = False
        for gid in gallery_ids:
            if gid not in gid_to_idx:
                skip = True
                break
            local_indices.append(gid_to_idx[gid])

        if skip or len(local_indices) < 2:
            continue

        local_gallery = gallery_feats[local_indices]
        target_local = target_pos

        proxy_path = os.path.join(ROOT, config['proxy_dir'], f'proxy_{sample_idx:05d}.jpg')

        with torch.no_grad():
            tokens_d1 = tokenizer([d1_text]).to(DEVICE)
            feat_d1 = F.normalize(model.encode_text(tokens_d1).float(), dim=-1)

            tokens_v7 = tokenizer([d2_v7]).to(DEVICE)
            feat_v7 = F.normalize(model.encode_text(tokens_v7).float(), dim=-1)

            tokens_new = tokenizer([d2_new]).to(DEVICE)
            feat_new = F.normalize(model.encode_text(tokens_new).float(), dim=-1)

            proxy_feat = None
            if os.path.exists(proxy_path):
                from PIL import Image
                try:
                    img = Image.open(proxy_path).convert('RGB')
                    img_t = preprocess(img).unsqueeze(0).to(DEVICE)
                    proxy_feat = F.normalize(model.encode_image(img_t).float(), dim=-1)
                    img.close()
                except Exception as e:
                    # Robustness: some proxy files may be truncated/corrupted.
                    # Fallback to text-only fusion for this query.
                    proxy_feat = None

        sim_d1 = (feat_d1 @ local_gallery.T).squeeze(0)
        sim_v7 = (feat_v7 @ local_gallery.T).squeeze(0)
        sim_new = (feat_new @ local_gallery.T).squeeze(0)
        sim_proxy = (proxy_feat @ local_gallery.T).squeeze(0) if proxy_feat is not None else None

        def get_rank(scores, target_pos):
            sorted_idx = torch.argsort(scores, descending=True)
            return (sorted_idx == target_pos).nonzero(as_tuple=True)[0].item()

        rank_baseline = get_rank(sim_d1, target_local)
        results['baseline'].append(rank_baseline)

        best_v7_rank = 999
        best_new_rank = 999
        for alpha in [0.80, 0.85, 0.90, 0.95, 1.00]:
            for beta in [0.30, 0.50, 0.60, 0.70, 0.80, 1.00]:
                ens_v7 = F.normalize(beta * feat_d1 + (1 - beta) * feat_v7, dim=-1)
                ens_new = F.normalize(beta * feat_d1 + (1 - beta) * feat_new, dim=-1)
                sim_ens_v7 = (ens_v7 @ local_gallery.T).squeeze(0)
                sim_ens_new = (ens_new @ local_gallery.T).squeeze(0)
                if sim_proxy is not None:
                    score_v7 = alpha * sim_ens_v7 + (1 - alpha) * (proxy_feat @ local_gallery.T).squeeze(0)
                    score_new = alpha * sim_ens_new + (1 - alpha) * (proxy_feat @ local_gallery.T).squeeze(0)
                else:
                    score_v7 = sim_ens_v7
                    score_new = sim_ens_new
                rank_v7 = get_rank(score_v7, target_local)
                rank_new = get_rank(score_new, target_local)
                best_v7_rank = min(best_v7_rank, rank_v7)
                best_new_rank = min(best_new_rank, rank_new)

        results['v7'].append(best_v7_rank)
        results['new_prompt'].append(best_new_rank)

    n = len(results['baseline'])
    if n == 0:
        for method in ['baseline', 'v7', 'new_prompt']:
            results[f'{method}_r1'] = results[f'{method}_r2'] = results[f'{method}_r3'] = 0.0
        return results, 0
    for method in ['baseline', 'v7', 'new_prompt']:
        ranks = results[method]
        r1 = sum(1 for r in ranks if r < 1) / n * 100
        r2 = sum(1 for r in ranks if r < 2) / n * 100
        r3 = sum(1 for r in ranks if r < 3) / n * 100
        results[f'{method}_r1'] = r1
        results[f'{method}_r2'] = r2
        results[f'{method}_r3'] = r3

    return results, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--sample_size', type=int, default=DEFAULT_SAMPLE_SIZE,
                        help='0 means full data, >0 means random sample size')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    print("=" * 60)
    mode = "FULL DATA" if args.sample_size == 0 else f"{args.sample_size} Random Samples"
    print(f"  GeneCIS Prompt Test — {mode}")
    print("=" * 60)

    import torch
    import open_clip
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nLoading CLIP on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14-quickgelu', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-L-14-quickgelu')
    model = model.to(DEVICE).eval()
    print("  Model loaded.", flush=True)

    for name in args.datasets:
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {name}")
            continue
        config = DATASET_CONFIGS[name]
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        pkl_path = os.path.join(ROOT, config['baseline_pkl'])
        baseline_data = pickle.load(open(pkl_path, 'rb'))
        all_samples = baseline_data['generated_results']
        total = len(all_samples)

        if args.sample_size == 0:
            sample_indices = list(range(total))
            print(f"  Total: {total}, Testing: {len(sample_indices)} (full)")
        else:
            random.seed(args.seed)
            sample_indices = sorted(random.sample(range(total), min(args.sample_size, total)))
            print(f"  Total: {total}, Testing: {len(sample_indices)} (seed={args.seed})")

        # Load V7 cache
        v7_cache_path = os.path.join(ROOT, config['v7_cache'])
        v7_refine_map = {}
        if os.path.exists(v7_cache_path):
            for entry in json.load(open(v7_cache_path)):
                v7_refine_map[entry['index']] = entry
        print(f"  V7 cache: {len(v7_refine_map)} entries")

        # Generate new refinements
        print(f"\n  Generating new refinements (prompt: {config['prompt_key']})...")
        print(f"  Workers: {args.num_workers}")
        new_refine_map = generate_refinements(
            name, config, sample_indices, all_samples, args.num_workers
        )

        # Show sample comparisons
        print(f"\n  --- Sample D1 vs V7-D2 vs New-D2 ---")
        shown = 0
        for idx in sample_indices[:10]:
            sample = all_samples[idx]
            d1 = sample.get('target_description', '')[:80]
            v7 = v7_refine_map.get(idx, {}).get('refined_description', '')[:80]
            new = new_refine_map.get(idx, {}).get('refined_description', '')[:80]
            if new:
                inst = sample.get('instruction', '')
                print(f"  [{inst}]")
                print(f"    D1:  {d1}")
                print(f"    V7:  {v7}")
                print(f"    NEW: {new}")
                print()
                shown += 1
                if shown >= 5:
                    break

        # Evaluate
        print(f"\n  Evaluating with CLIP...")
        results, n_eval = evaluate_with_clip(
            name, config, sample_indices, all_samples,
            new_refine_map, v7_refine_map, model, preprocess, tokenizer)

        print(f"\n  Results ({n_eval} queries evaluated):")
        print(f"  {'Method':<20} {'R@1':>8} {'R@2':>8} {'R@3':>8}")
        print(f"  {'-'*44}")
        for method, label in [('baseline', 'D1 only (baseline)'),
                               ('v7', 'V7 (best α/β)'),
                               ('new_prompt', 'New prompt (best α/β)')]:
            r1 = results[f'{method}_r1']
            r2 = results[f'{method}_r2']
            r3 = results[f'{method}_r3']
            print(f"  {label:<20} {r1:>7.2f}% {r2:>7.2f}% {r3:>7.2f}%")

        d_v7 = results['v7_r1'] - results['baseline_r1']
        d_new = results['new_prompt_r1'] - results['baseline_r1']
        d_diff = results['new_prompt_r1'] - results['v7_r1']
        print(f"\n  V7 vs baseline:     ΔR@1 = {d_v7:+.2f}pp")
        print(f"  New vs baseline:    ΔR@1 = {d_new:+.2f}pp")
        print(f"  New vs V7:          ΔR@1 = {d_diff:+.2f}pp")

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
