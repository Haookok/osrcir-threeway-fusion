"""
Combined Visual Proxy Pipeline (方案A+B叠加).

Round 1: reference + modification → MLLM(CoT) → initial description → text-to-image → proxy image
Round 2: reference + proxy + modification → MLLM(Reflective CoT) → refined description
Retrieval: α × CLIP_text(refined_desc) + (1-α) × CLIP_image(proxy) → hybrid ranking

Outputs comparison of all 4 approaches:
  1. Baseline: original text only
  2. Plan A:   original text + proxy image (hybrid)
  3. Plan B:   refined text only
  4. A+B:      refined text + proxy image (hybrid)
"""
import argparse
import json
import os
import pickle
import random
import time
import requests
import numpy as np
import torch
import tqdm
import PIL.Image
import clip

import cloudgpt_api

PROXY_CACHE_DIR = './proxy_cache'

REFINE_PROMPT = '''
You are an image description expert performing a REFINEMENT task.

You have been given:
1. An **Original Image** (the reference image the user wants to modify)
2. A **Proxy Image** (an AI-generated visualization of what the target might look like, based on a first-round description)
3. A **Manipulation Text** (the user's modification intent)

Your goal: Compare the Original Image with the Proxy Image, and use the Manipulation Text to generate a MORE ACCURATE target image description than the first round produced.

## Refinement Strategy
- Look at the Proxy Image: does it correctly capture the user's modification intent?
- Look at the Original Image: are important details from the original preserved as they should be?
- Identify any mismatches, hallucinations, or missing details in the Proxy Image.
- Generate a refined target description that fixes these issues.

## Guidelines
- The refined description should be SHORT, PRECISE, and contain ONLY the target image content.
- Focus on visual attributes that a retrieval model can match (objects, colors, spatial layout, actions).
- Minimize aesthetic or subjective descriptions.

## Input Format
{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation Text": <manipulation_text>
}

## Response Format (JSON)
{
    "Proxy Assessment": <brief assessment of what the proxy image got right and wrong>,
    "Refinement Reasoning": <how you will improve the target description>,
    "Refined Target Description": <the improved target image description>
}
'''


def parse_args():
    parser = argparse.ArgumentParser('Combined Visual Proxy (A+B)')
    parser.add_argument('--results_json', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
                                 'cirr', 'circo'])
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--clip_model', type=str, default='ViT-L/14')
    parser.add_argument('--img_features_cache', type=str, default=None)
    parser.add_argument('--minimax_key', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--alpha', type=float, nargs='+',
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--output_dir', type=str, default='./outputs/combined_experiments')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--openai_engine', type=str, default='qwen-vl-max-latest')
    return parser.parse_args()


# ── image generation ─────────────────────────────────────────────────

def generate_proxy_image(prompt, api_key, save_path, max_retries=3):
    if os.path.exists(save_path):
        return save_path
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                'https://api.minimax.chat/v1/image_generation',
                headers={'Authorization': f'Bearer {api_key}',
                         'Content-Type': 'application/json'},
                json={'model': 'image-01', 'prompt': prompt,
                      'aspect_ratio': '1:1', 'response_format': 'url', 'n': 1},
                timeout=60)
            if resp.status_code != 200:
                raise Exception(f'HTTP {resp.status_code}')
            result = resp.json()
            if result.get('base_resp', {}).get('status_code', -1) != 0:
                raise Exception(f'API error: {result.get("base_resp")}')
            img_url = result['data']['image_urls'][0]
            img_data = requests.get(img_url, timeout=30).content
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(img_data)
            return save_path
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f'  [FAIL] proxy gen: {e}')
                return None


# ── MLLM refinement ─────────────────────────────────────────────────

def call_mllm_dual_image(ref_path, proxy_path, manipulation_text, engine):
    ref_url = cloudgpt_api.encode_image(ref_path)
    proxy_url = cloudgpt_api.encode_image(proxy_path)
    if not ref_url or not proxy_url:
        return ""
    user_prompt = (
        '<Input>\n{\n'
        '    "Original Image": <image_1>,\n'
        '    "Proxy Image": <image_2>,\n'
        f'    "Manipulation Text": "{manipulation_text}"\n'
        '}\n'
    )
    messages = [
        {"role": "system", "content": REFINE_PROMPT},
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
                time.sleep(5 * (attempt + 1))
            else:
                print(f'  [FAIL] MLLM refine: {e}')
                return ""


def parse_refined_description(raw, fallback):
    resp = raw.strip()
    if resp.startswith('```json'):
        resp = resp.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(resp).get('Refined Target Description', fallback)
    except Exception:
        return fallback


# ── CLIP helpers ─────────────────────────────────────────────────────

@torch.no_grad()
def encode_texts(clip_model, tokenizer, texts, device, batch_size=32):
    feats = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i+batch_size]).to(device)
        feats.append(clip_model.encode_text(tokens).float().cpu())
    return torch.vstack(feats)


@torch.no_grad()
def encode_images(clip_model, preprocess, paths, device, batch_size=16):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            if p and os.path.exists(p):
                batch.append(preprocess(PIL.Image.open(p).convert('RGB')))
            else:
                batch.append(torch.zeros(3, 224, 224))
        feats.append(clip_model.encode_image(torch.stack(batch).to(device)).float().cpu())
    return torch.vstack(feats)


def compute_recall(sorted_names, target_names, ks=(1, 5, 10, 50)):
    labels = torch.tensor(
        sorted_names == np.repeat(
            np.array(target_names), sorted_names.shape[1]
        ).reshape(len(target_names), -1))
    return {f'R@{k}': round((torch.sum(labels[:, :k]) / len(labels)).item() * 100, 2)
            for k in ks}


def compute_circo_metrics(sorted_names, target_names, gt_targets_list,
                          ks=(5, 10, 25, 50)):
    """mAP + recall for CIRCO (multiple ground truths per query)."""
    maps = {k: [] for k in ks}
    recalls = {k: [] for k in ks}
    for i, (sn, tn, gts) in enumerate(zip(sorted_names, target_names, gt_targets_list)):
        gts_clean = [str(g) for g in gts if g != '' and g is not None]
        sn_top = sn[:max(ks)]
        map_labels = torch.tensor(np.isin(sn_top, gts_clean), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels
        precisions = precisions / torch.arange(1, len(sn_top) + 1)
        for k in ks:
            maps[k].append(float(torch.sum(precisions[:k]) / min(len(gts_clean), k)))
        single_label = torch.tensor(sn_top == str(tn))
        for k in ks:
            recalls[k].append(float(torch.sum(single_label[:k])))
    out = {f'mAP@{k}': round(np.mean(v) * 100, 2) for k, v in maps.items()}
    out.update({f'R@{k}': round(np.mean(v) * 100, 2) for k, v in recalls.items()})
    return out


def _get_sorted_names(text_feat, index_feat, index_names, proxy_feat=None, alpha=1.0):
    tf = torch.nn.functional.normalize(text_feat.float(), dim=-1)
    idx = torch.nn.functional.normalize(index_feat.float(), dim=-1)
    sim = tf @ idx.T
    if proxy_feat is not None and alpha < 1.0:
        pf = torch.nn.functional.normalize(proxy_feat.float(), dim=-1)
        sim = alpha * sim + (1 - alpha) * (pf @ idx.T)
    sorted_idx = torch.argsort(1 - sim, dim=-1).cpu()
    return np.array(index_names)[sorted_idx]


def retrieval(text_feat, index_feat, index_names, target_names,
              proxy_feat=None, alpha=1.0, gt_targets_list=None):
    """Single retrieval pass. alpha=1.0 means pure text."""
    sorted_names = _get_sorted_names(text_feat, index_feat, index_names,
                                     proxy_feat, alpha)
    if gt_targets_list is not None:
        return compute_circo_metrics(sorted_names, target_names, gt_targets_list)
    return compute_recall(sorted_names, target_names)


# ── main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load & sample
    print('Loading results...')
    with open(args.results_json) as f:
        all_samples = json.load(f)
    if args.random_seed is not None and args.max_samples:
        random.seed(args.random_seed)
        n = min(args.max_samples, len(all_samples))
        sample_indices = sorted(random.sample(range(len(all_samples)), n))
        samples = [all_samples[i] for i in sample_indices]
        print(f'  Randomly sampled {n}/{len(all_samples)} (seed={args.random_seed})')
    elif args.max_samples:
        samples = all_samples[:args.max_samples]
        sample_indices = list(range(len(samples)))
    else:
        samples = all_samples
        sample_indices = list(range(len(samples)))
    print(f'  {len(samples)} samples')

    original_descs = [s['target_description'] for s in samples]
    target_names = [s.get('target_name', '') for s in samples]
    is_circo = (args.dataset == 'circo')
    gt_targets_list = [s['gt_target_names'] for s in samples] if is_circo else None

    # 2. Proxy images (cache-aware)
    print('\n[Phase 1] Proxy images...')
    proxy_dir = os.path.join(PROXY_CACHE_DIR, args.dataset)
    os.makedirs(proxy_dir, exist_ok=True)
    proxy_paths = []
    for i, desc in enumerate(tqdm.tqdm(original_descs, desc='Proxies')):
        orig_idx = sample_indices[i]
        path = generate_proxy_image(
            desc, args.minimax_key,
            os.path.join(proxy_dir, f'proxy_{orig_idx:05d}.jpg'))
        proxy_paths.append(path)
    print(f'  Ready: {sum(1 for p in proxy_paths if p)}/{len(samples)}')

    # 3. MLLM refinement
    print('\n[Phase 2] MLLM refinement (reference + proxy)...')
    refined_descs = []
    details = []
    for i, s in enumerate(tqdm.tqdm(samples, desc='Refining')):
        ref_path = s.get('reference_image_path', '')
        proxy_path = proxy_paths[i]
        instruction = s.get('instruction', '')
        orig_desc = s.get('target_description', instruction)

        if not proxy_path or not os.path.exists(str(proxy_path)):
            refined_descs.append(orig_desc)
            details.append({'status': 'skipped'})
            continue

        raw = call_mllm_dual_image(ref_path, proxy_path, instruction, args.openai_engine)
        refined = parse_refined_description(raw, fallback=orig_desc)
        refined_descs.append(refined)
        details.append({
            'index': i, 'instruction': instruction,
            'original': orig_desc, 'refined': refined, 'raw': raw,
        })
        if i < 5 or i % 20 == 0:
            print(f'  [{i}] {instruction[:60]}')
            print(f'       orig: {orig_desc[:70]}...')
            print(f'       refi: {refined[:70]}...')

    # 4. CLIP encode everything
    print('\n[Phase 3] CLIP encoding...')
    clip_model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()
    tokenizer = lambda t: clip.tokenize(t, context_length=77, truncate=True)

    cache_path = args.img_features_cache
    if cache_path and os.path.exists(cache_path):
        data = pickle.load(open(cache_path, 'rb'))
        index_feat, index_names = data['index_features'], data['index_names']
    else:
        raise FileNotFoundError(f'Need index features cache: {cache_path}')
    print(f'  Index: {len(index_names)} images')

    orig_feat = encode_texts(clip_model, tokenizer, original_descs, device)
    refined_feat = encode_texts(clip_model, tokenizer, refined_descs, device)
    proxy_feat = encode_images(clip_model, preprocess, proxy_paths, device)

    # 5. Run all 4 approaches
    print('\n[Phase 4] Retrieval comparison...')
    results = {}

    ret_kw = dict(gt_targets_list=gt_targets_list) if is_circo else {}

    # (1) Baseline: original text only
    results['Baseline (orig text)'] = retrieval(
        orig_feat, index_feat, index_names, target_names, **ret_kw)

    # (2) Plan B: refined text only
    results['Plan B (refined text)'] = retrieval(
        refined_feat, index_feat, index_names, target_names, **ret_kw)

    # (3) Plan A: original text + proxy hybrid (multiple α)
    for a in args.alpha:
        if a < 1.0:
            results[f'Plan A orig+proxy α={a:.1f}'] = retrieval(
                orig_feat, index_feat, index_names, target_names,
                proxy_feat, alpha=a, **ret_kw)

    # (4) A+B: refined text + proxy hybrid (multiple α)
    for a in args.alpha:
        if a < 1.0:
            results[f'A+B refined+proxy α={a:.1f}'] = retrieval(
                refined_feat, index_feat, index_names, target_names,
                proxy_feat, alpha=a, **ret_kw)

    # 6. Print
    print('\n' + '=' * 90)
    print(f'  ALL RESULTS — {args.dataset} ({len(samples)} samples, seed={args.random_seed})')
    print('=' * 90)
    if is_circo:
        print(f'{"Method":<35} {"mAP@5":>8} {"mAP@10":>8} {"mAP@25":>8} {"mAP@50":>8} {"R@50":>8}')
        print('-' * 90)
        for method, m in results.items():
            print(f'{method:<35} {m["mAP@5"]:>8.2f} {m["mAP@10"]:>8.2f} '
                  f'{m["mAP@25"]:>8.2f} {m["mAP@50"]:>8.2f} {m["R@50"]:>8.2f}')
    else:
        print(f'{"Method":<35} {"R@1":>7} {"R@5":>7} {"R@10":>7} {"R@50":>7}')
        print('-' * 90)
        for method, m in results.items():
            print(f'{method:<35} {m["R@1"]:>7.2f} {m["R@5"]:>7.2f} '
                  f'{m["R@10"]:>7.2f} {m["R@50"]:>7.2f}')
    print('=' * 90)

    primary = 'mAP@5' if is_circo else 'R@10'
    secondary = 'mAP@50' if is_circo else 'R@50'
    best1 = max(results.items(), key=lambda x: x[1][primary])
    best2 = max(results.items(), key=lambda x: x[1][secondary])
    print(f'\n  Best {primary}: {best1[0]} → {best1[1][primary]:.2f}')
    print(f'  Best {secondary}: {best2[0]} → {best2[1][secondary]:.2f}')

    # 7. Save
    seed_tag = f'_seed{args.random_seed}' if args.random_seed is not None else ''
    out_path = os.path.join(args.output_dir,
                            f'{args.dataset}_combined_{len(samples)}s{seed_tag}.json')
    payload = {
        'dataset': args.dataset,
        'num_samples': len(samples),
        'random_seed': args.random_seed,
        'sample_indices': sample_indices,
        'method': 'combined_A+B',
        'alpha_values': args.alpha,
        'results': results,
        'refinement_details': details,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
