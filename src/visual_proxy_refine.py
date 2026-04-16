"""
Visual Self-Refinement Pipeline (方案B: 前融合).

Round 1: reference_image + modification_text → MLLM(CoT) → initial description → text-to-image → proxy image
Round 2: reference_image + proxy_image + modification_text → MLLM(CoT) → refined description → CLIP retrieval
"""
import argparse
import json
import os
import pickle
import random
import time
import base64
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
    parser = argparse.ArgumentParser('Visual Self-Refinement')
    parser.add_argument('--results_json', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
                                 'cirr', 'circo'])
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--clip_model', type=str, default='ViT-L/14')
    parser.add_argument('--img_features_cache', type=str, default=None)
    parser.add_argument('--minimax_key', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed for sampling. If set, randomly sample max_samples from dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/refine_experiments')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--openai_engine', type=str, default='qwen-vl-max-latest')
    return parser.parse_args()


def generate_proxy_image(prompt, api_key, save_path, max_retries=3):
    if os.path.exists(save_path):
        return save_path
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                'https://api.minimax.chat/v1/image_generation',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={'model': 'image-01', 'prompt': prompt, 'aspect_ratio': '1:1',
                      'response_format': 'url', 'n': 1},
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
                print(f'  [FAIL] proxy generation: {e}')
                return None


def call_mllm_dual_image(ref_image_path, proxy_image_path, manipulation_text, engine):
    """Call MLLM with two images (reference + proxy) for refinement."""
    ref_url = cloudgpt_api.encode_image(ref_image_path)
    proxy_url = cloudgpt_api.encode_image(proxy_image_path)
    if not ref_url or not proxy_url:
        return ""

    user_prompt = f'''
<Input>
{{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation Text": "{manipulation_text}"
}}
'''

    chat_message = [
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
                engine=engine, messages=chat_message,
                max_tokens=1024, timeout=60, temperature=0)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                print(f'  [FAIL] MLLM refinement: {e}')
                return ""


def parse_refined_description(raw_response, fallback):
    """Extract refined description from MLLM response."""
    resp = raw_response.strip()
    if resp.startswith('```json'):
        resp = resp.replace('```json', '').replace('```', '').strip()
    try:
        d = json.loads(resp)
        return d.get('Refined Target Description', fallback)
    except Exception:
        return fallback


def load_index_features(cache_path):
    if cache_path and os.path.exists(cache_path):
        data = pickle.load(open(cache_path, 'rb'))
        return data['index_features'], data['index_names']
    raise FileNotFoundError(f'Index features cache not found: {cache_path}')


@torch.no_grad()
def encode_texts(clip_model, tokenizer, texts, device, batch_size=32):
    all_feats = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i+batch_size]).to(device)
        feats = clip_model.encode_text(tokens).float().cpu()
        all_feats.append(feats)
    return torch.vstack(all_feats)


def compute_metrics(sorted_index_names, target_names, ks=[1, 5, 10, 50]):
    labels = torch.tensor(
        sorted_index_names == np.repeat(
            np.array(target_names), len(sorted_index_names[0])
        ).reshape(len(target_names), -1))
    return {f'Recall@{k}': (torch.sum(labels[:, :k]) / len(labels)).item() * 100 for k in ks}


def main():
    args = parse_args()
    device = torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load existing first-round results
    print('Loading first-round results...')
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

    # 2. Generate proxy images (named by original index for cache reuse)
    print(f'\nPhase 1: Generating proxy images...')
    proxy_dir = os.path.join(PROXY_CACHE_DIR, args.dataset)
    os.makedirs(proxy_dir, exist_ok=True)

    proxy_paths = []
    for i, s in enumerate(tqdm.tqdm(samples, desc='Generating proxies')):
        orig_idx = sample_indices[i]
        save_path = os.path.join(proxy_dir, f'proxy_{orig_idx:05d}.jpg')
        desc = s.get('target_description', '')
        path = generate_proxy_image(desc, args.minimax_key, save_path)
        proxy_paths.append(path)

    valid_proxies = sum(1 for p in proxy_paths if p)
    print(f'  Proxies ready: {valid_proxies}/{len(samples)}')

    # 3. Second-round MLLM refinement with dual images
    print(f'\nPhase 2: MLLM refinement with reference + proxy images...')
    refined_descriptions = []
    refinement_details = []

    for i, s in enumerate(tqdm.tqdm(samples, desc='Refining descriptions')):
        ref_path = s.get('reference_image_path', '')
        proxy_path = proxy_paths[i]
        instruction = s.get('instruction', '')
        original_desc = s.get('target_description', instruction)

        if not proxy_path or not os.path.exists(str(proxy_path)):
            refined_descriptions.append(original_desc)
            refinement_details.append({'status': 'skipped_no_proxy'})
            continue

        raw_resp = call_mllm_dual_image(ref_path, proxy_path, instruction, args.openai_engine)
        refined = parse_refined_description(raw_resp, fallback=original_desc)
        refined_descriptions.append(refined)

        detail = {
            'index': i,
            'instruction': instruction,
            'original_description': original_desc,
            'refined_description': refined,
            'raw_response': raw_resp,
        }
        refinement_details.append(detail)

        print(f'\n=== Sample {i} ===')
        print(f'  Instruction: {instruction}')
        print(f'  Original:    {original_desc[:80]}...')
        print(f'  Refined:     {refined[:80]}...')

    # 4. Load CLIP and index features
    print(f'\nPhase 3: CLIP encoding and retrieval...')
    clip_model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()
    tokenizer = lambda texts: clip.tokenize(texts, context_length=77, truncate=True)

    index_features, index_names = load_index_features(args.img_features_cache)
    print(f'  Index: {len(index_names)} images')

    original_descriptions = [s.get('target_description', '') for s in samples]
    target_names = [s.get('target_name', '') for s in samples]

    # Encode both original and refined descriptions
    original_features = encode_texts(clip_model, tokenizer, original_descriptions, device)
    refined_features = encode_texts(clip_model, tokenizer, refined_descriptions, device)

    # 5. Evaluate
    index_features_norm = torch.nn.functional.normalize(index_features.float(), dim=-1)

    results = {}
    for label, features in [('baseline (round 1)', original_features),
                            ('refined (round 2)', refined_features)]:
        features_norm = torch.nn.functional.normalize(features.float(), dim=-1)
        sim = features_norm @ index_features_norm.T
        sorted_indices = torch.argsort(1 - sim, dim=-1).cpu()
        sorted_names = np.array(index_names)[sorted_indices]
        metrics = compute_metrics(sorted_names, target_names)
        results[label] = metrics

    # 6. Print comparison
    print('\n' + '=' * 70)
    print(f'Results on {args.dataset} ({len(samples)} samples)')
    print('=' * 70)
    print(f'{"Method":<30} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"R@50":>8}')
    print('-' * 70)
    for method, metrics in results.items():
        print(f'{method:<30} {metrics["Recall@1"]:>8.2f} {metrics["Recall@5"]:>8.2f} '
              f'{metrics["Recall@10"]:>8.2f} {metrics["Recall@50"]:>8.2f}')
    print('=' * 70)

    # 7. Save
    seed_tag = f'_seed{args.random_seed}' if args.random_seed is not None else ''
    output_path = os.path.join(args.output_dir,
                               f'{args.dataset}_refine_{len(samples)}samples{seed_tag}.json')
    payload = {
        'dataset': args.dataset,
        'num_samples': len(samples),
        'random_seed': args.random_seed,
        'sample_indices': sample_indices,
        'method': 'visual_self_refinement',
        'results': results,
        'refinement_details': refinement_details,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    main()
