"""
Visual Proxy Pipeline for OSrCIR.
Generates proxy images from target descriptions using text-to-image API,
then performs hybrid (text + image) retrieval with CLIP.
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
import io
import clip

PROXY_CACHE_DIR = './proxy_cache'


def parse_args():
    parser = argparse.ArgumentParser('Visual Proxy Retrieval')
    parser.add_argument('--results_json', type=str, required=True,
                        help='Path to existing experiment results JSON (e.g. outputs/fashioniq_dress_full.json)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
                                 'cirr', 'circo'])
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--clip_model', type=str, default='ViT-L/14')
    parser.add_argument('--img_features_cache', type=str, default=None,
                        help='Path to cached index image features .pkl')
    parser.add_argument('--minimax_key', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7, 1.0],
                        help='Hybrid weights to evaluate. 0=pure image, 1=pure text')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed for sampling. If set, randomly sample max_samples from dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/proxy_experiments')
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def generate_proxy_image(prompt, api_key, save_path, max_retries=3):
    """Call MiniMax image-01 API and download the generated image."""
    if os.path.exists(save_path):
        return save_path

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                'https://api.minimax.chat/v1/image_generation',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'image-01',
                    'prompt': prompt,
                    'aspect_ratio': '1:1',
                    'response_format': 'url',
                    'n': 1
                },
                timeout=60
            )

            if resp.status_code != 200:
                raise Exception(f'HTTP {resp.status_code}: {resp.text[:200]}')

            result = resp.json()
            if result.get('base_resp', {}).get('status_code', -1) != 0:
                raise Exception(f'API error: {result.get("base_resp", {})}')

            image_url = result['data']['image_urls'][0]
            img_resp = requests.get(image_url, timeout=30)
            img_resp.raise_for_status()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(img_resp.content)
            return save_path

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f'  [RETRY] {e}, waiting {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [FAIL] {e}')
                return None


def load_clip_model(model_name, device):
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    tokenizer = lambda texts: clip.tokenize(texts, context_length=77, truncate=True)
    return model, preprocess, tokenizer


def load_index_features(cache_path, dataset, dataset_path, clip_model, preprocess, device, batch_size):
    """Load or compute index image features."""
    if cache_path and os.path.exists(cache_path):
        print(f'Loading cached index features from {cache_path}')
        data = pickle.load(open(cache_path, 'rb'))
        return data['index_features'], data['index_names']

    import datasets as ds
    import data_utils
    if 'fashioniq' in dataset:
        dress_type = dataset.split('_')[-1]
        target_ds = ds.FashionIQDataset(dataset_path, 'val', [dress_type], 'classic', preprocess=preprocess)
    elif dataset == 'cirr':
        target_ds = ds.CIRRDataset(dataset_path, 'val', 'classic', preprocess=preprocess)
    elif dataset == 'circo':
        target_ds = ds.CIRCODataset(dataset_path, 'val', 'classic', preprocess=preprocess)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    loader = torch.utils.data.DataLoader(
        target_ds, batch_size=batch_size, num_workers=4,
        collate_fn=data_utils.collate_fn, shuffle=False)

    index_features, index_names = [], []
    for batch in tqdm.tqdm(loader, desc='Encoding index images'):
        if 'image' in batch:
            images = batch['image']
            names = batch['image_name']
        elif 'image_path' in batch:
            processed = []
            valid_names = []
            for path, name in zip(batch['image_path'], batch['image_name']):
                try:
                    img = preprocess(PIL.Image.open(path).convert('RGB'))
                    processed.append(img)
                    valid_names.append(name)
                except Exception:
                    continue
            if not processed:
                continue
            images = torch.stack(processed)
            names = valid_names
        else:
            continue

        with torch.no_grad():
            feats = clip_model.encode_image(images.to(device)).float().cpu()
        index_features.append(feats)
        index_names.extend(names)

    index_features = torch.vstack(index_features)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        pickle.dump({'index_features': index_features, 'index_names': index_names},
                    open(cache_path, 'wb'))
    return index_features, index_names


@torch.no_grad()
def encode_texts(clip_model, tokenizer, texts, device, batch_size=32):
    all_feats = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i+batch_size]).to(device)
        feats = clip_model.encode_text(tokens).float().cpu()
        all_feats.append(feats)
    return torch.vstack(all_feats)


@torch.no_grad()
def encode_images(clip_model, preprocess, image_paths, device, batch_size=16):
    all_feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            if p and os.path.exists(p):
                imgs.append(preprocess(PIL.Image.open(p).convert('RGB')))
            else:
                imgs.append(torch.zeros(3, 224, 224))
        batch_tensor = torch.stack(imgs).to(device)
        feats = clip_model.encode_image(batch_tensor).float().cpu()
        all_feats.append(feats)
    return torch.vstack(all_feats)


def compute_fiq_metrics(sorted_index_names, target_names, ks=[1, 5, 10, 50]):
    labels = torch.tensor(
        sorted_index_names == np.repeat(
            np.array(target_names), len(sorted_index_names[0])
        ).reshape(len(target_names), -1)
    )
    return {f'Recall@{k}': (torch.sum(labels[:, :k]) / len(labels)).item() * 100 for k in ks}


def run_retrieval(text_features, proxy_features, index_features, index_names,
                  target_names, alpha_values):
    """Run retrieval with different alpha values and return metrics."""
    text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    proxy_features = torch.nn.functional.normalize(proxy_features.float(), dim=-1)
    index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)

    text_sim = text_features @ index_features.T
    img_sim = proxy_features @ index_features.T

    results = {}
    for alpha in alpha_values:
        hybrid_sim = alpha * text_sim + (1 - alpha) * img_sim
        sorted_indices = torch.argsort(1 - hybrid_sim, dim=-1).cpu()
        sorted_names = np.array(index_names)[sorted_indices]
        metrics = compute_fiq_metrics(sorted_names, target_names)
        label = f'alpha={alpha:.1f}'
        if alpha == 1.0:
            label += ' (pure text)'
        elif alpha == 0.0:
            label += ' (pure proxy)'
        results[label] = metrics
    return results


def main():
    args = parse_args()
    device = torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(PROXY_CACHE_DIR, exist_ok=True)

    # 1. Load existing results
    print('Loading existing results...')
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
    print(f'  {len(samples)} samples loaded')

    target_descriptions = [s['target_description'] for s in samples]
    target_names = [s.get('target_name', '') for s in samples]

    # 2. Generate proxy images (named by original dataset index for cache reuse)
    print(f'\nGenerating {len(samples)} proxy images via MiniMax...')
    proxy_dir = os.path.join(PROXY_CACHE_DIR, args.dataset)
    os.makedirs(proxy_dir, exist_ok=True)
    proxy_paths = []

    for i, desc in enumerate(tqdm.tqdm(target_descriptions, desc='Generating proxy images')):
        orig_idx = sample_indices[i]
        save_path = os.path.join(proxy_dir, f'proxy_{orig_idx:05d}.jpg')
        result = generate_proxy_image(desc, args.minimax_key, save_path)
        proxy_paths.append(result)

    valid_count = sum(1 for p in proxy_paths if p is not None)
    print(f'  Generated: {valid_count}/{len(samples)}')

    # 3. Load CLIP
    print(f'\nLoading CLIP {args.clip_model}...')
    clip_model, preprocess, tokenizer = load_clip_model(args.clip_model, device)

    # 4. Load/compute index features
    index_features, index_names = load_index_features(
        args.img_features_cache, args.dataset, args.dataset_path,
        clip_model, preprocess, device, args.batch_size)
    print(f'  Index: {len(index_names)} images, features shape: {index_features.shape}')

    # 5. Encode text features
    print('\nEncoding target descriptions with CLIP text encoder...')
    text_features = encode_texts(clip_model, tokenizer, target_descriptions, device)

    # 6. Encode proxy image features
    print('Encoding proxy images with CLIP image encoder...')
    proxy_features = encode_images(clip_model, preprocess, proxy_paths, device)

    # 7. Run retrieval with different alpha values
    print(f'\nRunning retrieval with alpha values: {args.alpha}')
    all_results = run_retrieval(
        text_features, proxy_features, index_features, index_names,
        target_names, args.alpha)

    # 8. Print results
    print('\n' + '=' * 70)
    print(f'Results on {args.dataset} ({len(samples)} samples)')
    print('=' * 70)
    print(f'{"Config":<30} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"R@50":>8}')
    print('-' * 70)
    for config, metrics in all_results.items():
        print(f'{config:<30} {metrics["Recall@1"]:>8.2f} {metrics["Recall@5"]:>8.2f} '
              f'{metrics["Recall@10"]:>8.2f} {metrics["Recall@50"]:>8.2f}')
    print('=' * 70)

    # 9. Save results
    seed_tag = f'_seed{args.random_seed}' if args.random_seed is not None else ''
    output_path = os.path.join(args.output_dir,
                               f'{args.dataset}_proxy_{len(samples)}samples{seed_tag}.json')
    save_payload = {
        'dataset': args.dataset,
        'num_samples': len(samples),
        'random_seed': args.random_seed,
        'sample_indices': sample_indices,
        'clip_model': args.clip_model,
        'proxy_model': 'minimax-image-01',
        'results': {k: v for k, v in all_results.items()},
    }
    with open(output_path, 'w') as f:
        json.dump(save_payload, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    main()
