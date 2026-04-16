"""
Pre-compute CLIP features for evaluation.
Saves text features (D1, D2) and proxy image features as .pkl files.
Run once, then use eval_from_cache.py for instant evaluation.

Usage:
  python3 -u precompute_features.py --datasets fashioniq_dress fashioniq_shirt fashioniq_toptee circo
"""
import json
import os
import sys
import pickle
import time
import gc
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = str(Path(__file__).resolve().parents[1])
CACHE_DIR = os.path.join(ROOT, 'precomputed_cache', 'eval_features')
os.makedirs(CACHE_DIR, exist_ok=True)

DATASET_CONFIGS = {
    'fashioniq_dress': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'fashioniq_dress_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'fashioniq_dress_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'fashioniq_dress'),
        'type': 'fashioniq',
    },
    'fashioniq_shirt': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'fashioniq_shirt_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'fashioniq_shirt_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'fashioniq_shirt'),
        'type': 'fashioniq',
    },
    'fashioniq_toptee': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'fashioniq_toptee_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'fashioniq_toptee_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'fashioniq_toptee'),
        'type': 'fashioniq',
    },
    'circo': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'circo_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'circo_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'circo'),
        'type': 'circo',
    },
    'cirr': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'cirr', 'cirr_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'cirr_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'cirr'),
        'type': 'cirr',
    },
}


def load_model():
    import open_clip
    LOCAL_WEIGHTS = '/root/.cache/clip/ViT-L-14.pt'
    print("Loading ViT-L-14 from local cache...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    jit_sd = torch.jit.load(LOCAL_WEIGHTS, map_location='cpu').state_dict()
    meta_keys = {'input_resolution', 'context_length', 'vocab_size'}
    filtered_sd = {k: v for k, v in jit_sd.items() if k not in meta_keys}
    model.load_state_dict(filtered_sd, strict=True)
    model.eval()
    del jit_sd, filtered_sd; gc.collect()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, texts, batch_size=32):
    all_feats = []
    total_batches = (len(texts) - 1) // batch_size + 1
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch)
        feats = model.encode_text(tokens).float()
        all_feats.append(feats)
        b = i // batch_size + 1
        if b % 10 == 0 or b == total_batches:
            print(f"    text {b}/{total_batches}", flush=True)
    return torch.vstack(all_feats)


@torch.no_grad()
def encode_images(model, preprocess, image_paths, batch_size=4):
    from PIL import Image
    all_feats = []
    dummy = torch.zeros(3, 224, 224)
    total_batches = (len(image_paths) - 1) // batch_size + 1
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        imgs = []
        for p in batch:
            if p and os.path.exists(p):
                try:
                    imgs.append(preprocess(Image.open(p).convert('RGB')))
                except Exception:
                    imgs.append(dummy)
            else:
                imgs.append(dummy)
        tensor = torch.stack(imgs)
        feats = model.encode_image(tensor).float()
        all_feats.append(feats)
        b = i // batch_size + 1
        if b % 20 == 0 or b == total_batches:
            elapsed = time.time() - encode_images._t0
            eta = elapsed / b * (total_batches - b)
            print(f"    img {b}/{total_batches}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
    return torch.vstack(all_feats)

encode_images._t0 = 0


def prepare_dataset(name, config):
    """Load and align baseline + refine data, return texts and proxy paths."""
    baseline = json.load(open(config['baseline_json']))
    refine_list = json.load(open(config['refine_cache']))
    refine_map = {r['index']: r for r in refine_list}

    d1_texts, d2_texts, proxy_paths = [], [], []
    meta = []

    for idx, sample in enumerate(baseline):
        d1 = sample.get('target_description', '')
        if not d1:
            continue
        target = sample.get('target_name', '')
        if config['type'] in ('fashioniq', 'cirr') and not target:
            continue

        d2 = d1
        if idx in refine_map:
            d2 = refine_map[idx].get('refined_description', d1)

        proxy_path = os.path.join(config['proxy_dir'], f'proxy_{idx:05d}.jpg')

        d1_texts.append(d1)
        d2_texts.append(d2)
        proxy_paths.append(proxy_path)

        sample_meta = {'index': idx, 'target_name': str(target)}
        if config['type'] == 'circo':
            sample_meta['gt_target_names'] = sample.get('gt_target_names', [])
        if config['type'] == 'cirr':
            sample_meta['ground_truth_candidates'] = sample.get('ground_truth_candidates', [])
        meta.append(sample_meta)

    return d1_texts, d2_texts, proxy_paths, meta


def precompute_dataset(name, config, model, preprocess, tokenizer):
    print(f"\n{'='*60}")
    print(f"  Pre-computing: {name}")
    print(f"{'='*60}")

    out_path = os.path.join(CACHE_DIR, f'{name}_eval_features.pkl')
    if os.path.exists(out_path):
        existing = pickle.load(open(out_path, 'rb'))
        has_all = all(k in existing for k in ['d1_features', 'd2_features', 'proxy_features', 'meta'])
        if has_all:
            print(f"  Already computed: {out_path}")
            return

    d1_texts, d2_texts, proxy_paths, meta = prepare_dataset(name, config)
    n = len(d1_texts)
    print(f"  Samples: {n}")

    t0 = time.time()

    print("  Encoding D1 texts...")
    d1_feats = encode_texts(model, tokenizer, d1_texts)
    print(f"  D1 done ({time.time()-t0:.0f}s)")

    print("  Encoding D2 texts...")
    d2_feats = encode_texts(model, tokenizer, d2_texts)
    print(f"  D2 done ({time.time()-t0:.0f}s)")

    print(f"  Encoding {n} proxy images (batch_size=4, CPU)...")
    encode_images._t0 = time.time()
    proxy_feats = encode_images(model, preprocess, proxy_paths)
    print(f"  Proxy done ({time.time()-t0:.0f}s)")

    result = {
        'd1_features': d1_feats,
        'd2_features': d2_feats,
        'proxy_features': proxy_feats,
        'meta': meta,
    }

    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1024/1024:.1f}MB)")

    del d1_feats, d2_feats, proxy_feats, result; gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['fashioniq_dress', 'fashioniq_shirt', 'fashioniq_toptee', 'circo'])
    parser.add_argument('--text-only', action='store_true',
                        help='Only compute text features (skip slow proxy image encoding)')
    args = parser.parse_args()

    model, preprocess, tokenizer = load_model()

    for ds_name in args.datasets:
        if ds_name not in DATASET_CONFIGS:
            print(f"  [SKIP] Unknown: {ds_name}")
            continue
        config = DATASET_CONFIGS[ds_name]
        if not os.path.exists(config['baseline_json']):
            print(f"  [SKIP] {ds_name}: missing baseline")
            continue

        if args.text_only:
            out_path = os.path.join(CACHE_DIR, f'{ds_name}_eval_features.pkl')
            d1_texts, d2_texts, proxy_paths, meta = prepare_dataset(ds_name, config)
            n = len(d1_texts)
            print(f"\n  {ds_name}: {n} samples (text-only mode)")

            t0 = time.time()
            print("  Encoding D1...")
            d1_feats = encode_texts(model, tokenizer, d1_texts)
            print("  Encoding D2...")
            d2_feats = encode_texts(model, tokenizer, d2_texts)
            print(f"  Text done in {time.time()-t0:.0f}s")

            result = {
                'd1_features': d1_feats,
                'd2_features': d2_feats,
                'meta': meta,
            }
            with open(out_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"  Saved (text-only): {out_path}")
            del d1_feats, d2_feats; gc.collect()
        else:
            precompute_dataset(ds_name, config, model, preprocess, tokenizer)

    print("\n  All done!")


if __name__ == '__main__':
    main()
