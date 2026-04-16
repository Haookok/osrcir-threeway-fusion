"""
CPU evaluation script for Three-Way Fusion (full-scale).
Uses open_clip (ViT-L-14 openai weights = identical to original CLIP ViT-L/14).
Optimized for low-memory server (3.6GB RAM + swap).

Usage:
  python3 -u eval_cpu.py [--datasets circo fashioniq_dress ...]

Three-way fusion formula:
  text_feat = normalize(BETA * CLIP(D1) + (1-BETA) * CLIP(D2))
  score = ALPHA * sim(text_feat, gallery) + (1-ALPHA) * sim(proxy_feat, gallery)
"""
import json
import os
import sys
import pickle
import time
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F

ROOT = '/root/osrcir'

BETA = 0.7
ALPHA = 0.9

DATASET_CONFIGS = {
    'fashioniq_dress': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'fashioniq_dress_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'fashioniq_dress_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'fashioniq_dress'),
        'img_features': os.path.join(ROOT, 'precomputed_cache', 'precomputed',
                                     'fashioniq_dress_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl'),
        'type': 'fashioniq',
    },
    'fashioniq_shirt': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'fashioniq_shirt_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'fashioniq_shirt_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'fashioniq_shirt'),
        'img_features': os.path.join(ROOT, 'precomputed_cache', 'precomputed',
                                     'fashioniq_shirt_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl'),
        'type': 'fashioniq',
    },
    'fashioniq_toptee': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'fashioniq_toptee_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'fashioniq_toptee_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'fashioniq_toptee'),
        'img_features': os.path.join(ROOT, 'precomputed_cache', 'precomputed',
                                     'fashioniq_toptee_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl'),
        'type': 'fashioniq',
    },
    'circo': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'circo_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'circo_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'circo'),
        'img_features': os.path.join(ROOT, 'precomputed_cache', 'precomputed',
                                     'circo_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl'),
        'type': 'circo',
    },
}


def load_model():
    import open_clip
    LOCAL_WEIGHTS = '/root/.cache/clip/ViT-L-14.pt'
    print("Loading ViT-L-14 from local cache via open_clip (no network)...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    jit_sd = torch.jit.load(LOCAL_WEIGHTS, map_location='cpu').state_dict()
    meta_keys = {'input_resolution', 'context_length', 'vocab_size'}
    filtered_sd = {k: v for k, v in jit_sd.items() if k not in meta_keys}
    model.load_state_dict(filtered_sd, strict=True)
    model.eval()
    del jit_sd, filtered_sd; gc.collect()
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, texts, batch_size=32):
    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch)
        feats = model.encode_text(tokens).float()
        all_feats.append(feats)
        if (i // batch_size) % 20 == 0:
            print(f"    text batch {i//batch_size+1}/{(len(texts)-1)//batch_size+1}", flush=True)
    return torch.vstack(all_feats)


@torch.no_grad()
def encode_images(model, preprocess, image_paths, batch_size=8):
    from PIL import Image
    all_feats = []
    dummy = torch.zeros(3, 224, 224)
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
        if (i // batch_size) % 50 == 0:
            print(f"    image batch {i//batch_size+1}/{(len(image_paths)-1)//batch_size+1}", flush=True)
    return torch.vstack(all_feats)


def compute_fiq_metrics(sorted_names, target_names, ks=(1, 5, 10, 50)):
    targets_arr = np.array(target_names)
    results = {}
    for k in ks:
        topk = sorted_names[:, :k]
        hits = np.any(topk == targets_arr[:, None], axis=1)
        results[f'R@{k}'] = float(np.mean(hits)) * 100
    return results


def compute_circo_metrics(sorted_names, gt_targets_list, ks=(5, 10, 25, 50)):
    results = {}
    for k in ks:
        ap_sum = 0
        count = 0
        for i, gts in enumerate(gt_targets_list):
            gt_set = set(str(g) for g in gts if g)
            if not gt_set:
                continue
            count += 1
            hits = 0
            precision_sum = 0
            for j in range(min(k, sorted_names.shape[1])):
                if str(sorted_names[i][j]) in gt_set:
                    hits += 1
                    precision_sum += hits / (j + 1)
            ap = precision_sum / min(len(gt_set), k)
            ap_sum += ap
        results[f'mAP@{k}'] = (ap_sum / count * 100) if count > 0 else 0
    return results


def evaluate_dataset(name, config, model, preprocess, tokenizer):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    baseline = json.load(open(config['baseline_json']))
    print(f"  Baseline: {len(baseline)} samples")

    refine_list = json.load(open(config['refine_cache']))
    refine_map = {r['index']: r for r in refine_list}
    print(f"  V7 refine: {len(refine_map)} entries")

    gallery_data = pickle.load(open(config['img_features'], 'rb'))
    gallery_features = F.normalize(gallery_data['index_features'].float(), dim=-1)
    gallery_names = gallery_data['index_names']
    print(f"  Gallery: {len(gallery_names)} images, feat shape={gallery_features.shape}")

    d1_texts, d2_texts, proxy_paths = [], [], []
    target_names, gt_targets = [], []

    for idx, sample in enumerate(baseline):
        d1 = sample.get('target_description', '')
        if not d1:
            continue
        target = sample.get('target_name', '')
        if config['type'] == 'fashioniq' and not target:
            continue

        d2 = d1
        if idx in refine_map:
            d2 = refine_map[idx].get('refined_description', d1)

        proxy_path = os.path.join(config['proxy_dir'], f'proxy_{idx:05d}.jpg')

        d1_texts.append(d1)
        d2_texts.append(d2)
        proxy_paths.append(proxy_path)
        target_names.append(str(target))

        if config['type'] == 'circo':
            gt_targets.append(sample.get('gt_target_names', []))

    n = len(d1_texts)
    print(f"  Valid queries: {n}")

    t0 = time.time()
    print("  Encoding D1 texts...")
    d1_feats = F.normalize(encode_texts(model, tokenizer, d1_texts).float(), dim=-1)
    print(f"  Encoding D2 texts...")
    d2_feats = F.normalize(encode_texts(model, tokenizer, d2_texts).float(), dim=-1)
    print(f"  Encoding proxy images... (batch_size=8, this may take a while on CPU)")
    proxy_feats = F.normalize(encode_images(model, preprocess, proxy_paths).float(), dim=-1)
    print(f"  Encoding done in {time.time()-t0:.0f}s")

    ens_feats = F.normalize(BETA * d1_feats + (1 - BETA) * d2_feats, dim=-1)

    print("  Computing similarities...")

    sim_baseline = d1_feats @ gallery_features.T
    idx_baseline = torch.argsort(sim_baseline, dim=-1, descending=True).numpy()
    names_baseline = np.array(gallery_names)[idx_baseline]
    del sim_baseline; gc.collect()

    sim_ens_text = ens_feats @ gallery_features.T
    idx_ens = torch.argsort(sim_ens_text, dim=-1, descending=True).numpy()
    names_ens_text = np.array(gallery_names)[idx_ens]

    sim_proxy = proxy_feats @ gallery_features.T
    sim_threeway = ALPHA * sim_ens_text + (1 - ALPHA) * sim_proxy
    del sim_ens_text, sim_proxy; gc.collect()

    idx_3way = torch.argsort(sim_threeway, dim=-1, descending=True).numpy()
    names_threeway = np.array(gallery_names)[idx_3way]
    del sim_threeway; gc.collect()

    if config['type'] == 'fashioniq':
        m_base = compute_fiq_metrics(names_baseline, target_names)
        m_ens = compute_fiq_metrics(names_ens_text, target_names)
        m_3way = compute_fiq_metrics(names_threeway, target_names)
        metric_keys = ['R@1', 'R@5', 'R@10', 'R@50']
    else:
        m_base = compute_circo_metrics(names_baseline, gt_targets)
        m_ens = compute_circo_metrics(names_ens_text, gt_targets)
        m_3way = compute_circo_metrics(names_threeway, gt_targets)
        metric_keys = ['mAP@5', 'mAP@10', 'mAP@25', 'mAP@50']

    del names_baseline, names_ens_text, names_threeway
    del idx_baseline, idx_ens, idx_3way
    del d1_feats, d2_feats, proxy_feats, ens_feats, gallery_features
    gc.collect()

    print(f"\n  {'Metric':<12} {'Baseline':>10} {'Ensemble':>10} {'3-Way':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for k in metric_keys:
        b, e, t = m_base[k], m_ens[k], m_3way[k]
        d = t - b
        sign = '+' if d > 0 else ''
        print(f"  {k:<12} {b:>10.2f} {e:>10.2f} {t:>10.2f} {sign}{d:>9.2f}")

    result = {
        'dataset': name,
        'num_queries': n,
        'num_gallery': len(gallery_names),
        'params': {'beta': BETA, 'alpha': ALPHA},
        'baseline': m_base,
        'ensemble_text_only': m_ens,
        'threeway_fusion': m_3way,
    }
    out_path = os.path.join(ROOT, 'outputs', 'full_pipeline', f'{name}_eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['fashioniq_dress', 'fashioniq_shirt', 'fashioniq_toptee', 'circo'])
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Three-Way Fusion Full-Scale Evaluation (CPU)")
    print(f"  BETA={BETA}, ALPHA={ALPHA}")
    print(f"{'='*60}")

    model, preprocess, tokenizer = load_model()

    all_results = {}
    for ds_name in args.datasets:
        if ds_name not in DATASET_CONFIGS:
            print(f"\n  [SKIP] Unknown dataset: {ds_name}")
            continue
        config = DATASET_CONFIGS[ds_name]
        for key in ['baseline_json', 'refine_cache', 'img_features']:
            if not os.path.exists(config[key]):
                print(f"\n  [SKIP] {ds_name}: missing {key} at {config[key]}")
                break
        else:
            result = evaluate_dataset(ds_name, config, model, preprocess, tokenizer)
            all_results[ds_name] = result
            gc.collect()

    print(f"\n\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"  BETA={BETA}, ALPHA={ALPHA}")
    print(f"{'='*60}")
    for name, r in all_results.items():
        m3 = r['threeway_fusion']
        mb = r['baseline']
        print(f"\n  {name} ({r['num_queries']} queries, {r['num_gallery']} gallery):")
        for k in m3:
            d = m3[k] - mb[k]
            sign = '+' if d > 0 else ''
            print(f"    {k}: {mb[k]:.2f} -> {m3[k]:.2f} ({sign}{d:.2f})")

    summary_path = os.path.join(ROOT, 'outputs', 'full_pipeline', 'eval_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
