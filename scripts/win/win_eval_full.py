"""
Windows GPU evaluation script for Three-Way Fusion (full-scale).

Reads all data from Z: (Samba mount to Linux server).
Runs CLIP encoding + retrieval on local RTX 4060 GPU.

Usage:
  D:\env-py311\python.exe win_eval_full.py
  # Optional: force root (SSH 会话无 Z: 时用 D 盘缓存目录)
  set OSRCIR_ROOT=D:\\osrcir_remote

Three-way fusion formula:
  text_feat = normalize(BETA * CLIP(D1) + (1-BETA) * CLIP(D2))
  score = ALPHA * sim(text_feat, gallery) + (1-ALPHA) * sim(proxy_feat, gallery)
"""
import json
import os
import sys
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F

def _detect_root():
    env = os.environ.get('OSRCIR_ROOT', '').strip().strip('"')
    if env:
        pc = os.path.join(env, 'proxy_cache')
        if os.path.isdir(pc):
            return env
        print(f"WARNING: OSRCIR_ROOT set but no proxy_cache under {env!r}, ignoring.")
    if os.path.isdir('Z:/proxy_cache'):
        return 'Z:/'
    if os.path.isdir(r'D:\osrcir_remote\proxy_cache'):
        return r'D:\osrcir_remote'
    if os.path.isdir('/root/osrcir/proxy_cache'):
        return '/root/osrcir'
    print("ERROR: Cannot find project root. Set OSRCIR_ROOT, mount Z:, or use D:\\osrcir_remote layout.")
    sys.exit(1)


ROOT = _detect_root()
DATASETS_ROOT = os.path.join(ROOT, 'datasets') if ROOT != 'Z:/' else 'Z:/datasets'

BETA = 0.7    # D1 weight in description ensemble
ALPHA = 0.9   # text weight in text+proxy fusion


def _path_baseline(name):
    fn = f'{name}_full.json'
    for sub in ('outputs', 'results'):
        p = os.path.join(ROOT, sub, fn)
        if os.path.isfile(p):
            return p
    return os.path.join(ROOT, 'outputs', fn)


def _path_img_features(name):
    fn = f'{name}_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl'
    p1 = os.path.join(ROOT, 'precomputed_cache', 'precomputed', fn)
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(ROOT, 'features', fn)
    if os.path.isfile(p2):
        return p2
    return p1


def _build_dataset_configs():
    rows = [
        ('fashioniq_dress', 'fashioniq'),
        ('fashioniq_shirt', 'fashioniq'),
        ('fashioniq_toptee', 'fashioniq'),
        ('circo', 'circo'),
    ]
    out = []
    for name, typ in rows:
        out.append({
            'name': name,
            'baseline_json': _path_baseline(name),
            'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', f'{name}_v7_refine_cache.json'),
            'proxy_dir': os.path.join(ROOT, 'proxy_cache', name),
            'img_features': _path_img_features(name),
            'type': typ,
        })
    return out


DATASETS = _build_dataset_configs()


def load_clip():
    import clip
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading CLIP ViT-L/14 on {device}...")
    model, preprocess = clip.load('ViT-L/14', device=device, jit=False)
    model.eval()
    tokenizer = lambda texts: clip.tokenize(texts, context_length=77, truncate=True)
    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=64):
    all_feats = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i+batch_size]).to(device)
        feats = model.encode_text(tokens).float().cpu()
        all_feats.append(feats)
    return torch.vstack(all_feats)


@torch.no_grad()
def encode_images(model, preprocess, image_paths, device, batch_size=32):
    import PIL.Image
    all_feats = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        imgs = []
        for p in batch:
            if p and os.path.exists(p):
                try:
                    imgs.append(preprocess(PIL.Image.open(p).convert('RGB')))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            else:
                imgs.append(torch.zeros(3, 224, 224))
        tensor = torch.stack(imgs).to(device)
        feats = model.encode_image(tensor).float().cpu()
        all_feats.append(feats)
    return torch.vstack(all_feats)


def compute_fiq_metrics(sorted_names, target_names, ks=[1, 5, 10, 50]):
    labels = torch.tensor(
        sorted_names == np.repeat(
            np.array(target_names), sorted_names.shape[1]
        ).reshape(len(target_names), -1)
    )
    return {f'R@{k}': (torch.sum(labels[:, :k]) / len(labels)).item() * 100 for k in ks}


def compute_circo_metrics(sorted_names, gt_targets_list, ks=[5, 10, 25, 50]):
    """mAP@k for CIRCO."""
    results = {}
    for k in ks:
        ap_sum = 0
        for i, gts in enumerate(gt_targets_list):
            gt_set = set(str(g) for g in gts if g)
            if not gt_set:
                continue
            hits = 0
            precision_sum = 0
            for j in range(min(k, sorted_names.shape[1])):
                if str(sorted_names[i][j]) in gt_set:
                    hits += 1
                    precision_sum += hits / (j + 1)
            ap = precision_sum / min(len(gt_set), k) if gt_set else 0
            ap_sum += ap
        results[f'mAP@{k}'] = ap_sum / len(gt_targets_list) * 100
    return results


def evaluate_dataset(ds_config, model, preprocess, tokenizer, device):
    name = ds_config['name']
    print(f"\n{'='*70}")
    print(f"  Evaluating: {name}")
    print(f"{'='*70}")

    # 1. Load baseline (D1 descriptions)
    baseline = json.load(open(ds_config['baseline_json'], encoding='utf-8'))
    total = len(baseline)
    print(f"  Baseline samples: {total}")

    # 2. Load V7 refine cache
    refine_list = json.load(open(ds_config['refine_cache'], encoding='utf-8'))
    refine_map = {r['index']: r for r in refine_list}
    print(f"  V7 refinements: {len(refine_map)}")

    # 3. Load gallery features
    gallery_data = pickle.load(open(ds_config['img_features'], 'rb'))
    gallery_features = gallery_data['index_features']
    gallery_names = gallery_data['index_names']
    gallery_features = F.normalize(gallery_features.float(), dim=-1)
    print(f"  Gallery: {len(gallery_names)} images")

    # 4. Build aligned arrays
    d1_texts = []
    d2_texts = []
    proxy_paths = []
    target_names = []
    gt_targets = []  # for CIRCO
    valid_indices = []

    for idx, sample in enumerate(baseline):
        d1 = sample.get('target_description', '')
        if not d1:
            continue

        target = sample.get('target_name', '')
        if ds_config['type'] == 'fashioniq' and not target:
            continue

        d2 = d1  # fallback
        if idx in refine_map:
            d2 = refine_map[idx].get('refined_description', d1)

        proxy_path = os.path.join(ds_config['proxy_dir'], f'proxy_{idx:05d}.jpg')

        d1_texts.append(d1)
        d2_texts.append(d2)
        proxy_paths.append(proxy_path)
        target_names.append(str(target))
        valid_indices.append(idx)

        if ds_config['type'] == 'circo':
            gts = sample.get('gt_target_names', sample.get('ground_truth_candidates', []))
            gt_targets.append(gts)

    n = len(d1_texts)
    print(f"  Valid queries: {n}")

    # 5. CLIP encode
    print("  Encoding D1 texts...")
    d1_feats = F.normalize(encode_texts(model, tokenizer, d1_texts, device).float(), dim=-1)

    print("  Encoding D2 texts...")
    d2_feats = F.normalize(encode_texts(model, tokenizer, d2_texts, device).float(), dim=-1)

    print("  Encoding proxy images...")
    proxy_feats = F.normalize(encode_images(model, preprocess, proxy_paths, device).float(), dim=-1)

    # 6. Three-way fusion
    # text_feat = normalize(BETA * D1 + (1-BETA) * D2)
    ens_feats = F.normalize(BETA * d1_feats + (1 - BETA) * d2_feats, dim=-1)

    # Baseline: pure D1
    sim_baseline = d1_feats @ gallery_features.T
    sorted_baseline = torch.argsort(1 - sim_baseline, dim=-1).cpu()
    names_baseline = np.array(gallery_names)[sorted_baseline]

    # Ensemble (text only, no proxy)
    sim_ens_text = ens_feats @ gallery_features.T
    sorted_ens_text = torch.argsort(1 - sim_ens_text, dim=-1).cpu()
    names_ens_text = np.array(gallery_names)[sorted_ens_text]

    # Three-way: ALPHA * text_sim + (1-ALPHA) * proxy_sim
    sim_proxy = proxy_feats @ gallery_features.T
    sim_threeway = ALPHA * sim_ens_text + (1 - ALPHA) * sim_proxy
    sorted_threeway = torch.argsort(1 - sim_threeway, dim=-1).cpu()
    names_threeway = np.array(gallery_names)[sorted_threeway]

    # 7. Compute metrics
    if ds_config['type'] == 'fashioniq':
        m_base = compute_fiq_metrics(names_baseline, target_names)
        m_ens = compute_fiq_metrics(names_ens_text, target_names)
        m_3way = compute_fiq_metrics(names_threeway, target_names)
        metric_keys = ['R@1', 'R@5', 'R@10', 'R@50']
    elif ds_config['type'] == 'circo':
        m_base = compute_circo_metrics(names_baseline, gt_targets)
        m_ens = compute_circo_metrics(names_ens_text, gt_targets)
        m_3way = compute_circo_metrics(names_threeway, gt_targets)
        metric_keys = ['mAP@5', 'mAP@10', 'mAP@25', 'mAP@50']

    # 8. Print results
    print(f"\n  {'Metric':<12} {'Baseline':>10} {'Ensemble':>10} {'3-Way':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for k in metric_keys:
        b = m_base[k]
        e = m_ens[k]
        t = m_3way[k]
        d = t - b
        marker = '+' if d > 0 else ''
        print(f"  {k:<12} {b:>10.2f} {e:>10.2f} {t:>10.2f} {marker}{d:>9.2f}")

    # 9. Save results
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
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {out_path}")

    return result


def main():
    print("="*70)
    print("  Three-Way Fusion Full-Scale Evaluation")
    print(f"  BETA={BETA}, ALPHA={ALPHA}")
    print(f"  Root: {ROOT}")
    print("="*70)

    model, preprocess, tokenizer, device = load_clip()

    all_results = {}
    for ds in DATASETS:
        if not os.path.exists(ds['baseline_json']):
            print(f"\n  [SKIP] {ds['name']}: baseline not found")
            continue
        if not os.path.exists(ds['refine_cache']):
            print(f"\n  [SKIP] {ds['name']}: V7 cache not found")
            continue
        if not os.path.exists(ds['img_features']):
            print(f"\n  [SKIP] {ds['name']}: gallery features not found")
            continue

        result = evaluate_dataset(ds, model, preprocess, tokenizer, device)
        all_results[ds['name']] = result

    # Final summary
    print(f"\n\n{'='*70}")
    print("  FINAL SUMMARY (Three-Way Fusion, full-scale)")
    print(f"  beta={BETA}, alpha={ALPHA}")
    print(f"{'='*70}")

    for name, r in all_results.items():
        m3 = r['threeway_fusion']
        mb = r['baseline']
        print(f"\n  {name} ({r['num_queries']} queries):")
        for k in m3:
            d = m3[k] - mb[k]
            marker = '+' if d > 0 else ''
            print(f"    {k}: {mb[k]:.2f} -> {m3[k]:.2f} ({marker}{d:.2f})")

    summary_path = os.path.join(ROOT, 'outputs', 'full_pipeline', 'eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
