"""
GeneCIS α/β grid search on FULL data.

Precomputes D1/D2/proxy/gallery features ONCE, then sweeps α×β in pure matrix ops.
Gallery features loaded from existing .pkl cache (produced by eval_genecis.py).

Usage:
  python3 -u scripts/eval/grid_search_genecis.py
  python3 -u scripts/eval/grid_search_genecis.py --datasets genecis_focus_object
"""
import json
import os
import pickle
import time
import gc
import argparse
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENECIS_PATH = os.path.join(ROOT, 'datasets', 'GENECIS')
FEAT_CACHE_DIR = os.path.join(ROOT, 'precomputed_cache', 'genecis')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_CONFIGS = {
    'genecis_change_object': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'genecis_change_object_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline',
                                     'genecis_change_object_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'genecis_change_object'),
        'annotation': os.path.join(GENECIS_PATH, 'genecis', 'change_object.json'),
        'image_type': 'coco',
    },
    'genecis_focus_object': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'genecis_focus_object_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline',
                                     'genecis_focus_object_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'genecis_focus_object'),
        'annotation': os.path.join(GENECIS_PATH, 'genecis', 'focus_object.json'),
        'image_type': 'coco',
    },
    'genecis_change_attribute': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'genecis_change_attribute_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline',
                                     'genecis_change_attribute_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'genecis_change_attribute'),
        'annotation': os.path.join(GENECIS_PATH, 'genecis', 'change_attribute.json'),
        'image_type': 'vg',
    },
    'genecis_focus_attribute': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'genecis_focus_attribute_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline',
                                     'genecis_focus_attribute_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'genecis_focus_attribute'),
        'annotation': os.path.join(GENECIS_PATH, 'genecis', 'focus_attribute.json'),
        'image_type': 'vg',
    },
}

ALPHAS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
BETAS  = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def load_model():
    import open_clip
    print(f"Loading ViT-L-14-quickgelu (openai) on {DEVICE}...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14-quickgelu', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-L-14-quickgelu')
    model = model.to(DEVICE).eval()
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    print(f"  Loaded in {time.time()-t0:.1f}s on {DEVICE}")
    return model, preprocess, tokenizer


def get_gallery_image_path(image_id, image_type):
    if image_type == 'coco':
        return os.path.join(GENECIS_PATH, 'coco2017', 'val2017',
                            f'{int(image_id):012d}.jpg')
    for subdir in ['VG_All', 'VG_100K', 'VG_100K_2']:
        p = os.path.join(GENECIS_PATH, 'Visual_Genome', subdir,
                         f'{image_id}.jpg')
        if os.path.exists(p):
            return p
    return os.path.join(GENECIS_PATH, 'Visual_Genome', 'VG_All',
                        f'{image_id}.jpg')


@torch.no_grad()
def encode_images_batched(model, preprocess, paths, batch_size=32, label="images"):
    import sys
    all_feats = []
    total = len(paths)
    print(f"    {label}: starting {total} images, batch_size={batch_size}", flush=True)
    for i in range(0, total, batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                imgs.append(preprocess(img))
                img.close()
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        t = torch.stack(imgs).to(DEVICE)
        f = model.encode_image(t).float().cpu()
        all_feats.append(f)
        del t, f, imgs
        done = min(i + batch_size, total)
        if done % 200 == 0 or done == total:
            print(f"    {label}: {done}/{total}", flush=True)
            sys.stdout.flush()
    if all_feats:
        return torch.cat(all_feats, dim=0)
    return torch.empty(0, 768)


def collect_unique_gallery_ids(annotation, image_type):
    all_ids = set()
    for ann in annotation:
        gallery = ann.get('gallery', [])
        target = ann.get('target', {})
        if image_type == 'coco':
            all_ids.update(g['val_image_id'] for g in gallery)
            tid = target.get('val_image_id')
        else:
            all_ids.update(g['image_id'] for g in gallery)
            tid = target.get('image_id')
        if tid is not None:
            all_ids.add(tid)
    return sorted(all_ids)


def load_or_compute_gallery(name, image_type, annotation, model, preprocess):
    cache_path = os.path.join(FEAT_CACHE_DIR, f'{name}_gallery.pkl')
    all_ids = collect_unique_gallery_ids(annotation, image_type)
    print(f"  Unique gallery images: {len(all_ids)}")

    if os.path.exists(cache_path):
        cached = pickle.load(open(cache_path, 'rb'))
        n_cached = len(cached['ids'])
        n_needed = len(all_ids)
        if n_cached > 0 and n_cached >= n_needed * 0.85:
            print(f"  Loaded gallery cache ({cache_path}, {n_cached}/{n_needed} ids)")
            return (cached['feats'],
                    {gid: i for i, gid in enumerate(cached['ids'])})
        if n_cached == 0:
            print(f"  Gallery cache is EMPTY — need images on disk to rebuild")
            return None, None
        print(f"  Cache mismatch ({n_cached} vs {n_needed}), recomputing")

    paths, valid_ids = [], []
    for gid in all_ids:
        p = get_gallery_image_path(gid, image_type)
        if os.path.exists(p):
            paths.append(p)
            valid_ids.append(gid)
    print(f"  Found {len(valid_ids)}/{len(all_ids)} images on disk")

    t0 = time.time()
    gf = encode_images_batched(model, preprocess, paths, label="gallery")
    gf = F.normalize(gf, dim=-1)
    print(f"  Gallery encoded in {time.time()-t0:.0f}s")

    with open(cache_path, 'wb') as f:
        pickle.dump({'feats': gf, 'ids': valid_ids}, f)
    return gf, {gid: i for i, gid in enumerate(valid_ids)}


def precompute_features(name, config, model, preprocess, tokenizer):
    """Precompute D1/D2/proxy/gallery features, with per-dataset caching."""
    print(f"\n{'='*60}")
    print(f"  Precomputing features: {name}")
    print(f"{'='*60}")

    feat_cache_path = os.path.join(FEAT_CACHE_DIR, f'{name}_grid_feats.pkl')

    baseline = json.load(open(config['baseline_json']))
    annotation = json.load(open(config['annotation']))
    total = min(len(baseline), len(annotation))
    print(f"  Samples: {total}")

    refine_map = {}
    if os.path.exists(config['refine_cache']):
        refine_list = json.load(open(config['refine_cache']))
        refine_map = {r['index']: r for r in refine_list}
        print(f"  Refinements: {len(refine_map)}")

    gallery_feats, id_to_idx = load_or_compute_gallery(
        name, config['image_type'], annotation[:total], model, preprocess)
    if gallery_feats is None:
        print(f"  SKIPPING {name} — no gallery features available")
        return None
    gc.collect()

    d1_texts, d2_texts = [], []
    for idx in range(total):
        d1 = baseline[idx].get('target_description', '') or ''
        d2 = d1
        if idx in refine_map:
            d2 = refine_map[idx].get('refined_description', d1) or d1
        d1_texts.append(d1)
        d2_texts.append(d2)

    if os.path.exists(feat_cache_path):
        print(f"  Loading cached D1/D2/proxy features...")
        cached = pickle.load(open(feat_cache_path, 'rb'))
        d1_feats = cached['d1_feats']
        d2_feats = cached['d2_feats']
        proxy_feats = cached['proxy_feats']
        print(f"  Loaded from cache: D1={d1_feats.shape} D2={d2_feats.shape} proxy={proxy_feats.shape}")
    else:
        print(f"  Encoding D1/D2 texts on {DEVICE}...")
        with torch.no_grad():
            d1_feats = []
            for i in range(0, total, 256):
                toks = tokenizer(d1_texts[i:i+256]).to(DEVICE)
                d1_feats.append(model.encode_text(toks).float().cpu())
            d1_feats = F.normalize(torch.cat(d1_feats), dim=-1)

            d2_feats = []
            for i in range(0, total, 256):
                toks = tokenizer(d2_texts[i:i+256]).to(DEVICE)
                d2_feats.append(model.encode_text(toks).float().cpu())
            d2_feats = F.normalize(torch.cat(d2_feats), dim=-1)
        print(f"  D1/D2 encoded: {d1_feats.shape}")

        print(f"  Encoding proxy images...")
        proxy_paths = [os.path.join(config['proxy_dir'], f'proxy_{i:05d}.jpg')
                       for i in range(total)]
        with torch.no_grad():
            proxy_feats = encode_images_batched(model, preprocess, proxy_paths,
                                                batch_size=16, label="proxy")
            proxy_feats = F.normalize(proxy_feats, dim=-1)
        print(f"  Proxy encoded: {proxy_feats.shape}")

        with open(feat_cache_path, 'wb') as f:
            pickle.dump({'d1_feats': d1_feats, 'd2_feats': d2_feats,
                         'proxy_feats': proxy_feats}, f)
        sz = os.path.getsize(feat_cache_path) / 1024 / 1024
        print(f"  Saved feature cache: {sz:.1f}MB")

    query_meta = []
    image_type = config['image_type']
    for idx in range(total):
        if not d1_texts[idx]:
            continue
        ann = annotation[idx]
        gi = ann.get('gallery', [])
        ti = ann.get('target', {})
        if image_type == 'coco':
            tid = ti.get('val_image_id')
            gids = [g['val_image_id'] for g in gi]
        else:
            tid = ti.get('image_id')
            gids = [g['image_id'] for g in gi]
        if tid is None or not gids:
            continue
        if tid not in gids:
            gids.append(tid)
        target_pos = gids.index(tid)

        indices = []
        skip = False
        for gid in gids:
            if gid not in id_to_idx:
                skip = True
                break
            indices.append(id_to_idx[gid])
        if skip:
            continue

        query_meta.append({
            'idx': idx,
            'gallery_indices': indices,
            'target_pos': target_pos,
        })

    print(f"  Valid queries: {len(query_meta)}/{total}")

    return {
        'd1_feats': d1_feats,
        'd2_feats': d2_feats,
        'proxy_feats': proxy_feats,
        'gallery_feats': gallery_feats,
        'query_meta': query_meta,
        'total': total,
    }


def evaluate_grid(data, alphas, betas):
    """Precompute per-query local sims ONCE, then sweep α×β with pure arithmetic.
    
    Key insight: for ranking, normalize(β*d1 + (1-β)*d2) @ g has the same order as
    (β*d1 + (1-β)*d2) @ g = β*(d1@g) + (1-β)*(d2@g), because the normalization
    factor is constant across gallery items for a given query.
    """
    d1 = data['d1_feats']
    d2 = data['d2_feats']
    proxy = data['proxy_feats']
    gallery = data['gallery_feats']
    queries = data['query_meta']
    n_valid = len(queries)

    print(f"  Pre-computing per-query local sims ({n_valid} queries)...", flush=True)
    t0 = time.time()

    sims_d1 = []
    sims_d2 = []
    sims_proxy = []
    target_positions = []

    for q in queries:
        idx = q['idx']
        g_idx = torch.tensor(q['gallery_indices'], dtype=torch.long)
        g_local = gallery[g_idx]
        sims_d1.append((d1[idx] @ g_local.T))
        sims_d2.append((d2[idx] @ g_local.T))
        sims_proxy.append((proxy[idx] @ g_local.T))
        target_positions.append(q['target_pos'])

    print(f"  Local sims done in {time.time()-t0:.1f}s", flush=True)

    def rank_metrics(sim_list):
        hits = {1: 0, 2: 0, 3: 0}
        for i, sim in enumerate(sim_list):
            rank = torch.argsort(sim, descending=True)
            pos = (rank == target_positions[i]).nonzero(as_tuple=True)[0].item()
            for k in [1, 2, 3]:
                if pos < k:
                    hits[k] += 1
        return {f'R@{k}': hits[k] / n_valid * 100 for k in [1, 2, 3]}

    baseline = rank_metrics(sims_d1)

    results = []
    for beta in betas:
        ens_sims = [beta * sims_d1[i] + (1 - beta) * sims_d2[i]
                    for i in range(n_valid)]
        ens_m = rank_metrics(ens_sims)

        for alpha in alphas:
            three_sims = [alpha * ens_sims[i] + (1 - alpha) * sims_proxy[i]
                          for i in range(n_valid)]
            three_m = rank_metrics(three_sims)
            results.append({
                'alpha': alpha, 'beta': beta,
                'ensemble': ens_m, 'threeway': three_m,
            })

    return baseline, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
        default=['genecis_change_object', 'genecis_focus_object',
                 'genecis_change_attribute', 'genecis_focus_attribute'])
    args = parser.parse_args()

    model, preprocess, tokenizer = load_model()

    all_grid_results = {}
    for ds in args.datasets:
        if ds not in DATASET_CONFIGS:
            continue
        data = precompute_features(ds, DATASET_CONFIGS[ds], model, preprocess, tokenizer)
        if data is None:
            continue
        gc.collect()

        print(f"\n  Grid search: {len(ALPHAS)} alphas × {len(BETAS)} betas = {len(ALPHAS)*len(BETAS)} combos")
        t0 = time.time()
        baseline, grid = evaluate_grid(data, ALPHAS, BETAS)
        elapsed = time.time() - t0
        print(f"  Grid search done in {elapsed:.1f}s")

        best_r1 = max(grid, key=lambda x: x['threeway']['R@1'])
        best_sum = max(grid, key=lambda x: sum(x['threeway'][f'R@{k}'] for k in [1,2,3]))
        current = [r for r in grid if abs(r['alpha']-0.9)<0.01 and abs(r['beta']-0.7)<0.01][0]

        print(f"\n  {'='*70}")
        print(f"  {ds} GRID SEARCH RESULTS")
        print(f"  {'='*70}")
        print(f"  Baseline:            R@1={baseline['R@1']:.2f}  R@2={baseline['R@2']:.2f}  R@3={baseline['R@3']:.2f}")
        print(f"  Current (β=0.7 α=0.9): R@1={current['threeway']['R@1']:.2f}  R@2={current['threeway']['R@2']:.2f}  R@3={current['threeway']['R@3']:.2f}")
        print(f"  Best R@1 (β={best_r1['beta']:.2f} α={best_r1['alpha']:.2f}): R@1={best_r1['threeway']['R@1']:.2f}  R@2={best_r1['threeway']['R@2']:.2f}  R@3={best_r1['threeway']['R@3']:.2f}")
        print(f"  Best sum (β={best_sum['beta']:.2f} α={best_sum['alpha']:.2f}): R@1={best_sum['threeway']['R@1']:.2f}  R@2={best_sum['threeway']['R@2']:.2f}  R@3={best_sum['threeway']['R@3']:.2f}")

        print(f"\n  --- Full grid (3-way, sorted by R@1 desc) ---")
        print(f"  {'β':>5} {'α':>5}  {'R@1':>7} {'R@2':>7} {'R@3':>7}  {'ΔR@1':>7} {'ΔR@2':>7} {'ΔR@3':>7}")
        sorted_grid = sorted(grid, key=lambda x: x['threeway']['R@1'], reverse=True)
        for r in sorted_grid[:25]:
            t = r['threeway']
            d1 = t['R@1'] - baseline['R@1']
            d2 = t['R@2'] - baseline['R@2']
            d3 = t['R@3'] - baseline['R@3']
            s1 = '+' if d1 > 0 else ''
            s2 = '+' if d2 > 0 else ''
            s3 = '+' if d3 > 0 else ''
            print(f"  {r['beta']:>5.2f} {r['alpha']:>5.2f}  {t['R@1']:>7.2f} {t['R@2']:>7.2f} {t['R@3']:>7.2f}  {s1}{d1:>6.2f} {s2}{d2:>6.2f} {s3}{d3:>6.2f}")

        # Also show ensemble-only (α=1.0) results
        print(f"\n  --- Ensemble only (α=1.0, no proxy) ---")
        ens_only = [r for r in grid if abs(r['alpha']-1.0)<0.01]
        ens_only.sort(key=lambda x: x['ensemble']['R@1'], reverse=True)
        for r in ens_only:
            e = r['ensemble']
            d1 = e['R@1'] - baseline['R@1']
            d2 = e['R@2'] - baseline['R@2']
            d3 = e['R@3'] - baseline['R@3']
            s1 = '+' if d1 > 0 else ''
            s2 = '+' if d2 > 0 else ''
            s3 = '+' if d3 > 0 else ''
            print(f"  β={r['beta']:>5.2f}: R@1={e['R@1']:>7.2f} R@2={e['R@2']:>7.2f} R@3={e['R@3']:>7.2f}  Δ={s1}{d1:.2f}/{s2}{d2:.2f}/{s3}{d3:.2f}")

        all_grid_results[ds] = {
            'baseline': baseline,
            'current_params': {'beta': 0.7, 'alpha': 0.9},
            'current_threeway': current['threeway'],
            'best_r1_params': {'beta': best_r1['beta'], 'alpha': best_r1['alpha']},
            'best_r1_threeway': best_r1['threeway'],
            'best_sum_params': {'beta': best_sum['beta'], 'alpha': best_sum['alpha']},
            'best_sum_threeway': best_sum['threeway'],
            'full_grid': grid,
            'valid': len(data['query_meta']),
            'total': data['total'],
        }

        del data
        gc.collect()

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  FINAL SUMMARY — GeneCIS Full-Scale Grid Search")
    print(f"{'='*70}")
    for ds, r in all_grid_results.items():
        b = r['baseline']
        c = r['current_threeway']
        best = r['best_r1_threeway']
        bp = r['best_r1_params']
        d_cur = c['R@1'] - b['R@1']
        d_best = best['R@1'] - b['R@1']
        sc = '+' if d_cur > 0 else ''
        sb = '+' if d_best > 0 else ''
        print(f"\n  {ds} ({r['valid']}/{r['total']}):")
        print(f"    Baseline:          R@1={b['R@1']:.2f}")
        print(f"    Current β=0.7 α=0.9: R@1={c['R@1']:.2f} ({sc}{d_cur:.2f})")
        print(f"    Best β={bp['beta']:.2f} α={bp['alpha']:.2f}: R@1={best['R@1']:.2f} ({sb}{d_best:.2f})")

    out_path = os.path.join(ROOT, 'outputs', 'full_pipeline', 'genecis_grid_search_full.json')
    save_data = {}
    for ds, r in all_grid_results.items():
        save_data[ds] = {k: v for k, v in r.items() if k != 'full_grid'}
        save_data[ds]['grid_size'] = f"{len(ALPHAS)}x{len(BETAS)}"
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Also save full grid for detailed analysis
    full_path = os.path.join(ROOT, 'outputs', 'full_pipeline', 'genecis_grid_search_full_detail.json')
    with open(full_path, 'w') as f:
        json.dump(all_grid_results, f, indent=2, default=str)
    print(f"  Saved detail: {full_path}")


if __name__ == '__main__':
    main()
