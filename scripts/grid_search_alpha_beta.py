"""
Grid search over alpha and beta for Three-Way Fusion.
Uses precomputed features — no CLIP model needed, runs in seconds.
"""
import json
import os
import pickle
import gc
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = str(Path(__file__).resolve().parents[1])
CACHE_DIR = os.path.join(ROOT, 'precomputed_cache', 'eval_features')

def _gal(name):
    return os.path.join(ROOT, 'precomputed_cache', 'precomputed',
                        f'{name}_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl')

DATASETS = {
    'fashioniq_dress':  {'gallery_pkl': _gal('fashioniq_dress'),  'type': 'fashioniq', 'primary': 'R@10'},
    'fashioniq_shirt':  {'gallery_pkl': _gal('fashioniq_shirt'),  'type': 'fashioniq', 'primary': 'R@10'},
    'fashioniq_toptee': {'gallery_pkl': _gal('fashioniq_toptee'), 'type': 'fashioniq', 'primary': 'R@10'},
    'circo':            {'gallery_pkl': _gal('circo'),            'type': 'circo',     'primary': 'mAP@10'},
    'cirr':             {'gallery_pkl': _gal('cirr'),             'type': 'cirr',      'primary': 'R@1'},
}


def compute_fiq(sorted_names, targets, ks=(1, 5, 10, 50)):
    ta = np.array(targets)
    return {f'R@{k}': float(np.mean(np.any(sorted_names[:, :k] == ta[:, None], axis=1))) * 100 for k in ks}


def compute_cirr(sorted_names, targets, groups, gnames, ks=(1, 5, 10, 50)):
    ta = np.array(targets)
    r = {f'R@{k}': float(np.mean(np.any(sorted_names[:, :k] == ta[:, None], axis=1))) * 100 for k in ks}
    for k in [1, 2, 3]:
        hits, cnt = 0, 0
        for i, members in enumerate(groups):
            ms = set(str(m) for m in members)
            t = str(targets[i])
            if t not in ms or len(ms) < 2:
                continue
            cnt += 1
            rank = 0
            for name in sorted_names[i]:
                if str(name) in ms:
                    rank += 1
                    if str(name) == t:
                        break
            if rank <= k:
                hits += 1
        r[f'R_sub@{k}'] = (hits / cnt * 100) if cnt > 0 else 0
    return r


def compute_circo(sorted_names, gt_list, ks=(5, 10, 25, 50)):
    r = {}
    for k in ks:
        ap_sum, cnt = 0, 0
        for i, gts in enumerate(gt_list):
            gs = set(str(g) for g in gts if g)
            if not gs:
                continue
            cnt += 1
            h, ps = 0, 0
            for j in range(min(k, sorted_names.shape[1])):
                if str(sorted_names[i][j]) in gs:
                    h += 1
                    ps += h / (j + 1)
            ap_sum += ps / min(len(gs), k)
        r[f'mAP@{k}'] = (ap_sum / cnt * 100) if cnt > 0 else 0
    return r


def load_dataset(ds_name, ds_cfg):
    feat_path = ds_cfg.get('feature_pkl', os.path.join(CACHE_DIR, f'{ds_name}_eval_features.pkl'))
    if not os.path.exists(feat_path):
        return None
    feats = pickle.load(open(feat_path, 'rb'))
    gal = pickle.load(open(ds_cfg['gallery_pkl'], 'rb'))

    d1 = F.normalize(feats['d1_features'].float(), dim=-1)
    d2 = F.normalize(feats['d2_features'].float(), dim=-1)
    proxy = F.normalize(feats['proxy_features'].float(), dim=-1) if 'proxy_features' in feats else None
    gf = F.normalize(gal['index_features'].float(), dim=-1)
    gn = np.array(gal['index_names'])
    meta = feats['meta']

    targets = [m['target_name'] for m in meta]
    gt_targets = [m.get('gt_target_names', []) for m in meta] if ds_cfg['type'] == 'circo' else None
    groups = [m.get('group_members', m.get('ground_truth_candidates', [])) for m in meta] if ds_cfg['type'] == 'cirr' else None

    return {
        'd1': d1, 'd2': d2, 'proxy': proxy, 'gf': gf,
        'gn': gn, 'targets': targets, 'gt_targets': gt_targets, 'groups': groups,
        'type': ds_cfg['type'], 'primary': ds_cfg['primary'],
        'n': len(meta), 'ng': len(gn),
    }


def eval_params(data, alpha, beta):
    ens = F.normalize(beta * data['d1'] + (1 - beta) * data['d2'], dim=-1)
    sim_ens = ens @ data['gf'].T

    if data['proxy'] is not None and alpha < 1.0:
        sim_proxy = data['proxy'] @ data['gf'].T
        sim_final = alpha * sim_ens + (1 - alpha) * sim_proxy
    else:
        sim_final = sim_ens

    idx = torch.argsort(sim_final, dim=-1, descending=True).numpy()
    names = data['gn'][idx]

    if data['type'] == 'fashioniq':
        return compute_fiq(names, data['targets'])
    elif data['type'] == 'cirr':
        return compute_cirr(names, data['targets'], data['groups'], data['gn'])
    else:
        return compute_circo(names, data['gt_targets'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--feature-path', type=str, default=None)
    parser.add_argument('--gallery-pkl', type=str, default=None)
    parser.add_argument('--dataset-type', choices=['fashioniq', 'cirr', 'circo'], default=None)
    parser.add_argument('--dataset-name', type=str, default='custom_experiment')
    parser.add_argument('--primary-metric', type=str, default=None)
    parser.add_argument('--alphas', nargs='+', type=float,
                        default=[0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 1.00])
    parser.add_argument('--betas', nargs='+', type=float,
                        default=[0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00])
    parser.add_argument('--output-json', type=str, default=None)
    args = parser.parse_args()

    alphas = args.alphas
    betas = args.betas
    datasets = DATASETS
    if args.feature_path:
        if not args.gallery_pkl or not args.dataset_type or not args.primary_metric:
            raise ValueError("--feature-path requires --gallery-pkl, --dataset-type and --primary-metric")
        datasets = {
            args.dataset_name: {
                'gallery_pkl': args.gallery_pkl,
                'feature_pkl': args.feature_path,
                'type': args.dataset_type,
                'primary': args.primary_metric,
            }
        }
    elif args.datasets:
        datasets = {name: DATASETS[name] for name in args.datasets if name in DATASETS}

    print(f"Loading datasets...")
    loaded = {}
    for name, cfg in datasets.items():
        d = load_dataset(name, cfg)
        if d:
            loaded[name] = d
            print(f"  {name}: {d['n']} queries, {d['ng']} gallery")

    results = {}
    baseline_vals = {}

    for name, data in loaded.items():
        baseline = eval_params(data, alpha=1.0, beta=1.0)
        pk = data['primary']
        baseline_vals[name] = baseline[pk]

    print(f"\n{'='*80}")
    print(f"  Grid Search: {len(alphas)} alphas x {len(betas)} betas = {len(alphas)*len(betas)} combos")
    print(f"{'='*80}")

    for name, data in loaded.items():
        pk = data['primary']
        bv = baseline_vals[name]
        print(f"\n{'='*70}")
        print(f"  {name}  (primary: {pk}, baseline={bv:.2f})")
        print(f"{'='*70}")

        ab_label = 'a\\b'
        header = f"  {ab_label:<6}"
        for b in betas:
            header += f" {b:>6.2f}"
        print(header)
        print(f"  {'-'*62}")

        best_val, best_a, best_b = bv, 1.0, 1.0
        grid = {}
        for a in alphas:
            row = f"  {a:<6.2f}"
            for b in betas:
                m = eval_params(data, a, b)
                v = m[pk]
                grid[(a, b)] = m
                delta = v - bv
                if v > best_val:
                    best_val, best_a, best_b = v, a, b
                marker = '*' if delta > 0 else ' '
                row += f" {v:>5.2f}{marker}"
            print(row)

        delta = best_val - bv
        print(f"  Best: α={best_a}, β={best_b} → {pk}={best_val:.2f} (baseline {bv:.2f}, Δ={delta:+.2f})")

        results[name] = {
            'baseline': bv,
            'best_alpha': best_a,
            'best_beta': best_b,
            'best_value': best_val,
            'best_delta': delta,
            'current_07_09': eval_params(data, 0.9, 0.7)[pk],
        }

    print(f"\n\n{'='*80}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Dataset':<22} {'Primary':>8} {'Base':>8} {'α=.9β=.7':>10} {'Best':>8} {'BestΔ':>8} {'Best α':>8} {'Best β':>8}")
    print(f"  {'-'*82}")
    for name, r in results.items():
        print(f"  {name:<22} {loaded[name]['primary']:>8} {r['baseline']:>8.2f} "
              f"{r['current_07_09']:>10.2f} {r['best_value']:>8.2f} "
              f"{r['best_delta']:>+8.2f} {r['best_alpha']:>8.2f} {r['best_beta']:>8.2f}")

    out = args.output_json or os.path.join(ROOT, 'outputs', 'full_pipeline', 'grid_search_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == '__main__':
    main()
