"""
Evaluate Three-Way Fusion from pre-computed features.
No CLIP model needed -- pure matrix operations, finishes in seconds.

Usage:
  python3 eval_from_cache.py                          # full 3-way (needs proxy features)
  python3 eval_from_cache.py --text-only              # baseline + ensemble only
  python3 eval_from_cache.py --alpha 0.85 --beta 0.7  # sweep params
  python3 eval_from_cache.py --datasets cirr          # single dataset
"""
import json
import os
import pickle
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = str(Path(__file__).resolve().parents[2])
CACHE_DIR = os.path.join(ROOT, 'precomputed_cache', 'eval_features')

def _gal(name):
    return os.path.join(ROOT, 'precomputed_cache', 'precomputed',
                        f'{name}_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl')

DATASETS = {
    'fashioniq_dress':  {'gallery_pkl': _gal('fashioniq_dress'),  'type': 'fashioniq'},
    'fashioniq_shirt':  {'gallery_pkl': _gal('fashioniq_shirt'),  'type': 'fashioniq'},
    'fashioniq_toptee': {'gallery_pkl': _gal('fashioniq_toptee'), 'type': 'fashioniq'},
    'circo':            {'gallery_pkl': _gal('circo'),            'type': 'circo'},
    'cirr':             {'gallery_pkl': _gal('cirr'),             'type': 'cirr'},
}


def compute_fiq_metrics(sorted_names, target_names, ks=(1, 5, 10, 50)):
    targets_arr = np.array(target_names)
    results = {}
    for k in ks:
        topk = sorted_names[:, :k]
        hits = np.any(topk == targets_arr[:, None], axis=1)
        results[f'R@{k}'] = float(np.mean(hits)) * 100
    return results


def compute_cirr_metrics(sorted_names, target_names, group_members_list, gallery_names, ks=(1, 5, 10, 50)):
    targets_arr = np.array(target_names)
    results = {}
    for k in ks:
        topk = sorted_names[:, :k]
        hits = np.any(topk == targets_arr[:, None], axis=1)
        results[f'R@{k}'] = float(np.mean(hits)) * 100

    for k in [1, 2, 3]:
        subset_hits = 0
        count = 0
        for i, members in enumerate(group_members_list):
            member_set = set(str(m) for m in members)
            target = str(target_names[i])
            if target not in member_set or len(member_set) < 2:
                continue
            count += 1
            row = sorted_names[i]
            rank_in_group = 0
            for name in row:
                if str(name) in member_set:
                    rank_in_group += 1
                    if str(name) == target:
                        break
            if rank_in_group <= k:
                subset_hits += 1
        results[f'R_sub@{k}'] = (subset_hits / count * 100) if count > 0 else 0
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


def evaluate(ds_name, ds_config, alpha, beta, text_only=False):
    feat_path = ds_config.get('feature_pkl', os.path.join(CACHE_DIR, f'{ds_name}_eval_features.pkl'))
    if not os.path.exists(feat_path):
        print(f"  [SKIP] {ds_name}: features not found at {feat_path}")
        return None

    feats = pickle.load(open(feat_path, 'rb'))
    gallery_pkl = ds_config['gallery_pkl']
    gallery_data = pickle.load(open(gallery_pkl, 'rb'))

    d1 = F.normalize(feats['d1_features'].float(), dim=-1)
    d2 = F.normalize(feats['d2_features'].float(), dim=-1)
    meta = feats['meta']
    has_proxy = 'proxy_features' in feats and not text_only

    gallery_features = F.normalize(gallery_data['index_features'].float(), dim=-1)
    gallery_names = np.array(gallery_data['index_names'])

    target_names = [m['target_name'] for m in meta]
    gt_targets = [m.get('gt_target_names', []) for m in meta] if ds_config['type'] == 'circo' else None
    group_members = [m.get('group_members', m.get('ground_truth_candidates', [])) for m in meta] if ds_config['type'] == 'cirr' else None
    n = len(meta)

    ens = F.normalize(beta * d1 + (1 - beta) * d2, dim=-1)

    sim_base = d1 @ gallery_features.T
    sim_ens = ens @ gallery_features.T

    idx_base = torch.argsort(sim_base, dim=-1, descending=True).numpy()
    names_base = gallery_names[idx_base]

    idx_ens = torch.argsort(sim_ens, dim=-1, descending=True).numpy()
    names_ens = gallery_names[idx_ens]

    if has_proxy:
        proxy = F.normalize(feats['proxy_features'].float(), dim=-1)
        sim_proxy = proxy @ gallery_features.T
        sim_3way = alpha * sim_ens + (1 - alpha) * sim_proxy
        idx_3way = torch.argsort(sim_3way, dim=-1, descending=True).numpy()
        names_3way = gallery_names[idx_3way]
        del sim_proxy, sim_3way; gc.collect()

    del sim_base, sim_ens, gallery_features; gc.collect()

    if ds_config['type'] == 'fashioniq':
        m_base = compute_fiq_metrics(names_base, target_names)
        m_ens = compute_fiq_metrics(names_ens, target_names)
        m_3way = compute_fiq_metrics(names_3way, target_names) if has_proxy else None
        metric_keys = ['R@1', 'R@5', 'R@10', 'R@50']
    elif ds_config['type'] == 'cirr':
        m_base = compute_cirr_metrics(names_base, target_names, group_members, gallery_names)
        m_ens = compute_cirr_metrics(names_ens, target_names, group_members, gallery_names)
        m_3way = compute_cirr_metrics(names_3way, target_names, group_members, gallery_names) if has_proxy else None
        metric_keys = ['R@1', 'R@5', 'R@10', 'R@50', 'R_sub@1', 'R_sub@2', 'R_sub@3']
    else:
        m_base = compute_circo_metrics(names_base, gt_targets)
        m_ens = compute_circo_metrics(names_ens, gt_targets)
        m_3way = compute_circo_metrics(names_3way, gt_targets) if has_proxy else None
        metric_keys = ['mAP@5', 'mAP@10', 'mAP@25', 'mAP@50']

    print(f"\n  {ds_name} ({n} queries, {len(gallery_names)} gallery)")
    header = f"  {'Metric':<12} {'Baseline':>10} {'Ensemble':>10}"
    if has_proxy:
        header += f" {'3-Way':>10} {'Delta':>10}"
    print(header)
    print(f"  {'-'*55}")

    for k in metric_keys:
        b, e = m_base[k], m_ens[k]
        line = f"  {k:<12} {b:>10.2f} {e:>10.2f}"
        if has_proxy:
            t = m_3way[k]
            d = t - b
            sign = '+' if d > 0 else ''
            line += f" {t:>10.2f} {sign}{d:>9.2f}"
        print(line)

    result = {
        'dataset': ds_name,
        'num_queries': n,
        'num_gallery': len(gallery_names),
        'params': {'beta': beta, 'alpha': alpha},
        'baseline': m_base,
        'ensemble_text_only': m_ens,
    }
    if has_proxy:
        result['threeway_fusion'] = m_3way
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['fashioniq_dress', 'fashioniq_shirt', 'fashioniq_toptee', 'circo', 'cirr'])
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--feature-path', type=str, default=None)
    parser.add_argument('--gallery-pkl', type=str, default=None)
    parser.add_argument('--dataset-type', choices=['fashioniq', 'cirr', 'circo'], default=None)
    parser.add_argument('--dataset-name', type=str, default='custom_experiment')
    parser.add_argument('--summary-out', type=str, default=None)
    args = parser.parse_args()

    print(f"{'='*60}")
    mode = "Text-only" if args.text_only else "Three-Way Fusion"
    print(f"  {mode} Evaluation from Cache")
    print(f"  BETA={args.beta}, ALPHA={args.alpha}")
    print(f"{'='*60}")

    all_results = {}
    if args.feature_path:
        if not args.gallery_pkl or not args.dataset_type:
            raise ValueError("--feature-path requires --gallery-pkl and --dataset-type")
        ds_config = {
            'gallery_pkl': args.gallery_pkl,
            'type': args.dataset_type,
            'feature_pkl': args.feature_path,
        }
        r = evaluate(args.dataset_name, ds_config, args.alpha, args.beta, args.text_only)
        if r:
            all_results[args.dataset_name] = r
    else:
        for ds_name in args.datasets:
            if ds_name not in DATASETS:
                continue
            r = evaluate(ds_name, DATASETS[ds_name], args.alpha, args.beta, args.text_only)
            if r:
                all_results[ds_name] = r

    print(f"\n\n{'='*60}")
    print(f"  SUMMARY  beta={args.beta}  alpha={args.alpha}")
    print(f"{'='*60}")
    for name, r in all_results.items():
        mb = r['baseline']
        me = r['ensemble_text_only']
        m3 = r.get('threeway_fusion')
        print(f"\n  {name}:")
        for k in mb:
            line = f"    {k}: base={mb[k]:.2f}  ens={me[k]:.2f}"
            if m3:
                d = m3[k] - mb[k]
                sign = '+' if d > 0 else ''
                line += f"  3way={m3[k]:.2f} ({sign}{d:.2f})"
            print(line)

    out_path = args.summary_out or os.path.join(ROOT, 'outputs', 'full_pipeline', 'eval_summary.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
