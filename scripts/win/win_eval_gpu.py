"""
Windows GPU evaluation for Three-Way Fusion (full-scale).
Reads ALL data directly from Z: drive (Samba mount to Linux server).
Must run in Windows desktop session (Z: not visible in SSH).

Usage: schtasks triggers this from desktop session.

Three-way fusion:
  text_feat = normalize(BETA * CLIP(D1) + (1-BETA) * CLIP(D2))
  score = ALPHA * sim(text_feat, gallery) + (1-ALPHA) * sim(proxy_feat, gallery)
"""
import json
import os
import sys
import pickle
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F

ZROOT = r'Z:\\'
LOCAL = r'D:\osrcir_remote'

if os.path.exists(os.path.join(ZROOT, 'proxy_cache')):
    ROOT = ZROOT
elif os.path.exists(os.path.join(LOCAL, 'proxy_cache')):
    ROOT = LOCAL
elif os.path.exists('/root/osrcir/proxy_cache'):
    ROOT = '/root/osrcir'
else:
    print("ERROR: Cannot find data root"); sys.exit(1)

PRECOMPUTED = os.path.join(ROOT, 'precomputed_cache', 'precomputed')
if not os.path.isdir(PRECOMPUTED):
    PRECOMPUTED = ROOT

BETA = 0.7
ALPHA = 0.9

LOG_FILE = os.path.join(LOCAL, 'gpu_eval_log.txt')

def _feat_pkl(name):
    return os.path.join(PRECOMPUTED, f'{name}_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl')

DATASETS = {
    'fashioniq_dress': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'fashioniq_dress_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'fashioniq_dress_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'fashioniq_dress'),
        'img_features': _feat_pkl('fashioniq_dress'),
        'type': 'fashioniq',
    },
    'fashioniq_shirt': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'fashioniq_shirt_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'fashioniq_shirt_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'fashioniq_shirt'),
        'img_features': _feat_pkl('fashioniq_shirt'),
        'type': 'fashioniq',
    },
    'fashioniq_toptee': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'fashioniq_toptee_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'fashioniq_toptee_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'fashioniq_toptee'),
        'img_features': _feat_pkl('fashioniq_toptee'),
        'type': 'fashioniq',
    },
    'circo': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'circo_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'circo_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'circo'),
        'img_features': _feat_pkl('circo'),
        'type': 'circo',
    },
    'cirr': {
        'baseline_json': os.path.join(ROOT, 'outputs', 'cirr', 'cirr_full.json'),
        'refine_cache': os.path.join(ROOT, 'outputs', 'full_pipeline', 'cirr_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'cirr'),
        'img_features': _feat_pkl('cirr'),
        'type': 'cirr',
    },
}


def load_clip():
    import clip
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading CLIP ViT-L/14 on {device}...")
    model, preprocess = clip.load('ViT-L/14', device=device, jit=False)
    model.eval()
    tokenizer = lambda texts: clip.tokenize(texts, context_length=77, truncate=True)
    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=256):
    all_feats = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i+batch_size]).to(device)
        feats = model.encode_text(tokens).float().cpu()
        all_feats.append(feats)
    return torch.vstack(all_feats)


@torch.no_grad()
def encode_images(model, preprocess, image_paths, device, batch_size=64):
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
        tensor = torch.stack(imgs).to(device)
        feats = model.encode_image(tensor).float().cpu()
        all_feats.append(feats)
        if (i // batch_size) % 10 == 0:
            print(f"    img batch {i//batch_size+1}/{(len(image_paths)-1)//batch_size+1}", flush=True)
    return torch.vstack(all_feats)


def compute_fiq_metrics(sorted_names, target_names, ks=(1, 5, 10, 50)):
    targets_arr = np.array(target_names)
    results = {}
    for k in ks:
        topk = sorted_names[:, :k]
        hits = np.any(topk == targets_arr[:, None], axis=1)
        results[f'R@{k}'] = float(np.mean(hits)) * 100
    return results


def compute_cirr_metrics(sorted_names, target_names, group_members_list, gallery_names, ks=(1, 5, 10, 50)):
    """R@K (full gallery) and R_subset@K (within group only)."""
    targets_arr = np.array(target_names)
    results = {}
    for k in ks:
        topk = sorted_names[:, :k]
        hits = np.any(topk == targets_arr[:, None], axis=1)
        results[f'R@{k}'] = float(np.mean(hits)) * 100

    gallery_list = list(gallery_names)
    for k in [1, 2, 3]:
        subset_hits = 0
        count = 0
        for i, members in enumerate(group_members_list):
            member_set = set(str(m) for m in members)
            target = str(target_names[i])
            if target not in member_set or len(member_set) < 2:
                continue
            count += 1
            member_indices = [gallery_list.index(m) for m in member_set if m in gallery_list]
            if not member_indices:
                continue
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


def evaluate_dataset(name, config, model, preprocess, tokenizer, device):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    for key in ['baseline_json', 'refine_cache', 'img_features']:
        if not os.path.exists(config[key]):
            print(f"  [SKIP] Missing: {config[key]}")
            return None

    baseline = json.load(open(config['baseline_json'], encoding='utf-8'))
    print(f"  Baseline: {len(baseline)} samples")

    refine_list = json.load(open(config['refine_cache'], encoding='utf-8'))
    refine_map = {r['index']: r for r in refine_list}
    print(f"  V7 refine: {len(refine_map)} entries")

    gallery_data = pickle.load(open(config['img_features'], 'rb'))
    gallery_features = F.normalize(gallery_data['index_features'].float(), dim=-1)
    gallery_names = np.array(gallery_data['index_names'])
    print(f"  Gallery: {len(gallery_names)} images")

    d1_texts, d2_texts, proxy_paths = [], [], []
    target_names, gt_targets, group_members = [], [], []

    for idx, sample in enumerate(baseline):
        d1 = sample.get('target_description', '')
        if not d1:
            continue
        target = sample.get('target_name', '')
        if config['type'] == 'fashioniq' and not target:
            continue
        if config['type'] == 'cirr' and not target:
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
        if config['type'] == 'cirr':
            group_members.append(sample.get('ground_truth_candidates', []))

    n = len(d1_texts)
    print(f"  Valid queries: {n}")

    t0 = time.time()
    print("  Encoding D1 texts...")
    d1_feats = F.normalize(encode_texts(model, tokenizer, d1_texts, device).float(), dim=-1)
    print("  Encoding D2 texts...")
    d2_feats = F.normalize(encode_texts(model, tokenizer, d2_texts, device).float(), dim=-1)
    print("  Encoding proxy images...")
    proxy_feats = F.normalize(encode_images(model, preprocess, proxy_paths, device).float(), dim=-1)
    print(f"  Encoding done in {time.time()-t0:.1f}s")

    ens_feats = F.normalize(BETA * d1_feats + (1 - BETA) * d2_feats, dim=-1)

    sim_base = d1_feats @ gallery_features.T
    idx_base = torch.argsort(sim_base, dim=-1, descending=True).numpy()
    names_base = gallery_names[idx_base]
    del sim_base; gc.collect()

    sim_ens = ens_feats @ gallery_features.T
    sim_proxy = proxy_feats @ gallery_features.T
    sim_3way = ALPHA * sim_ens + (1 - ALPHA) * sim_proxy
    del sim_proxy; gc.collect()

    idx_ens = torch.argsort(sim_ens, dim=-1, descending=True).numpy()
    names_ens = gallery_names[idx_ens]
    del sim_ens; gc.collect()

    idx_3way = torch.argsort(sim_3way, dim=-1, descending=True).numpy()
    names_3way = gallery_names[idx_3way]
    del sim_3way; gc.collect()

    if config['type'] == 'fashioniq':
        m_base = compute_fiq_metrics(names_base, target_names)
        m_ens = compute_fiq_metrics(names_ens, target_names)
        m_3way = compute_fiq_metrics(names_3way, target_names)
        keys = ['R@1', 'R@5', 'R@10', 'R@50']
    elif config['type'] == 'cirr':
        m_base = compute_cirr_metrics(names_base, target_names, group_members, gallery_names)
        m_ens = compute_cirr_metrics(names_ens, target_names, group_members, gallery_names)
        m_3way = compute_cirr_metrics(names_3way, target_names, group_members, gallery_names)
        keys = ['R@1', 'R@5', 'R@10', 'R@50', 'R_sub@1', 'R_sub@2', 'R_sub@3']
    else:
        m_base = compute_circo_metrics(names_base, gt_targets)
        m_ens = compute_circo_metrics(names_ens, gt_targets)
        m_3way = compute_circo_metrics(names_3way, gt_targets)
        keys = ['mAP@5', 'mAP@10', 'mAP@25', 'mAP@50']

    del names_base, names_ens, names_3way, d1_feats, d2_feats, proxy_feats, ens_feats, gallery_features
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n  {'Metric':<12} {'Baseline':>10} {'Ensemble':>10} {'3-Way':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for k in keys:
        b, e, t = m_base[k], m_ens[k], m_3way[k]
        d = t - b
        sign = '+' if d > 0 else ''
        print(f"  {k:<12} {b:>10.2f} {e:>10.2f} {t:>10.2f} {sign}{d:>9.2f}")

    result = {
        'dataset': name, 'num_queries': n, 'num_gallery': len(gallery_names),
        'params': {'beta': BETA, 'alpha': ALPHA},
        'baseline': m_base, 'ensemble_text_only': m_ens, 'threeway_fusion': m_3way,
    }
    out_path = os.path.join(ROOT, 'outputs', 'full_pipeline', f'{name}_eval_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result


class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log = open(log_path, 'w', encoding='utf-8')
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    sys.stdout = Tee(LOG_FILE)

    ds_names = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    print(f"{'='*60}")
    print(f"  Three-Way Fusion Full-Scale Evaluation (GPU)")
    print(f"  BETA={BETA}, ALPHA={ALPHA}, Root={ROOT}")
    print(f"  Log: {LOG_FILE}")
    print(f"{'='*60}")

    model, preprocess, tokenizer, device = load_clip()

    all_results = {}
    for name in ds_names:
        if name not in DATASETS:
            print(f"  [SKIP] Unknown: {name}")
            continue
        r = evaluate_dataset(name, DATASETS[name], model, preprocess, tokenizer, device)
        if r:
            all_results[name] = r
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n\n{'='*60}")
    print(f"  FINAL SUMMARY  beta={BETA}  alpha={ALPHA}")
    print(f"{'='*60}")
    for name, r in all_results.items():
        mb, me, m3 = r['baseline'], r['ensemble_text_only'], r['threeway_fusion']
        print(f"\n  {name} ({r['num_queries']} queries, {r['num_gallery']} gallery):")
        for k in m3:
            d = m3[k] - mb[k]
            sign = '+' if d > 0 else ''
            print(f"    {k}: {mb[k]:.2f} -> ens={me[k]:.2f} -> 3way={m3[k]:.2f} ({sign}{d:.2f})")

    summary_path = os.path.join(ROOT, 'outputs', 'full_pipeline', 'eval_summary_gpu.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    local_summary = os.path.join(LOCAL, 'eval_summary_gpu.json')
    with open(local_summary, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Local copy: {local_summary}")


if __name__ == '__main__':
    main()
