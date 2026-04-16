"""
GeneCIS Three-Way Fusion Evaluation with gallery feature precomputation.

Strategy: encode all unique gallery images ONCE, cache to .pkl,
then evaluation is pure lookup + matrix ops (seconds instead of hours).

Usage:
  python3 -u eval_genecis.py
  python3 -u eval_genecis.py --datasets genecis_change_object genecis_focus_object
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

ROOT = '/root/osrcir'
GENECIS_PATH = os.path.join(ROOT, 'datasets', 'GENECIS')
FEAT_CACHE_DIR = os.path.join(ROOT, 'precomputed_cache', 'genecis')
os.makedirs(FEAT_CACHE_DIR, exist_ok=True)

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


def load_model():
    import open_clip
    LOCAL_WEIGHTS = '/root/.cache/clip/ViT-L-14.pt'
    print("Loading ViT-L-14 from local cache...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained=None)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    jit_sd = torch.jit.load(LOCAL_WEIGHTS, map_location='cpu').state_dict()
    meta_keys = {'input_resolution', 'context_length', 'vocab_size'}
    filtered_sd = {k: v for k, v in jit_sd.items() if k not in meta_keys}
    model.load_state_dict(filtered_sd, strict=True)
    model.eval()
    del jit_sd, filtered_sd
    gc.collect()
    print(f"  Loaded in {time.time()-t0:.1f}s")
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
def encode_images_batched(model, preprocess, paths, batch_size=4,
                          label="images"):
    import numpy as np
    np_feats = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                imgs.append(preprocess(img))
                img.close()
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        t = torch.stack(imgs)
        f = model.encode_image(t).float().numpy()
        np_feats.append(f)
        del t, f, imgs
        done = min(i + batch_size, len(paths))
        if done % 500 == 0 or done == len(paths):
            print(f"    {label}: {done}/{len(paths)}", flush=True)
    if np_feats:
        return torch.from_numpy(np.concatenate(np_feats, axis=0))
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


def precompute_gallery_features(name, image_type, annotation, model,
                                preprocess):
    cache_path = os.path.join(FEAT_CACHE_DIR, f'{name}_gallery.pkl')
    all_ids = collect_unique_gallery_ids(annotation, image_type)
    print(f"  Unique gallery images: {len(all_ids)}")

    if os.path.exists(cache_path):
        cached = pickle.load(open(cache_path, 'rb'))
        if len(cached['ids']) == len(all_ids):
            print(f"  Loaded gallery cache ({cache_path})")
            return (cached['feats'],
                    {gid: i for i, gid in enumerate(cached['ids'])})
        print(f"  Cache mismatch ({len(cached['ids'])} vs {len(all_ids)})")

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
    elapsed = time.time() - t0
    rate = len(valid_ids) / max(elapsed, 1)
    print(f"  Gallery encoded: {len(valid_ids)} in {elapsed:.0f}s "
          f"({rate:.1f} img/s)")

    with open(cache_path, 'wb') as f:
        pickle.dump({'feats': gf, 'ids': valid_ids}, f)
    sz = os.path.getsize(cache_path) / 1024 / 1024
    print(f"  Saved gallery cache: {sz:.1f}MB")

    return gf, {gid: i for i, gid in enumerate(valid_ids)}


def evaluate_dataset(name, config, model, preprocess, tokenizer,
                     alpha, beta, text_only):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    baseline = json.load(open(config['baseline_json']))
    annotation = json.load(open(config['annotation']))
    total = min(len(baseline), len(annotation))
    print(f"  Samples: {total}")

    refine_map = {}
    if os.path.exists(config['refine_cache']):
        refine_list = json.load(open(config['refine_cache']))
        refine_map = {r['index']: r for r in refine_list}
        print(f"  Refinements: {len(refine_map)}")

    gallery_feats, id_to_idx = precompute_gallery_features(
        name, config['image_type'], annotation[:total], model, preprocess)
    gc.collect()

    print(f"  Encoding texts...", flush=True)
    t_txt = time.time()
    d1_texts, d2_texts = [], []
    for idx in range(total):
        d1 = baseline[idx].get('target_description', '') or ''
        d2 = d1
        if idx in refine_map:
            d2 = refine_map[idx].get('refined_description', d1)
        d1_texts.append(d1)
        d2_texts.append(d2)

    with torch.no_grad():
        d1_feats = []
        for i in range(0, total, 64):
            toks = tokenizer(d1_texts[i:i + 64])
            d1_feats.append(model.encode_text(toks).float())
        d1_feats = F.normalize(torch.cat(d1_feats), dim=-1)

        d2_feats = []
        for i in range(0, total, 64):
            toks = tokenizer(d2_texts[i:i + 64])
            d2_feats.append(model.encode_text(toks).float())
        d2_feats = F.normalize(torch.cat(d2_feats), dim=-1)
    print(f"  Texts encoded in {time.time() - t_txt:.0f}s")

    ens_feats = F.normalize(beta * d1_feats + (1 - beta) * d2_feats, dim=-1)

    proxy_feats = None
    if not text_only:
        print(f"  Encoding proxy images...", flush=True)
        t_p = time.time()
        proxy_paths = []
        for i in range(total):
            proxy_paths.append(
                os.path.join(config['proxy_dir'], f'proxy_{i:05d}.jpg'))
        with torch.no_grad():
            proxy_feats = encode_images_batched(
                model, preprocess, proxy_paths, label="proxy")
            proxy_feats = F.normalize(proxy_feats, dim=-1)
        print(f"  Proxy encoded in {time.time() - t_p:.0f}s")
    gc.collect()

    print(f"  Evaluating (matrix ops)...", flush=True)
    t_eval = time.time()
    hits_b = {1: 0, 2: 0, 3: 0}
    hits_e = {1: 0, 2: 0, 3: 0}
    hits_3 = {1: 0, 2: 0, 3: 0}
    valid = 0
    proxy_used = 0

    for idx in range(total):
        if not d1_texts[idx]:
            continue

        ann = annotation[idx]
        gi = ann.get('gallery', [])
        ti = ann.get('target', {})
        if config['image_type'] == 'coco':
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

        g_local = gallery_feats[indices]

        sim_b = (d1_feats[idx:idx + 1] @ g_local.T).squeeze(0)
        sim_e = (ens_feats[idx:idx + 1] @ g_local.T).squeeze(0)

        rb = torch.argsort(sim_b, descending=True)
        re = torch.argsort(sim_e, descending=True)
        pb = (rb == target_pos).nonzero(as_tuple=True)[0].item()
        pe = (re == target_pos).nonzero(as_tuple=True)[0].item()

        for k in [1, 2, 3]:
            if pb < k:
                hits_b[k] += 1
            if pe < k:
                hits_e[k] += 1

        if proxy_feats is not None:
            sim_p = (proxy_feats[idx:idx + 1] @ g_local.T).squeeze(0)
            sim_3 = alpha * sim_e + (1 - alpha) * sim_p
            r3 = torch.argsort(sim_3, descending=True)
            p3 = (r3 == target_pos).nonzero(as_tuple=True)[0].item()
            proxy_used += 1
            for k in [1, 2, 3]:
                if p3 < k:
                    hits_3[k] += 1

        valid += 1

    elapsed_eval = time.time() - t_eval
    print(f"  Eval done in {elapsed_eval:.1f}s  "
          f"(valid={valid}/{total}, proxy_used={proxy_used})")

    m_b = {f'R@{k}': hits_b[k] / valid * 100 for k in [1, 2, 3]}
    m_e = {f'R@{k}': hits_e[k] / valid * 100 for k in [1, 2, 3]}
    m_3 = None
    if not text_only:
        m_3 = {f'R@{k}': hits_3[k] / valid * 100 for k in [1, 2, 3]}

    header = f"  {'Metric':<8} {'Base':>8} {'Ens':>8}"
    if m_3:
        header += f" {'3Way':>8} {'Delta':>8}"
    print(header)
    print(f"  {'-' * 45}")
    for k in ['R@1', 'R@2', 'R@3']:
        line = f"  {k:<8} {m_b[k]:>8.2f} {m_e[k]:>8.2f}"
        if m_3:
            d = m_3[k] - m_b[k]
            sign = '+' if d > 0 else ''
            line += f" {m_3[k]:>8.2f} {sign}{d:>7.2f}"
        print(line)

    return {
        'dataset': name,
        'valid': valid,
        'total': total,
        'baseline': m_b,
        'ensemble': m_e,
        'threeway': m_3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets', nargs='+',
        default=['genecis_change_object', 'genecis_focus_object',
                 'genecis_change_attribute', 'genecis_focus_attribute'])
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--text-only', action='store_true')
    args = parser.parse_args()

    model, preprocess, tokenizer = load_model()

    all_results = {}
    for ds in args.datasets:
        if ds not in DATASET_CONFIGS:
            continue
        r = evaluate_dataset(
            ds, DATASET_CONFIGS[ds], model, preprocess, tokenizer,
            args.alpha, args.beta, args.text_only)
        all_results[ds] = r
        gc.collect()

    print(f"\n\n{'=' * 60}")
    print(f"  SUMMARY  beta={args.beta}  alpha={args.alpha}")
    print(f"{'=' * 60}")
    for name, r in all_results.items():
        mb, me, m3 = r['baseline'], r['ensemble'], r['threeway']
        print(f"\n  {name} ({r['valid']}/{r['total']}):")
        for k in ['R@1', 'R@2', 'R@3']:
            line = f"    {k}: base={mb[k]:.2f}  ens={me[k]:.2f}"
            if m3:
                d = m3[k] - mb[k]
                sign = '+' if d > 0 else ''
                line += f"  3way={m3[k]:.2f} ({sign}{d:.2f})"
            print(line)

    out_path = os.path.join(ROOT, 'outputs', 'full_pipeline',
                            'genecis_eval_summary.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
