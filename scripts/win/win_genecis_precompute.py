"""
Windows GPU: precompute ALL GeneCIS features (gallery + text + proxy).
Save .pkl files back to server via UNC path. Then server does matrix eval instantly.

Run on Windows via schtasks in desktop session.
"""
import os
import sys
import json
import pickle
import time
import torch
import torch.nn.functional as F
from PIL import Image

# --- Path resolution ---
UNC_OSRCIR = r'\\sshfs.k\root@1.15.92.20\osrcir'
UNC_DATASETS = r'\\sshfs.k\root@1.15.92.20\data\disk\datasets'

ROOT = None
for p in [UNC_OSRCIR, r'Z:\osrcir', r'Z:\\', r'D:\osrcir_remote']:
    try:
        if os.path.exists(os.path.join(p, 'outputs')):
            ROOT = p
            break
    except Exception:
        continue
if ROOT is None:
    ROOT = '.'
print(f"ROOT = {ROOT}")

GENECIS_PATH = None
for p in [os.path.join(UNC_DATASETS, 'GENECIS'),
          os.path.join(ROOT, 'datasets', 'GENECIS'),
          r'Y:\GENECIS', r'D:\osrcir_remote\datasets\GENECIS']:
    try:
        if os.path.exists(os.path.join(p, 'genecis')):
            GENECIS_PATH = p
            break
    except Exception:
        continue
print(f"GENECIS_PATH = {GENECIS_PATH}")

if GENECIS_PATH is None:
    print("ERROR: Cannot find GeneCIS dataset")
    sys.exit(1)

CACHE_DIR = os.path.join(ROOT, 'precomputed_cache', 'genecis')
os.makedirs(CACHE_DIR, exist_ok=True)

DATASETS = {
    'genecis_change_object': {
        'baseline': os.path.join(ROOT, 'outputs', 'genecis_change_object_full.json'),
        'refine': os.path.join(ROOT, 'outputs', 'full_pipeline',
                               'genecis_change_object_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'genecis_change_object'),
        'annotation': os.path.join(GENECIS_PATH, 'genecis', 'change_object.json'),
        'img_type': 'coco',
    },
    'genecis_focus_object': {
        'baseline': os.path.join(ROOT, 'outputs', 'genecis_focus_object_full.json'),
        'refine': os.path.join(ROOT, 'outputs', 'full_pipeline',
                               'genecis_focus_object_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'genecis_focus_object'),
        'annotation': os.path.join(GENECIS_PATH, 'genecis', 'focus_object.json'),
        'img_type': 'coco',
    },
    'genecis_change_attribute': {
        'baseline': os.path.join(ROOT, 'outputs', 'genecis_change_attribute_full.json'),
        'refine': os.path.join(ROOT, 'outputs', 'full_pipeline',
                               'genecis_change_attribute_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'genecis_change_attribute'),
        'annotation': os.path.join(GENECIS_PATH, 'genecis', 'change_attribute.json'),
        'img_type': 'vg',
    },
    'genecis_focus_attribute': {
        'baseline': os.path.join(ROOT, 'outputs', 'genecis_focus_attribute_full.json'),
        'refine': os.path.join(ROOT, 'outputs', 'full_pipeline',
                               'genecis_focus_attribute_v7_refine_cache.json'),
        'proxy_dir': os.path.join(ROOT, 'proxy_cache', 'genecis_focus_attribute'),
        'annotation': os.path.join(GENECIS_PATH, 'genecis', 'focus_attribute.json'),
        'img_type': 'vg',
    },
}

ALPHA = 0.9
BETA = 0.7


def load_clip():
    clip_path = r'C:\Users\12427\.cache\clip\ViT-L-14.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if os.path.exists(clip_path):
        print("Loading CLIP from local JIT weights...")
        model = torch.jit.load(clip_path, map_location=device).eval()
        import clip as clip_lib
        _, preprocess = clip_lib.load('ViT-L/14', device='cpu')
        tokenizer = lambda texts: clip_lib.tokenize(texts, truncate=True)
    else:
        print("Loading CLIP from open_clip...")
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai')
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

    print(f"CLIP loaded on {device}")
    return model, preprocess, tokenizer, device


def get_img_path(img_id, img_type):
    if img_type == 'coco':
        return os.path.join(GENECIS_PATH, 'coco2017', 'val2017',
                            f'{int(img_id):012d}.jpg')
    for sub in ['VG_All', 'VG_100K', 'VG_100K_2']:
        p = os.path.join(GENECIS_PATH, 'Visual_Genome', sub, f'{img_id}.jpg')
        if os.path.exists(p):
            return p
    return os.path.join(GENECIS_PATH, 'Visual_Genome', 'VG_All',
                        f'{img_id}.jpg')


@torch.no_grad()
def encode_images_gpu(model, preprocess, paths, device, batch_size=32,
                      label="img"):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        imgs = []
        for p in batch:
            try:
                imgs.append(preprocess(Image.open(p).convert('RGB')))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        tensor = torch.stack(imgs).to(device)
        f = model.encode_image(tensor).float().cpu()
        feats.append(f)
        done = min(i + batch_size, len(paths))
        if done % 500 == 0 or done == len(paths):
            print(f"  {label}: {done}/{len(paths)}", flush=True)
    if feats:
        return torch.cat(feats, dim=0)
    return torch.empty(0, 768)


def process_dataset(name, cfg, model, preprocess, tokenizer, device):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    cache_path = os.path.join(CACHE_DIR, f'{name}_all.pkl')
    if os.path.exists(cache_path):
        print(f"  Cache exists: {cache_path}, loading for eval...")
        cached = pickle.load(open(cache_path, 'rb'))
        return run_eval(name, cfg, cached)

    if not os.path.exists(cfg['baseline']):
        print(f"  SKIP: baseline not found: {cfg['baseline']}")
        return None

    baseline = json.load(open(cfg['baseline'], encoding='utf-8'))
    annotation = json.load(open(cfg['annotation'], encoding='utf-8'))
    total = min(len(baseline), len(annotation))
    print(f"  Samples: {total}")

    refine_map = {}
    if os.path.exists(cfg['refine']):
        rl = json.load(open(cfg['refine'], encoding='utf-8'))
        refine_map = {r['index']: r for r in rl}
        print(f"  Refinements: {len(refine_map)}")

    # 1) Collect unique gallery IDs
    all_ids = set()
    for ann in annotation[:total]:
        gi = ann.get('gallery', [])
        ti = ann.get('target', {})
        if cfg['img_type'] == 'coco':
            all_ids.update(g['val_image_id'] for g in gi)
            tid = ti.get('val_image_id')
        else:
            all_ids.update(g['image_id'] for g in gi)
            tid = ti.get('image_id')
        if tid is not None:
            all_ids.add(tid)
    all_ids = sorted(all_ids)
    print(f"  Unique gallery images: {len(all_ids)}")

    # 2) Encode gallery images on GPU
    t0 = time.time()
    paths, valid_ids = [], []
    for gid in all_ids:
        p = get_img_path(gid, cfg['img_type'])
        if os.path.exists(p):
            paths.append(p)
            valid_ids.append(gid)
    print(f"  Found {len(valid_ids)}/{len(all_ids)} on disk")

    gallery_feats = encode_images_gpu(model, preprocess, paths, device,
                                      label="gallery")
    gallery_feats = F.normalize(gallery_feats, dim=-1)
    id_to_idx = {gid: i for i, gid in enumerate(valid_ids)}
    print(f"  Gallery: {time.time() - t0:.1f}s")

    # 3) Encode texts
    t1 = time.time()
    d1_texts, d2_texts = [], []
    for idx in range(total):
        d1 = baseline[idx].get('target_description', '') or ''
        d2 = d1
        if idx in refine_map:
            d2 = refine_map[idx].get('refined_description', d1)
        d1_texts.append(d1)
        d2_texts.append(d2)

    d1_feats, d2_feats = [], []
    for i in range(0, total, 64):
        toks = tokenizer(d1_texts[i:i + 64]).to(device)
        d1_feats.append(model.encode_text(toks).float().cpu())
        toks2 = tokenizer(d2_texts[i:i + 64]).to(device)
        d2_feats.append(model.encode_text(toks2).float().cpu())
    d1_feats = F.normalize(torch.cat(d1_feats), dim=-1)
    d2_feats = F.normalize(torch.cat(d2_feats), dim=-1)
    print(f"  Texts: {time.time() - t1:.1f}s")

    # 4) Encode proxy images
    t2 = time.time()
    proxy_paths = [os.path.join(cfg['proxy_dir'], f'proxy_{i:05d}.jpg')
                   for i in range(total)]
    proxy_feats = encode_images_gpu(model, preprocess, proxy_paths, device,
                                    label="proxy")
    proxy_feats = F.normalize(proxy_feats, dim=-1)
    print(f"  Proxy: {time.time() - t2:.1f}s")

    # 5) Save all features
    cache_data = {
        'gallery_feats': gallery_feats,
        'gallery_ids': valid_ids,
        'id_to_idx': id_to_idx,
        'd1_feats': d1_feats,
        'd2_feats': d2_feats,
        'd1_texts': d1_texts,
        'proxy_feats': proxy_feats,
        'total': total,
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    sz = os.path.getsize(cache_path) / 1024 / 1024
    print(f"  Saved: {cache_path} ({sz:.1f}MB)")

    return run_eval(name, cfg, cache_data)


def run_eval(name, cfg, cache):
    annotation = json.load(open(cfg['annotation'], encoding='utf-8'))
    total = cache['total']

    gf = cache['gallery_feats']
    id_to_idx = cache['id_to_idx']
    d1_feats = cache['d1_feats']
    d2_feats = cache['d2_feats']
    d1_texts = cache['d1_texts']
    proxy_feats = cache['proxy_feats']

    ens_feats = F.normalize(BETA * d1_feats + (1 - BETA) * d2_feats, dim=-1)

    hits_b = {1: 0, 2: 0, 3: 0}
    hits_e = {1: 0, 2: 0, 3: 0}
    hits_3 = {1: 0, 2: 0, 3: 0}
    valid = 0

    for idx in range(total):
        if not d1_texts[idx]:
            continue

        ann = annotation[idx]
        gi = ann.get('gallery', [])
        ti = ann.get('target', {})
        if cfg['img_type'] == 'coco':
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

        g_local = gf[indices]
        sim_b = (d1_feats[idx:idx + 1] @ g_local.T).squeeze(0)
        sim_e = (ens_feats[idx:idx + 1] @ g_local.T).squeeze(0)
        sim_p = (proxy_feats[idx:idx + 1] @ g_local.T).squeeze(0)
        sim_3 = ALPHA * sim_e + (1 - ALPHA) * sim_p

        for sims, hits in [(sim_b, hits_b), (sim_e, hits_e), (sim_3, hits_3)]:
            rank = torch.argsort(sims, descending=True)
            pos = (rank == target_pos).nonzero(as_tuple=True)[0].item()
            for k in [1, 2, 3]:
                if pos < k:
                    hits[k] += 1
        valid += 1

    m_b = {f'R@{k}': hits_b[k] / valid * 100 for k in [1, 2, 3]}
    m_e = {f'R@{k}': hits_e[k] / valid * 100 for k in [1, 2, 3]}
    m_3 = {f'R@{k}': hits_3[k] / valid * 100 for k in [1, 2, 3]}

    print(f"\n  {name} (valid={valid}/{total})")
    print(f"  {'Metric':<8} {'Base':>8} {'Ens':>8} {'3Way':>8} {'Delta':>8}")
    print(f"  {'-' * 45}")
    for k in ['R@1', 'R@2', 'R@3']:
        d = m_3[k] - m_b[k]
        sign = '+' if d > 0 else ''
        print(f"  {k:<8} {m_b[k]:>8.2f} {m_e[k]:>8.2f} "
              f"{m_3[k]:>8.2f} {sign}{d:>7.2f}")

    return {'dataset': name, 'valid': valid, 'total': total,
            'baseline': m_b, 'ensemble': m_e, 'threeway': m_3}


def main():
    model, preprocess, tokenizer, device = load_clip()

    results = {}
    for name, cfg in DATASETS.items():
        r = process_dataset(name, cfg, model, preprocess, tokenizer, device)
        if r:
            results[name] = r

    print(f"\n\n{'=' * 60}")
    print(f"  FINAL SUMMARY  beta={BETA}  alpha={ALPHA}")
    print(f"{'=' * 60}")
    for n, r in results.items():
        mb, me, m3 = r['baseline'], r['ensemble'], r['threeway']
        print(f"\n  {n} ({r['valid']}/{r['total']}):")
        for k in ['R@1', 'R@2', 'R@3']:
            d = m3[k] - mb[k]
            sign = '+' if d > 0 else ''
            print(f"    {k}: base={mb[k]:.2f}  ens={me[k]:.2f}  "
                  f"3way={m3[k]:.2f} ({sign}{d:.2f})")

    out = os.path.join(ROOT, 'outputs', 'full_pipeline',
                       'genecis_eval_summary.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == '__main__':
    main()
