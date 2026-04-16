"""
GeneCIS Full-Scale Three-Way Fusion Evaluation — Windows GPU v2.

Key improvements over v1:
  - Phase 1: Pre-read ALL images from SSHFS to CPU RAM (tolerates slow I/O)
  - Phase 2: GPU batch-encode from RAM (fast, no SSHFS during GPU ops)
  - Gallery feature caching to .pkl (encode once, reuse forever)
  - Falls back to server-precomputed gallery .pkl if available
  - Per-image retry logic for SSHFS flakiness
  - Detailed progress logging

Run via schtasks on Windows desktop session (required for SSHFS UNC access).
"""
import os
import sys
import json
import pickle
import time
import traceback
import torch
import torch.nn.functional as F
from PIL import Image

ALPHA = 0.9
BETA = 0.7

LOG_PATH = r'D:\osrcir_remote\genecis_v2_eval.log'
RESULT_PATH = r'D:\osrcir_remote\genecis_v2_results.json'

UNC_OSRCIR = r'\\sshfs.k\root@1.15.92.20\osrcir'
UNC_DATASETS = r'\\sshfs.k\root@1.15.92.20\data\disk\datasets'

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass

def resolve_root():
    for p in [UNC_OSRCIR, r'Z:\osrcir', r'Z:\\', r'D:\osrcir_remote']:
        try:
            if os.path.exists(os.path.join(p, 'outputs')):
                return p
        except Exception:
            continue
    return '.'

def resolve_genecis(root):
    for p in [os.path.join(UNC_DATASETS, 'GENECIS'),
              os.path.join(root, 'datasets', 'GENECIS'),
              r'Y:\GENECIS', r'D:\osrcir_remote\datasets\GENECIS']:
        try:
            if os.path.exists(os.path.join(p, 'genecis')):
                return p
        except Exception:
            continue
    return None


def load_clip():
    log("Loading CLIP ViT-L/14...")
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_path = r'C:\Users\12427\.cache\clip\ViT-L-14.pt'

    if os.path.exists(clip_path):
        model = torch.jit.load(clip_path, map_location=device).eval()
        import clip as clip_lib
        _, preprocess = clip_lib.load('ViT-L/14', device='cpu')
        tokenize_fn = lambda texts: clip_lib.tokenize(texts, truncate=True)
    else:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai')
        model = model.to(device).eval()
        tokenize_fn = open_clip.get_tokenizer('ViT-L-14')

    log(f"CLIP ready on {device} ({time.time()-t0:.1f}s)")
    return model, preprocess, tokenize_fn, device


def get_img_path(img_id, img_type, genecis_path):
    if img_type == 'coco':
        return os.path.join(genecis_path, 'coco2017', 'val2017',
                            f'{int(img_id):012d}.jpg')
    for sub in ['VG_All', 'VG_ALL', 'VG_100K', 'VG_100K_2']:
        p = os.path.join(genecis_path, 'Visual_Genome', sub, f'{img_id}.jpg')
        if os.path.exists(p):
            return p
    return os.path.join(genecis_path, 'Visual_Genome', 'VG_All',
                        f'{img_id}.jpg')


def read_image_safe(path, preprocess, max_retries=3):
    """Read and preprocess one image with retries for SSHFS flakiness."""
    for attempt in range(max_retries):
        try:
            img = Image.open(path).convert('RGB')
            tensor = preprocess(img)
            img.close()
            return tensor
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None
    return None


def preread_images(paths, preprocess, label="images"):
    """Phase 1: Read all images from disk to CPU RAM as preprocessed tensors."""
    tensors = []
    ok = 0
    fail = 0
    t0 = time.time()
    for i, p in enumerate(paths):
        t = read_image_safe(p, preprocess)
        if t is not None:
            tensors.append(t)
            ok += 1
        else:
            tensors.append(torch.zeros(3, 224, 224))
            fail += 1
        if (i + 1) % 200 == 0 or (i + 1) == len(paths):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.1)
            eta = (len(paths) - i - 1) / max(rate, 0.01)
            log(f"  {label} read: {i+1}/{len(paths)} "
                f"({rate:.1f}/s, ok={ok} fail={fail}, ETA={eta:.0f}s)")
    return tensors


@torch.no_grad()
def gpu_encode_tensors(model, tensors, device, batch_size=64, label="encode"):
    """Phase 2: GPU batch-encode preprocessed tensors from RAM."""
    feats = []
    t0 = time.time()
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        f = model.encode_image(batch).float().cpu()
        feats.append(f)
        del batch
        done = min(i + batch_size, len(tensors))
        if done % 2000 == 0 or done == len(tensors):
            log(f"  {label} GPU: {done}/{len(tensors)} "
                f"({time.time()-t0:.1f}s)")
    if feats:
        return F.normalize(torch.cat(feats, dim=0), dim=-1)
    return torch.empty(0, 768)


@torch.no_grad()
def gpu_encode_texts(model, tokenize_fn, texts, device, batch_size=128):
    """Encode texts on GPU."""
    feats = []
    for i in range(0, len(texts), batch_size):
        toks = tokenize_fn(texts[i:i + batch_size]).to(device)
        f = model.encode_text(toks).float().cpu()
        feats.append(f)
    return F.normalize(torch.cat(feats, dim=0), dim=-1)


def load_or_compute_gallery(name, cfg, annotation, total,
                            model, preprocess, device, genecis_path,
                            cache_dir):
    """Load gallery features from cache, or precompute them."""
    cache_path = os.path.join(cache_dir, f'{name}_gallery.pkl')

    img_type = cfg['img_type']
    all_ids = set()
    for ann in annotation[:total]:
        gi = ann.get('gallery', [])
        ti = ann.get('target', {})
        if img_type == 'coco':
            all_ids.update(g['val_image_id'] for g in gi)
            tid = ti.get('val_image_id')
        else:
            all_ids.update(g['image_id'] for g in gi)
            tid = ti.get('image_id')
        if tid is not None:
            all_ids.add(tid)
    all_ids = sorted(all_ids)
    log(f"  Unique gallery images: {len(all_ids)}")

    if os.path.exists(cache_path):
        try:
            cached = pickle.load(open(cache_path, 'rb'))
            if len(cached['ids']) >= len(all_ids) * 0.95:
                log(f"  Loaded gallery cache: {cache_path} "
                    f"({len(cached['ids'])} images)")
                id_to_idx = {gid: i for i, gid in enumerate(cached['ids'])}
                return cached['feats'], id_to_idx
            log(f"  Cache mismatch ({len(cached['ids'])} vs {len(all_ids)})")
        except Exception as e:
            log(f"  Cache load failed: {e}")

    paths = []
    valid_ids = []
    for gid in all_ids:
        p = get_img_path(gid, img_type, genecis_path)
        if os.path.exists(p):
            paths.append(p)
            valid_ids.append(gid)
    log(f"  Found {len(valid_ids)}/{len(all_ids)} on disk")

    log(f"  Phase 1: Pre-reading {len(paths)} gallery images to RAM...")
    tensors = preread_images(paths, preprocess, label="gallery")

    log(f"  Phase 2: GPU encoding gallery...")
    gallery_feats = gpu_encode_tensors(model, tensors, device,
                                       batch_size=64, label="gallery")
    del tensors
    torch.cuda.empty_cache()

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'feats': gallery_feats, 'ids': valid_ids}, f)
        sz = os.path.getsize(cache_path) / 1024 / 1024
        log(f"  Saved gallery cache: {sz:.1f}MB")
    except Exception as e:
        log(f"  WARNING: Could not save gallery cache: {e}")

    return gallery_feats, {gid: i for i, gid in enumerate(valid_ids)}


def evaluate_dataset(name, cfg, model, preprocess, tokenize_fn, device,
                     root, genecis_path):
    log(f"\n{'='*60}")
    log(f"  {name}")
    log(f"{'='*60}")
    t_start = time.time()

    bl_path = os.path.join(root, 'outputs', f'{name}_full.json')
    rc_path = os.path.join(root, 'outputs', 'full_pipeline',
                           f'{name}_v7_refine_cache.json')
    proxy_dir = os.path.join(root, 'proxy_cache', name)
    ann_path = os.path.join(genecis_path, 'genecis',
                            cfg['ann_file'])

    for tag, p in [('baseline', bl_path), ('annotation', ann_path)]:
        if not os.path.exists(p):
            log(f"  SKIP: {tag} not found: {p}")
            return None

    log("  Loading JSONs...")
    baseline = json.load(open(bl_path, encoding='utf-8'))
    annotation = json.load(open(ann_path, encoding='utf-8'))
    total = min(len(baseline), len(annotation))
    log(f"  Samples: {total}")

    refine_map = {}
    if os.path.exists(rc_path):
        rl = json.load(open(rc_path, encoding='utf-8'))
        refine_map = {r['index']: r for r in rl}
        log(f"  Refinements: {len(refine_map)}")
    else:
        log("  No refine cache, D2=D1")

    cache_dir = os.path.join(root, 'precomputed_cache', 'genecis')
    gallery_feats, id_to_idx = load_or_compute_gallery(
        name, cfg, annotation, total,
        model, preprocess, device, genecis_path, cache_dir)

    log("  Encoding texts...")
    d1_texts = []
    d2_texts = []
    for idx in range(total):
        d1 = baseline[idx].get('target_description', '') or ''
        d2 = d1
        if idx in refine_map:
            d2 = refine_map[idx].get('refined_description', d1)
        d1_texts.append(d1)
        d2_texts.append(d2)
    d1_feats = gpu_encode_texts(model, tokenize_fn, d1_texts, device)
    d2_feats = gpu_encode_texts(model, tokenize_fn, d2_texts, device)
    ens_feats = F.normalize(BETA * d1_feats + (1 - BETA) * d2_feats, dim=-1)
    log(f"  Text features: {d1_feats.shape}")

    log("  Encoding proxy images...")
    proxy_paths = [os.path.join(proxy_dir, f'proxy_{i:05d}.jpg')
                   for i in range(total)]
    proxy_tensors = preread_images(proxy_paths, preprocess, label="proxy")
    proxy_feats = gpu_encode_tensors(model, proxy_tensors, device,
                                     batch_size=64, label="proxy")
    del proxy_tensors
    torch.cuda.empty_cache()

    log("  Running matrix evaluation...")
    t_eval = time.time()
    hits_b = {1: 0, 2: 0, 3: 0}
    hits_e = {1: 0, 2: 0, 3: 0}
    hits_3 = {1: 0, 2: 0, 3: 0}
    valid = 0
    proxy_used = 0
    img_type = cfg['img_type']

    for idx in range(total):
        if not d1_texts[idx]:
            continue
        ann = annotation[idx]
        gi = ann.get('gallery', [])
        ti = ann.get('target', {})
        if img_type == 'coco':
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
        sim_b = (d1_feats[idx:idx+1] @ g_local.T).squeeze(0)
        sim_e = (ens_feats[idx:idx+1] @ g_local.T).squeeze(0)

        rb = torch.argsort(sim_b, descending=True)
        re = torch.argsort(sim_e, descending=True)
        pb = (rb == target_pos).nonzero(as_tuple=True)[0].item()
        pe = (re == target_pos).nonzero(as_tuple=True)[0].item()

        for k in [1, 2, 3]:
            if pb < k: hits_b[k] += 1
            if pe < k: hits_e[k] += 1

        sim_p = (proxy_feats[idx:idx+1] @ g_local.T).squeeze(0)
        sim_3 = ALPHA * sim_e + (1 - ALPHA) * sim_p
        r3 = torch.argsort(sim_3, descending=True)
        p3 = (r3 == target_pos).nonzero(as_tuple=True)[0].item()
        proxy_used += 1
        for k in [1, 2, 3]:
            if p3 < k: hits_3[k] += 1

        valid += 1

    eval_sec = time.time() - t_eval
    total_sec = time.time() - t_start
    log(f"  Eval done: valid={valid}/{total}, proxy={proxy_used}, "
        f"eval={eval_sec:.1f}s, total={total_sec:.0f}s")

    if valid == 0:
        log("  ERROR: 0 valid samples")
        return None

    m_b = {f'R@{k}': hits_b[k] / valid * 100 for k in [1, 2, 3]}
    m_e = {f'R@{k}': hits_e[k] / valid * 100 for k in [1, 2, 3]}
    m_3 = {f'R@{k}': hits_3[k] / valid * 100 for k in [1, 2, 3]}

    log(f"  {'Metric':<8} {'Base':>8} {'Ens':>8} {'3Way':>8} {'Delta':>8}")
    log(f"  {'-'*45}")
    for k in ['R@1', 'R@2', 'R@3']:
        d = m_3[k] - m_b[k]
        sign = '+' if d >= 0 else ''
        log(f"  {k:<8} {m_b[k]:>8.2f} {m_e[k]:>8.2f} "
            f"{m_3[k]:>8.2f} {sign}{d:>7.2f}")

    return {
        'dataset': name, 'valid': valid, 'total': total,
        'params': {'beta': BETA, 'alpha': ALPHA},
        'baseline': m_b, 'ensemble': m_e, 'threeway': m_3,
        'time_seconds': int(total_sec),
    }


DATASETS = {
    'genecis_change_object':    {'ann_file': 'change_object.json',    'img_type': 'coco'},
    'genecis_focus_object':     {'ann_file': 'focus_object.json',     'img_type': 'coco'},
    'genecis_change_attribute': {'ann_file': 'change_attribute.json', 'img_type': 'vg'},
    'genecis_focus_attribute':  {'ann_file': 'focus_attribute.json',  'img_type': 'vg'},
}


def main():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(f"GeneCIS v2 Eval — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    root = resolve_root()
    genecis_path = resolve_genecis(root)
    log(f"ROOT = {root}")
    log(f"GENECIS = {genecis_path}")

    if genecis_path is None:
        log("ERROR: Cannot find GeneCIS dataset. Exiting.")
        sys.exit(1)

    model, preprocess, tokenize_fn, device = load_clip()

    results = {}
    for name, cfg in DATASETS.items():
        try:
            r = evaluate_dataset(name, cfg, model, preprocess, tokenize_fn,
                                 device, root, genecis_path)
            if r:
                results[name] = r
                with open(RESULT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                log(f"  Partial results saved to {RESULT_PATH}")
        except Exception as e:
            log(f"  EXCEPTION in {name}: {e}")
            log(traceback.format_exc())
        torch.cuda.empty_cache()

    log(f"\n{'='*60}")
    log(f"  FINAL SUMMARY  beta={BETA}  alpha={ALPHA}")
    log(f"{'='*60}")
    for n, r in results.items():
        mb, me, m3 = r['baseline'], r['ensemble'], r['threeway']
        log(f"\n  {n} ({r['valid']}/{r['total']}, {r['time_seconds']}s):")
        for k in ['R@1', 'R@2', 'R@3']:
            d = m3[k] - mb[k]
            sign = '+' if d >= 0 else ''
            log(f"    {k}: base={mb[k]:.2f}  ens={me[k]:.2f}  "
                f"3way={m3[k]:.2f} ({sign}{d:.2f})")

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\nFinal results: {RESULT_PATH}")

    try:
        server_path = os.path.join(root, 'outputs', 'full_pipeline',
                                   'genecis_eval_summary.json')
        with open(server_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log(f"Also saved to: {server_path}")
    except Exception as e:
        log(f"WARNING: Could not write to server: {e}")

    log("\nDone!")


if __name__ == '__main__':
    main()
