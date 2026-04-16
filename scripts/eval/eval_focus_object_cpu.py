"""
CPU evaluation of focus_object 200 samples: old V7 vs new V7-Focus D2.
Uses JIT state_dict loaded into open_clip. No GPU needed.
"""
import json
import os
import random
import time
import gc
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

SEED = 42
N_SAMPLES = 200
ALPHA = 0.9
BETA = 0.7

BASELINE_JSON = 'outputs/genecis_focus_object_full.json'
OLD_REFINE = 'outputs/full_pipeline/genecis_focus_object_v7_refine_cache.json'
NEW_REFINE = 'outputs/full_pipeline/genecis_focus_object_v7focus_refine_cache.json'
ANNOTATION = 'datasets/GENECIS/genecis/focus_object.json'
GENECIS_PATH = 'datasets/GENECIS'
PROXY_DIR = 'proxy_cache/genecis_focus_object'


def load_clip():
    print("Loading CLIP ViT-L/14 on CPU...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    jit_model = torch.jit.load('/root/.cache/clip/ViT-L-14.pt', map_location='cpu')
    jit_sd = jit_model.state_dict()
    filtered = {k: v for k, v in jit_sd.items() if k in model.state_dict()}
    model.load_state_dict(filtered, strict=True)
    model.eval()
    del jit_model, jit_sd; gc.collect()

    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, texts, batch_size=32):
    feats = []
    for i in range(0, len(texts), batch_size):
        toks = tokenizer(texts[i:i+batch_size])
        f = model.encode_text(toks).float()
        feats.append(f)
    return F.normalize(torch.cat(feats), dim=-1)


@torch.no_grad()
def encode_images(model, preprocess, paths, batch_size=8, label="img"):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = []
        for p in batch:
            try:
                imgs.append(preprocess(Image.open(p).convert('RGB')))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        tensor = torch.stack(imgs)
        f = model.encode_image(tensor).float()
        feats.append(f)
        done = min(i + batch_size, len(paths))
        if done % 200 == 0 or done == len(paths):
            print(f"    {label}: {done}/{len(paths)}", flush=True)
    return F.normalize(torch.cat(feats), dim=-1) if feats else torch.empty(0, 768)


def get_img_path(img_id):
    return os.path.join(GENECIS_PATH, 'coco2017', 'val2017', f'{int(img_id):012d}.jpg')


def evaluate_with_d2(d1_feats, d2_feats, proxy_feats, gallery_feats, id_to_idx,
                     annotation, total, d1_texts, alpha, beta):
    ens = F.normalize(beta * d1_feats + (1 - beta) * d2_feats, dim=-1)

    results = {}
    for method_name, query_feats, use_proxy in [
        ('baseline', d1_feats, False),
        ('ensemble', ens, False),
        ('threeway', ens, True),
    ]:
        hits = {1: 0, 2: 0, 3: 0}
        valid = 0
        for idx in range(total):
            if not d1_texts[idx]:
                continue
            ann = annotation[idx]
            gi = ann.get('gallery', [])
            ti = ann.get('target', {})
            tid = ti.get('val_image_id')
            gids = [g['val_image_id'] for g in gi]
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
            sim = (query_feats[idx:idx+1] @ g_local.T).squeeze(0)
            if use_proxy:
                sim_p = (proxy_feats[idx:idx+1] @ g_local.T).squeeze(0)
                sim = alpha * sim + (1 - alpha) * sim_p

            rank = torch.argsort(sim, descending=True)
            pos = (rank == target_pos).nonzero(as_tuple=True)[0].item()
            for k in [1, 2, 3]:
                if pos < k:
                    hits[k] += 1
            valid += 1

        results[method_name] = {f'R@{k}': hits[k] / valid * 100 for k in [1, 2, 3]}
        results[method_name]['valid'] = valid

    return results


def main():
    baseline = json.load(open(BASELINE_JSON, encoding='utf-8'))
    annotation = json.load(open(ANNOTATION, encoding='utf-8'))
    old_refine = json.load(open(OLD_REFINE, encoding='utf-8'))
    new_refine = json.load(open(NEW_REFINE, encoding='utf-8'))
    old_map = {r['index']: r for r in old_refine}
    new_map = {r['index']: r for r in new_refine}

    random.seed(SEED)
    indices = sorted(random.sample(range(len(baseline)), min(N_SAMPLES, len(baseline))))
    print(f"Selected {len(indices)} samples (seed={SEED})")

    model, preprocess, tokenizer = load_clip()

    total = len(indices)
    d1_texts, old_d2_texts, new_d2_texts = [], [], []
    for idx in indices:
        d1 = baseline[idx].get('target_description', '') or ''
        old_d2 = old_map.get(idx, {}).get('refined_description', d1) or d1
        new_d2 = new_map.get(idx, {}).get('refined_description', d1) or d1
        d1_texts.append(d1)
        old_d2_texts.append(old_d2)
        new_d2_texts.append(new_d2)

    print(f"\nEncoding texts ({total} samples)...")
    t0 = time.time()
    d1_feats = encode_texts(model, tokenizer, d1_texts)
    old_d2_feats = encode_texts(model, tokenizer, old_d2_texts)
    new_d2_feats = encode_texts(model, tokenizer, new_d2_texts)
    print(f"  Text encoding: {time.time()-t0:.1f}s")

    print(f"\nCollecting gallery images...")
    all_gal_ids = set()
    sub_annotations = [annotation[i] for i in indices]
    for ann in sub_annotations:
        gi = ann.get('gallery', [])
        ti = ann.get('target', {})
        all_gal_ids.update(g['val_image_id'] for g in gi)
        tid = ti.get('val_image_id')
        if tid is not None:
            all_gal_ids.add(tid)
    all_gal_ids = sorted(all_gal_ids)
    print(f"  Unique gallery images: {len(all_gal_ids)}")

    gal_paths = []
    valid_ids = []
    for gid in all_gal_ids:
        p = get_img_path(gid)
        if os.path.exists(p):
            gal_paths.append(p)
            valid_ids.append(gid)
    print(f"  Found on disk: {len(valid_ids)}/{len(all_gal_ids)}")

    print(f"\nEncoding gallery images on CPU (this takes a while)...")
    t1 = time.time()
    gallery_feats = encode_images(model, preprocess, gal_paths, batch_size=4, label="gallery")
    id_to_idx = {gid: i for i, gid in enumerate(valid_ids)}
    print(f"  Gallery encoding: {time.time()-t1:.1f}s")

    print(f"\nEncoding proxy images...")
    t2 = time.time()
    proxy_paths = [os.path.join(PROXY_DIR, f'proxy_{i:05d}.jpg') for i in indices]
    proxy_feats = encode_images(model, preprocess, proxy_paths, batch_size=4, label="proxy")
    print(f"  Proxy encoding: {time.time()-t2:.1f}s")

    del model; gc.collect()

    # Evaluate with different alpha/beta combos for both old and new D2
    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS (200 samples, seed={SEED})")
    print(f"{'='*70}")

    for ab_label, alphas, betas in [
        ("Fixed α=0.9 β=0.7", [0.9], [0.7]),
        ("Grid search", [0.80, 0.85, 0.90, 0.95, 1.00], [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]),
    ]:
        print(f"\n--- {ab_label} ---")
        for alpha in alphas:
            for beta in betas:
                old_r = evaluate_with_d2(d1_feats, old_d2_feats, proxy_feats, gallery_feats,
                                         id_to_idx, sub_annotations, total, d1_texts, alpha, beta)
                new_r = evaluate_with_d2(d1_feats, new_d2_feats, proxy_feats, gallery_feats,
                                         id_to_idx, sub_annotations, total, d1_texts, alpha, beta)

                if len(alphas) == 1 and len(betas) == 1:
                    print(f"\n  α={alpha} β={beta}:")
                    print(f"  {'Method':<12} {'R@1':>8} {'R@2':>8} {'R@3':>8}")
                    print(f"  {'-'*40}")
                    for m in ['baseline', 'ensemble', 'threeway']:
                        o = old_r[m]
                        print(f"  old {m:<7} {o['R@1']:>8.2f} {o['R@2']:>8.2f} {o['R@3']:>8.2f}")
                    for m in ['ensemble', 'threeway']:
                        n = new_r[m]
                        d1r = old_r['baseline']['R@1']
                        delta = n['R@1'] - d1r
                        sign = '+' if delta >= 0 else ''
                        print(f"  NEW {m:<7} {n['R@1']:>8.2f} {n['R@2']:>8.2f} {n['R@3']:>8.2f}  (R@1 Δ={sign}{delta:.2f})")
                else:
                    ob = old_r['baseline']['R@1']
                    o3 = old_r['threeway']['R@1']
                    n3 = new_r['threeway']['R@1']
                    d_old = o3 - ob
                    d_new = n3 - ob
                    marker = '***' if n3 > ob else '  ' if n3 == ob else ''
                    if n3 > ob:
                        print(f"  α={alpha:.2f} β={beta:.2f}: base={ob:.1f}  old_3way={o3:.1f}({d_old:+.1f})  NEW_3way={n3:.1f}({d_new:+.1f}) {marker}")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
