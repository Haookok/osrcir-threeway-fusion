"""
Complete ensemble evaluation for ALL datasets.
Supports two modes:
  - LOCAL mode: reads from D:\osrcir_remote (pre-transferred data)
  - SSHFS mode: reads from Z:\ (server mounted via SSHFS)

For GeneCIS: requires dataset images (COCO/VG), only works in SSHFS mode.
For FashionIQ/CIRR/CIRCO: works in both modes.

Usage:
  python win_eval_all.py                    # auto-detect
  python win_eval_all.py --base Z:\         # force SSHFS mode
  python win_eval_all.py --base D:\osrcir_remote  # force local mode
"""
import json, os, pickle, random, gc, sys, argparse
import numpy as np, torch, clip, PIL.Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BETA = 0.7
ALPHA = 0.9


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=str, default=None,
                   help='Base directory. Auto-detects Z:\\ or D:\\osrcir_remote')
    p.add_argument('--beta', type=float, default=0.7)
    p.add_argument('--alpha', type=float, default=0.9)
    p.add_argument('--datasets', nargs='+', default=['all'],
                   help='Which datasets to run: dress shirt toptee cirr circo genecis_all or all')
    return p.parse_args()


def detect_base():
    for b in [r'Z:\\', r'Z:', r'D:\osrcir_remote']:
        if os.path.exists(b):
            return b
    raise RuntimeError("No base directory found. Mount SSHFS to Z: or use D:\\osrcir_remote")


@torch.no_grad()
def encode_texts(model, texts):
    tok = lambda t: clip.tokenize(t, context_length=77, truncate=True)
    feats = []
    for i in range(0, len(texts), 64):
        feats.append(model.encode_text(tok(texts[i:i+64]).to(DEVICE)).float().cpu())
    return torch.nn.functional.normalize(torch.vstack(feats), dim=-1)


@torch.no_grad()
def encode_images(model, preprocess, paths):
    feats = []
    for p in paths:
        if p and os.path.exists(p):
            img = preprocess(PIL.Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
            feats.append(model.encode_image(img).float().cpu().squeeze(0))
        else:
            feats.append(torch.zeros(768))
    return torch.nn.functional.normalize(torch.stack(feats), dim=-1)


def recall_at_k(text_f, proxy_f, idx_n, idx_names, target_names, alpha, ks=[1,5,10,50]):
    sim = text_f @ idx_n.T
    if alpha < 1.0:
        sim = alpha * sim + (1 - alpha) * (proxy_f @ idx_n.T)
    si = torch.argsort(1 - sim, dim=-1).cpu()
    sn = np.array(idx_names)[si]
    lb = torch.tensor(sn == np.array(target_names).reshape(-1, 1))
    return {f'R@{k}': round((lb[:, :k].sum() / len(lb)).item() * 100, 2) for k in ks}


def map_at_k(text_f, proxy_f, idx_n, idx_names, target_names, gt_targets_list, alpha, ks=[5,10,25,50]):
    sim = text_f @ idx_n.T
    if alpha < 1.0:
        sim = alpha * sim + (1 - alpha) * (proxy_f @ idx_n.T)
    si = torch.argsort(1 - sim, dim=-1).cpu()
    sn = np.array(idx_names)[si]
    maps = {k: [] for k in ks}
    recalls = {k: [] for k in ks}
    for i in range(len(target_names)):
        gts = [str(g) for g in gt_targets_list[i] if g != '' and g is not None]
        sn_top = sn[i][:max(ks)]
        map_labels = torch.tensor(np.isin(sn_top, gts), dtype=torch.uint8)
        prec = torch.cumsum(map_labels, dim=0) * map_labels
        prec = prec / torch.arange(1, len(sn_top) + 1)
        for k in ks:
            maps[k].append(float(torch.sum(prec[:k]) / min(len(gts), k)))
        single = torch.tensor(sn_top == str(target_names[i]))
        for k in ks:
            recalls[k].append(float(torch.sum(single[:k])))
    out = {f'mAP@{k}': round(np.mean(v) * 100, 2) for k, v in maps.items()}
    out.update({f'R@{k}': round(np.mean(v) * 100, 2) for k, v in recalls.items()})
    return out


def resolve_path(base, *parts):
    """Try multiple path patterns to find the file."""
    candidates = [
        os.path.join(base, *parts),
        os.path.join(base, 'results', parts[-1]) if len(parts) == 1 else None,
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return os.path.join(base, *parts)


def run_fiq_cirr(name, label, base, model, preprocess, beta, alpha):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    # Find files
    rjson_candidates = [
        os.path.join(base, 'results', f'{name}_full.json'),
        os.path.join(base, 'outputs', f'{name}_full.json'),
        os.path.join(base, 'outputs', 'cirr', 'cirr_full.json') if name == 'cirr' else None,
    ]
    rjson = next((p for p in rjson_candidates if p and os.path.exists(p)), None)
    if not rjson:
        print(f"  SKIP: baseline results not found"); return None

    feat_candidates = [
        os.path.join(base, 'features', f'{name}_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl'),
        os.path.join(base, 'precomputed_cache', 'precomputed', f'{name}_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl'),
    ]
    feat_path = next((p for p in feat_candidates if os.path.exists(p)), None)
    if not feat_path:
        print(f"  SKIP: CLIP features not found"); return None

    v7_candidates = [
        os.path.join(base, 'v7_caches', f'{name}_v7_anti_hallucination_refine_200s_seed42.json'),
        os.path.join(base, 'outputs', 'prompt_ab_test', f'{name}_v7_anti_hallucination_refine_200s_seed42.json'),
    ]
    v7_path = next((p for p in v7_candidates if os.path.exists(p)), None)
    if not v7_path:
        print(f"  SKIP: v7 refine cache not found"); return None

    with open(rjson, encoding='utf-8') as f:
        all_s = json.load(f)
    random.seed(42)
    indices = sorted(random.sample(range(len(all_s)), min(200, len(all_s))))
    orig_descs = [all_s[i]['target_description'] for i in indices]
    target_names = [all_s[i].get('target_name', '') for i in indices]
    del all_s; gc.collect()

    with open(v7_path, encoding='utf-8') as f:
        v7_descs = json.load(f)['refined_descs']

    data = pickle.load(open(feat_path, 'rb'))
    idx_feat = torch.nn.functional.normalize(data['index_features'].float(), dim=-1)
    idx_names = data['index_names']
    del data; gc.collect()

    orig_f = encode_texts(model, orig_descs)
    v7_f = encode_texts(model, v7_descs)

    proxy_candidates = [
        os.path.join(base, 'proxy_cache', name),
        os.path.join(base, 'proxy_cache', name),
    ]
    proxy_dir = next((p for p in proxy_candidates if os.path.exists(p)), proxy_candidates[0])
    proxy_paths = [os.path.join(proxy_dir, f'proxy_{i:05d}.jpg') for i in indices]
    proxy_f = encode_images(model, preprocess, proxy_paths)

    bl = recall_at_k(orig_f, proxy_f, idx_feat, idx_names, target_names, 1.0)
    ens_f = torch.nn.functional.normalize(beta * orig_f + (1 - beta) * v7_f, dim=-1)
    er = recall_at_k(ens_f, proxy_f, idx_feat, idx_names, target_names, alpha)

    for m in ['R@1', 'R@5', 'R@10', 'R@50']:
        d = er[m] - bl[m]
        mark = '+' if d > 0 else ('=' if d == 0 else '-')
        print(f"  {m}: BL={bl[m]:>6}  Ens={er[m]:>6}  {d:>+6.1f} {mark}")

    del idx_feat, orig_f, v7_f, proxy_f; gc.collect()
    return {'baseline': bl, 'ensemble': er}


def run_circo(base, model, preprocess, beta, alpha):
    print(f"\n{'='*60}\n  CIRCO\n{'='*60}")

    rjson_candidates = [
        os.path.join(base, 'results', 'circo_full.json'),
        os.path.join(base, 'outputs', 'circo', 'circo_full.json'),
        os.path.join(base, 'outputs', 'circo_full.json'),
    ]
    rjson = next((p for p in rjson_candidates if p and os.path.exists(p)), None)
    if not rjson:
        print(f"  SKIP: CIRCO results not found"); return None

    feat_candidates = [
        os.path.join(base, 'features', 'circo_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl'),
        os.path.join(base, 'precomputed_cache', 'precomputed', 'circo_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl'),
    ]
    feat_path = next((p for p in feat_candidates if os.path.exists(p)), None)
    if not feat_path:
        print(f"  SKIP: CIRCO features not found (363MB, may need SSHFS)"); return None

    v7_candidates = [
        os.path.join(base, 'v7_caches', 'circo_v7_anti_hallucination_refine_200s_seed42.json'),
        os.path.join(base, 'outputs', 'prompt_ab_test', 'circo_v7_anti_hallucination_refine_200s_seed42.json'),
    ]
    v7_path = next((p for p in v7_candidates if os.path.exists(p)), None)

    with open(rjson, encoding='utf-8') as f:
        all_s = json.load(f)
    random.seed(42)
    n = min(200, len(all_s))
    indices = sorted(random.sample(range(len(all_s)), n))
    orig_descs = [all_s[i]['target_description'] for i in indices]
    target_names = [all_s[i].get('target_name', '') for i in indices]
    gt_targets = [all_s[i].get('gt_target_names', []) for i in indices]
    del all_s; gc.collect()

    data = pickle.load(open(feat_path, 'rb'))
    idx_feat = torch.nn.functional.normalize(data['index_features'].float(), dim=-1)
    idx_names = data['index_names']
    del data; gc.collect()

    orig_f = encode_texts(model, orig_descs)

    proxy_dir_candidates = [
        os.path.join(base, 'proxy_cache', 'circo'),
    ]
    proxy_dir = next((p for p in proxy_dir_candidates if os.path.exists(p)), proxy_dir_candidates[0])
    proxy_paths = [os.path.join(proxy_dir, f'proxy_{i:05d}.jpg') for i in indices]
    proxy_f = encode_images(model, preprocess, proxy_paths)

    bl = map_at_k(orig_f, proxy_f, idx_feat, idx_names, target_names, gt_targets, 1.0)

    if v7_path and os.path.exists(v7_path):
        with open(v7_path, encoding='utf-8') as f:
            v7_descs = json.load(f)['refined_descs']
        v7_f = encode_texts(model, v7_descs)
        ens_f = torch.nn.functional.normalize(beta * orig_f + (1 - beta) * v7_f, dim=-1)
        er = map_at_k(ens_f, proxy_f, idx_feat, idx_names, target_names, gt_targets, alpha)
    else:
        print("  (No v7 cache for CIRCO, showing Plan A only)")
        er = map_at_k(orig_f, proxy_f, idx_feat, idx_names, target_names, gt_targets, alpha)

    for m in ['mAP@5', 'mAP@50', 'R@50']:
        d = er[m] - bl[m]
        mark = '+' if d > 0 else ('=' if d == 0 else '-')
        print(f"  {m}: BL={bl[m]:>6}  Ens={er[m]:>6}  {d:>+6.1f} {mark}")

    del idx_feat, orig_f, proxy_f; gc.collect()
    return {'baseline': bl, 'ensemble': er}


def run_genecis(name, label, base, model, preprocess, beta, alpha):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    rjson_candidates = [
        os.path.join(base, 'results', f'{name}_full.json'),
        os.path.join(base, 'outputs', f'{name}_full.json'),
    ]
    rjson = next((p for p in rjson_candidates if p and os.path.exists(p)), None)
    if not rjson:
        print(f"  SKIP: results not found"); return None

    v7_candidates = [
        os.path.join(base, 'v7_caches', f'{name}_v7_anti_hallucination_refine_cache_200s_seed42.json'),
        os.path.join(base, 'outputs', 'prompt_ab_test', f'{name}_v7_anti_hallucination_refine_cache_200s_seed42.json'),
    ]
    v7_path = next((p for p in v7_candidates if os.path.exists(p)), None)
    if not v7_path:
        print(f"  SKIP: v7 cache not found"); return None

    # Need dataset images for per-query gallery
    dataset_base = None
    for db in [os.path.join(base, 'datasets', 'GENECIS'), r'Z:\datasets\GENECIS']:
        if os.path.exists(db):
            dataset_base = db; break
    if not dataset_base:
        print(f"  SKIP: GeneCIS dataset not found (need SSHFS Z: mount)"); return None

    # Add src to path for datasets module
    src_candidates = [os.path.join(base, 'src'), r'Z:\src']
    for sp in src_candidates:
        if os.path.exists(sp) and sp not in sys.path:
            sys.path.insert(0, sp)

    try:
        import datasets as ds_module
    except ImportError:
        print(f"  SKIP: cannot import datasets module"); return None

    with open(rjson, encoding='utf-8') as f:
        all_s = json.load(f)
    random.seed(42)
    indices = sorted(random.sample(range(len(all_s)), min(200, len(all_s))))
    orig_descs = [all_s[i]['target_description'] for i in indices]
    del all_s; gc.collect()

    with open(v7_path, encoding='utf-8') as f:
        v7_descs = json.load(f)['refined_descs']

    orig_f = encode_texts(model, orig_descs)
    v7_f = encode_texts(model, v7_descs)

    proxy_dir_candidates = [
        os.path.join(base, 'proxy_cache', name),
    ]
    proxy_dir = next((p for p in proxy_dir_candidates if os.path.exists(p)), proxy_dir_candidates[0])
    proxy_paths = [os.path.join(proxy_dir, f'proxy_{i:05d}.jpg') for i in indices]
    proxy_f = encode_images(model, preprocess, proxy_paths)

    data_split = '_'.join(name.split('_')[1:])
    prop_file = os.path.join(dataset_base, 'genecis', data_split + '.json')

    if 'object' in name:
        datapath = os.path.join(dataset_base, 'coco2017', 'val2017')
        eval_ds = ds_module.COCOValSubset(root_dir=datapath, val_split_path=prop_file,
                                          data_split=data_split, transform=preprocess)
    else:
        datapath = os.path.join(dataset_base, 'Visual_Genome', 'VG_All')
        if not os.path.exists(datapath):
            datapath = os.path.join(dataset_base, 'Visual_Genome', 'VG_ALL')
        eval_ds = ds_module.VAWValSubset(image_dir=datapath, val_split_path=prop_file,
                                         data_split=data_split, transform=preprocess)

    topk = [1, 2, 3]

    def eval_genecis(text_feats, proxy_feats, alpha_val):
        hits = {k: [] for k in topk}
        valid = 0
        for qi, idx in enumerate(indices):
            sample = eval_ds[idx]
            if sample is None: continue
            gallery_and_target = sample[3]
            with torch.no_grad():
                gf = torch.nn.functional.normalize(
                    model.encode_image(gallery_and_target.to(DEVICE)).float().cpu(), dim=-1)
            tf = torch.nn.functional.normalize(text_feats[qi].unsqueeze(0), dim=-1)
            sim = (tf @ gf.T).squeeze(0)
            if alpha_val < 1.0:
                pf = torch.nn.functional.normalize(proxy_feats[qi].unsqueeze(0), dim=-1)
                sim = alpha_val * sim + (1 - alpha_val) * (pf @ gf.T).squeeze(0)
            rank = torch.argsort(sim, descending=True)
            pos = (rank == 0).nonzero(as_tuple=True)[0].item()
            for k in topk:
                hits[k].append(1.0 if pos < k else 0.0)
            valid += 1
        return {f'R@{k}': round(np.mean(v) * 100, 2) for k, v in hits.items()}, valid

    bl, n = eval_genecis(orig_f, proxy_f, 1.0)
    ens_f = torch.nn.functional.normalize(beta * orig_f + (1 - beta) * v7_f, dim=-1)
    er, _ = eval_genecis(ens_f, proxy_f, alpha)

    for m in ['R@1', 'R@2', 'R@3']:
        d = er[m] - bl[m]
        mark = '+' if d > 0 else ('=' if d == 0 else '-')
        print(f"  {m}: BL={bl[m]:>6}  Ens={er[m]:>6}  {d:>+6.1f} {mark}")

    del orig_f, v7_f, proxy_f; gc.collect()
    return {'baseline': bl, 'ensemble': er, 'valid_samples': n}


def main():
    args = parse_args()
    base = args.base or detect_base()
    beta = args.beta
    alpha = args.alpha

    print(f"Device: {DEVICE}")
    print(f"Base: {base}")
    print(f"Params: beta={beta} alpha={alpha}")
    print(f"Datasets: {args.datasets}")

    model, preprocess = clip.load('ViT-L/14', device=DEVICE, jit=False)
    model.eval()

    results = {}
    ds_list = args.datasets
    run_all = 'all' in ds_list

    if run_all or 'dress' in ds_list:
        r = run_fiq_cirr('fashioniq_dress', 'FashionIQ dress', base, model, preprocess, beta, alpha)
        if r: results['dress'] = r

    if run_all or 'shirt' in ds_list:
        r = run_fiq_cirr('fashioniq_shirt', 'FashionIQ shirt', base, model, preprocess, beta, alpha)
        if r: results['shirt'] = r

    if run_all or 'toptee' in ds_list:
        r = run_fiq_cirr('fashioniq_toptee', 'FashionIQ toptee', base, model, preprocess, beta, alpha)
        if r: results['toptee'] = r

    if run_all or 'cirr' in ds_list:
        r = run_fiq_cirr('cirr', 'CIRR', base, model, preprocess, beta, alpha)
        if r: results['cirr'] = r

    if run_all or 'circo' in ds_list:
        r = run_circo(base, model, preprocess, beta, alpha)
        if r: results['circo'] = r

    genecis_names = [
        ('genecis_change_object', 'GeneCIS change_object'),
        ('genecis_focus_object', 'GeneCIS focus_object'),
        ('genecis_change_attribute', 'GeneCIS change_attribute'),
        ('genecis_focus_attribute', 'GeneCIS focus_attribute'),
    ]
    for gname, glabel in genecis_names:
        short = gname.replace('genecis_', 'g_')
        if run_all or short in ds_list or 'genecis_all' in ds_list or gname in ds_list:
            r = run_genecis(gname, glabel, base, model, preprocess, beta, alpha)
            if r: results[glabel] = r

    del model, preprocess; gc.collect()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY (beta={beta}, alpha={alpha})")
    print(f"{'='*60}")
    all_ok = True
    for label, r in results.items():
        bl = r['baseline']
        er = r['ensemble']
        metrics = [m for m in bl.keys() if 'R@10' in m or 'R@50' in m or 'mAP@5' in m or 'mAP@50' in m or 'R@1' in m or 'R@3' in m]
        if not metrics:
            metrics = list(bl.keys())[:2]
        parts = []
        for m in metrics[:2]:
            d = er[m] - bl[m]
            if d < 0: all_ok = False
            parts.append(f"{m}: {d:>+5.1f}")
        ok = all(er[m] >= bl[m] for m in metrics[:2])
        print(f"  {label:<25s}  {'  '.join(parts)}  [{'OK' if ok else 'FAIL'}]")

    print(f"\n  All main metrics positive: {'YES' if all_ok else 'NO'}")

    out_path = os.path.join(base, 'ensemble_results_all.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved to {out_path}")
    except:
        out_path2 = os.path.join(r'D:\osrcir_remote', 'ensemble_results_all.json')
        with open(out_path2, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved to {out_path2}")


if __name__ == '__main__':
    main()
