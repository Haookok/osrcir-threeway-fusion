"""
Three-signal ensemble evaluation — runs on Windows GPU.
Self-contained: only needs baseline results JSONs, v7 refine caches, CLIP feature caches, and proxy images.
All read from D:\osrcir_remote\ (mapped from Linux server).
"""
import json, os, pickle, random, gc, sys
import numpy as np, torch, clip, PIL.Image

BETA = 0.7   # orig weight (vs refined)
ALPHA = 0.9  # text weight (vs proxy image)
BASE = r"D:\osrcir_remote"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def retrieve(text_f, proxy_f, idx_n, idx_names, target_names, alpha):
    sim = text_f @ idx_n.T
    if alpha < 1.0:
        sim = alpha * sim + (1 - alpha) * (proxy_f @ idx_n.T)
    si = torch.argsort(1 - sim, dim=-1).cpu()
    sn = np.array(idx_names)[si]
    lb = torch.tensor(sn == np.array(target_names).reshape(-1, 1))
    return {f'R@{k}': round((lb[:, :k].sum() / len(lb)).item() * 100, 2) for k in [1, 5, 10, 50]}


def main():
    print(f"Device: {DEVICE}")
    print(f"Ensemble params: beta={BETA} alpha={ALPHA}")
    print(f"Base dir: {BASE}")

    model, preprocess = clip.load('ViT-L/14', device=DEVICE, jit=False)
    model.eval()

    results_all = {}

    for name, label, rjson_name, feat_name, proxy_subdir, v7_name in [
        ('dress', 'FashionIQ dress',
         'fashioniq_dress_full.json',
         'fashioniq_dress_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl',
         'fashioniq_dress',
         'fashioniq_dress_v7_anti_hallucination_refine_200s_seed42.json'),
        ('shirt', 'FashionIQ shirt',
         'fashioniq_shirt_full.json',
         'fashioniq_shirt_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl',
         'fashioniq_shirt',
         'fashioniq_shirt_v7_anti_hallucination_refine_200s_seed42.json'),
        ('toptee', 'FashionIQ toptee',
         'fashioniq_toptee_full.json',
         'fashioniq_toptee_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl',
         'fashioniq_toptee',
         'fashioniq_toptee_v7_anti_hallucination_refine_200s_seed42.json'),
        ('cirr', 'CIRR',
         'cirr_full.json',
         'cirr_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl',
         'cirr',
         'cirr_v7_anti_hallucination_refine_200s_seed42.json'),
    ]:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        rjson = os.path.join(BASE, 'results', rjson_name)
        with open(rjson) as f:
            all_s = json.load(f)
        random.seed(42)
        indices = sorted(random.sample(range(len(all_s)), 200))
        orig_descs = [all_s[i]['target_description'] for i in indices]
        target_names = [all_s[i].get('target_name', '') for i in indices]
        del all_s; gc.collect()

        v7_path = os.path.join(BASE, 'v7_caches', v7_name)
        with open(v7_path) as f:
            v7_descs = json.load(f)['refined_descs']

        feat_path = os.path.join(BASE, 'features', feat_name)
        data = pickle.load(open(feat_path, 'rb'))
        idx_feat = torch.nn.functional.normalize(data['index_features'].float(), dim=-1)
        idx_names = data['index_names']
        del data; gc.collect()

        orig_f = encode_texts(model, orig_descs)
        v7_f = encode_texts(model, v7_descs)

        proxy_paths = [os.path.join(BASE, 'proxy_cache', proxy_subdir, f'proxy_{i:05d}.jpg')
                       for i in indices]
        proxy_f = encode_images(model, preprocess, proxy_paths)

        # Baseline
        bl = retrieve(orig_f, proxy_f, idx_feat, idx_names, target_names, 1.0)

        # Ensemble
        ens_f = torch.nn.functional.normalize(BETA * orig_f + (1 - BETA) * v7_f, dim=-1)
        er = retrieve(ens_f, proxy_f, idx_feat, idx_names, target_names, ALPHA)

        for m in ['R@1', 'R@5', 'R@10', 'R@50']:
            d = er[m] - bl[m]
            mark = '+' if d > 0 else ('=' if d == 0 else '-')
            print(f"  {m}: Baseline={bl[m]:>6}  Ensemble={er[m]:>6}  delta={d:>+6.1f} {mark}")

        results_all[label] = {'baseline': bl, 'ensemble': er}
        del idx_feat, orig_f, v7_f, proxy_f; gc.collect()

    del model, preprocess; gc.collect()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY (beta={BETA}, alpha={ALPHA})")
    print(f"{'='*60}")
    all_positive = True
    for label, r in results_all.items():
        d10 = r['ensemble']['R@10'] - r['baseline']['R@10']
        d50 = r['ensemble']['R@50'] - r['baseline']['R@50']
        mark = 'OK' if d10 >= 0 and d50 >= 0 else 'FAIL'
        if d10 < 0 or d50 < 0:
            all_positive = False
        print(f"  {label:<20s}  R@10: {d10:>+5.1f}  R@50: {d50:>+5.1f}  [{mark}]")

    print(f"\n  All positive: {'YES' if all_positive else 'NO'}")

    out_path = os.path.join(BASE, 'ensemble_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == '__main__':
    main()
