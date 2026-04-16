"""
GeneCIS + CIRCO ensemble evaluation.
Large files (CLIP features) read from D:\osrcir_remote\features\ (local).
Dataset images read from Y:\ (SSHFS cloud storage) or Z:\datasets\ (SSHFS main).
Small files (v7 caches, baseline JSONs, proxy images) from Z:\ (SSHFS main).
"""
import json, os, sys, pickle, random, gc, numpy as np, torch, clip, PIL.Image

BETA, ALPHA = 0.7, 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Z = r'Z:'          # server /root/osrcir
Y = r'Y:'          # server /root/data/disk/datasets (cloud storage)
D = r'D:\osrcir_remote'  # local cached large files

sys.path.insert(0, os.path.join(Z, 'src'))

print(f"Device: {DEVICE}")
print(f"Params: beta={BETA} alpha={ALPHA}")
print(f"Z={Z} Y={Y} D={D}")
print("Loading CLIP model...", flush=True)

model, preprocess = clip.load('ViT-L/14', device=DEVICE, jit=False)
model.eval()
tok = lambda t: clip.tokenize(t, context_length=77, truncate=True)
print("CLIP loaded.", flush=True)

@torch.no_grad()
def encode_texts(texts):
    feats = []
    for i in range(0, len(texts), 64):
        feats.append(model.encode_text(tok(texts[i:i+64]).to(DEVICE)).float().cpu())
    return torch.nn.functional.normalize(torch.vstack(feats), dim=-1)

@torch.no_grad()
def encode_images(paths):
    feats = []
    for p in paths:
        if p and os.path.exists(p):
            img = preprocess(PIL.Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
            feats.append(model.encode_image(img).float().cpu().squeeze(0))
        else:
            feats.append(torch.zeros(768))
    return torch.nn.functional.normalize(torch.stack(feats), dim=-1)

# ===================== CIRCO =====================
print(f"\n{'='*60}\n  CIRCO\n{'='*60}", flush=True)

circo_json = os.path.join(Z, 'outputs', 'circo', 'circo_full.json')
print(f"  Loading baseline from {circo_json}...", flush=True)
with open(circo_json, encoding='utf-8') as f:
    all_s = json.load(f)
random.seed(42)
n = min(200, len(all_s))
indices = sorted(random.sample(range(len(all_s)), n))
orig_descs = [all_s[i]['target_description'] for i in indices]
target_names = [all_s[i].get('target_name', '') for i in indices]
gt_targets = [all_s[i].get('gt_target_names', []) for i in indices]
del all_s; gc.collect()

v7_path = os.path.join(Z, 'outputs', 'prompt_ab_test', 'circo_v7_anti_hallucination_refine_200s_seed42.json')
print(f"  Loading v7 cache...", flush=True)
with open(v7_path, encoding='utf-8') as f:
    v7_descs = json.load(f)['refined_descs']

# Features from LOCAL D: drive (fast!)
feat_path = os.path.join(D, 'features', 'circo_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl')
print(f"  Loading features from {feat_path}...", flush=True)
data = pickle.load(open(feat_path, 'rb'))
idx_feat = torch.nn.functional.normalize(data['index_features'].float(), dim=-1)
idx_names = data['index_names']
del data; gc.collect()

print(f"  Encoding texts...", flush=True)
orig_f = encode_texts(orig_descs)
v7_f = encode_texts(v7_descs)
proxy_paths = [os.path.join(Z, 'proxy_cache', 'circo', f'proxy_{i:05d}.jpg') for i in indices]
print(f"  Encoding proxy images...", flush=True)
proxy_f = encode_images(proxy_paths)

def circo_metrics(text_f, proxy_f_in, alpha, ks=[5,10,25,50]):
    sim = text_f @ idx_feat.T
    if alpha < 1.0:
        sim = alpha * sim + (1-alpha) * (proxy_f_in @ idx_feat.T)
    si = torch.argsort(1-sim, dim=-1).cpu()
    sn = np.array(idx_names)[si]
    maps = {k: [] for k in ks}
    recalls = {k: [] for k in [50]}
    for i in range(len(target_names)):
        gts = [str(g) for g in gt_targets[i] if g != '' and g is not None]
        if not gts: continue
        sn_top = sn[i][:max(ks)]
        ml = torch.tensor(np.isin(sn_top, gts), dtype=torch.uint8)
        prec = torch.cumsum(ml, dim=0) * ml
        prec = prec / torch.arange(1, len(sn_top)+1)
        for k in ks:
            maps[k].append(float(torch.sum(prec[:k]) / min(len(gts), k)))
        single = torch.tensor(sn_top == str(target_names[i]))
        for k in [50]:
            recalls[k].append(float(torch.sum(single[:k])))
    out = {f'mAP@{k}': round(np.mean(v)*100, 2) for k,v in maps.items()}
    out.update({f'R@{k}': round(np.mean(v)*100, 2) for k,v in recalls.items()})
    return out

bl = circo_metrics(orig_f, proxy_f, 1.0)
ens_f = torch.nn.functional.normalize(BETA * orig_f + (1-BETA) * v7_f, dim=-1)
er = circo_metrics(ens_f, proxy_f, ALPHA)

for m in ['mAP@5', 'mAP@10', 'mAP@25', 'mAP@50', 'R@50']:
    d = er[m] - bl[m]
    mark = '+' if d > 0 else ('=' if d == 0 else '-')
    print(f"  {m}: BL={bl[m]:>7}  Ens={er[m]:>7}  {d:>+7.2f} {mark}", flush=True)

del idx_feat, orig_f, v7_f, proxy_f; gc.collect()
torch.cuda.empty_cache()

# ===================== GeneCIS =====================
import datasets as ds_module

for name, label in [
    ('genecis_change_object', 'GeneCIS change_object'),
    ('genecis_focus_object', 'GeneCIS focus_object'),
    ('genecis_change_attribute', 'GeneCIS change_attribute'),
    ('genecis_focus_attribute', 'GeneCIS focus_attribute'),
]:
    print(f"\n{'='*60}\n  {label}\n{'='*60}", flush=True)

    rjson = os.path.join(Z, 'outputs', f'{name}_full.json')
    if not os.path.exists(rjson):
        print(f"  SKIP: not found", flush=True); continue

    print(f"  Loading baseline...", flush=True)
    with open(rjson, encoding='utf-8') as f:
        all_s = json.load(f)
    random.seed(42)
    n2 = min(200, len(all_s))
    indices2 = sorted(random.sample(range(len(all_s)), n2))
    orig_descs2 = [all_s[i]['target_description'] for i in indices2]
    del all_s; gc.collect()

    v7_p = os.path.join(Z, 'outputs', 'prompt_ab_test', f'{name}_v7_anti_hallucination_refine_cache_200s_seed42.json')
    if not os.path.exists(v7_p):
        print(f"  SKIP: no v7 cache", flush=True); continue
    with open(v7_p, encoding='utf-8') as f:
        v7_descs2 = json.load(f)['refined_descs']

    print(f"  Encoding texts...", flush=True)
    orig_f2 = encode_texts(orig_descs2)
    v7_f2 = encode_texts(v7_descs2)

    proxy_paths2 = [os.path.join(Z, 'proxy_cache', name, f'proxy_{i:05d}.jpg') for i in indices2]
    print(f"  Encoding proxy images...", flush=True)
    proxy_f2 = encode_images(proxy_paths2)

    # Dataset from Y: drive (cloud storage)
    data_split = '_'.join(name.split('_')[1:])
    prop_file = os.path.join(Y, 'GENECIS', 'genecis', data_split + '.json')
    if not os.path.exists(prop_file):
        prop_file = os.path.join(Z, 'datasets', 'GENECIS', 'genecis', data_split + '.json')

    if 'object' in name:
        datapath = os.path.join(Y, 'GENECIS', 'coco2017', 'val2017')
        if not os.path.exists(datapath):
            datapath = os.path.join(Y, 'CIRCO', 'coco2017', 'val2017')
        print(f"  Dataset: {datapath}", flush=True)
        eval_ds = ds_module.COCOValSubset(root_dir=datapath, val_split_path=prop_file,
                                          data_split=data_split, transform=preprocess)
    else:
        datapath = os.path.join(Y, 'GENECIS', 'Visual_Genome', 'VG_All')
        if not os.path.exists(datapath):
            datapath = os.path.join(Y, 'GENECIS', 'Visual_Genome', 'VG_ALL')
        print(f"  Dataset: {datapath}", flush=True)
        eval_ds = ds_module.VAWValSubset(image_dir=datapath, val_split_path=prop_file,
                                         data_split=data_split, transform=preprocess)

    topk = [1, 2, 3]
    @torch.no_grad()
    def eval_genecis(text_feats, proxy_feats, alpha_val):
        hits = {k: [] for k in topk}
        for qi, idx in enumerate(indices2):
            sample = eval_ds[idx]
            if sample is None: continue
            gat = sample[3]
            gf = torch.nn.functional.normalize(
                model.encode_image(gat.to(DEVICE)).float().cpu(), dim=-1)
            tf = torch.nn.functional.normalize(text_feats[qi].unsqueeze(0), dim=-1)
            sim = (tf @ gf.T).squeeze(0)
            if alpha_val < 1.0:
                pf = torch.nn.functional.normalize(proxy_feats[qi].unsqueeze(0), dim=-1)
                sim = alpha_val * sim + (1-alpha_val) * (pf @ gf.T).squeeze(0)
            rank = torch.argsort(sim, descending=True)
            pos = (rank == 0).nonzero(as_tuple=True)[0].item()
            for k in topk:
                hits[k].append(1.0 if pos < k else 0.0)
            if qi % 50 == 0: print(f"    [{qi}/{n2}]", flush=True)
        return {f'R@{k}': round(np.mean(v)*100, 2) for k,v in hits.items()}

    print(f"  Evaluating baseline...", flush=True)
    bl2 = eval_genecis(orig_f2, proxy_f2, 1.0)
    print(f"  Evaluating ensemble...", flush=True)
    ens_f2 = torch.nn.functional.normalize(BETA * orig_f2 + (1-BETA) * v7_f2, dim=-1)
    er2 = eval_genecis(ens_f2, proxy_f2, ALPHA)

    for m in topk:
        k = f'R@{m}'
        d = er2[k] - bl2[k]
        mark = '+' if d > 0 else ('=' if d == 0 else '-')
        print(f"  {k}: BL={bl2[k]:>6}  Ens={er2[k]:>6}  {d:>+6.1f} {mark}", flush=True)

    del orig_f2, v7_f2, proxy_f2; gc.collect()
    torch.cuda.empty_cache()

print(f"\n{'='*60}\n  ALL DONE\n{'='*60}", flush=True)
