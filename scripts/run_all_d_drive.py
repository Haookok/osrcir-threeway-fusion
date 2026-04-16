"""
Complete CIRCO + GeneCIS ensemble evaluation.
ALL data from D:\osrcir_remote (local). No network drives needed.
"""
import json, os, sys, pickle, random, gc, numpy as np, torch, clip, PIL.Image

BETA, ALPHA = 0.7, 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D = r'D:\osrcir_remote'

sys.path.insert(0, os.path.join(D, 'src'))

print(f"Device: {DEVICE}", flush=True)
print(f"Base: {D}", flush=True)

print("Loading CLIP...", flush=True)
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
            feats.append(model.encode_image(preprocess(PIL.Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)).float().cpu().squeeze(0))
        else:
            feats.append(torch.zeros(768))
    return torch.nn.functional.normalize(torch.stack(feats), dim=-1)

# ===================== CIRCO =====================
print(f"\n{'='*60}\n  CIRCO\n{'='*60}", flush=True)

with open(os.path.join(D, 'results', 'circo_full.json'), encoding='utf-8') as f:
    all_s = json.load(f)
random.seed(42)
n = min(200, len(all_s))
indices = sorted(random.sample(range(len(all_s)), n))
orig_descs = [all_s[i]['target_description'] for i in indices]
target_names = [all_s[i].get('target_name', '') for i in indices]
gt_targets = [all_s[i].get('gt_target_names', []) for i in indices]
del all_s; gc.collect()

with open(os.path.join(D, 'v7_caches', 'circo_v7_anti_hallucination_refine_200s_seed42.json'), encoding='utf-8') as f:
    v7_descs = json.load(f)['refined_descs']

print("  Loading features (local)...", flush=True)
data = pickle.load(open(os.path.join(D, 'features', 'circo_qwen-vl-max-latest_ViT-L-14_val_img_features.pkl'), 'rb'))
idx_feat = torch.nn.functional.normalize(data['index_features'].float(), dim=-1)
idx_names = data['index_names']
del data; gc.collect()

orig_f = encode_texts(orig_descs)
v7_f = encode_texts(v7_descs)
proxy_paths = [os.path.join(D, 'proxy_cache', 'circo', f'proxy_{i:05d}.jpg') for i in indices]
proxy_f = encode_images(proxy_paths)

def circo_metrics(tf, pf, alpha, ks=[5,10,25,50]):
    sim = tf @ idx_feat.T
    if alpha < 1: sim = alpha*sim + (1-alpha)*(pf@idx_feat.T)
    si = torch.argsort(1-sim, dim=-1).cpu()
    sn = np.array(idx_names)[si]
    maps = {k:[] for k in ks}; recalls = {50:[]}
    for i in range(len(target_names)):
        gts = [str(g) for g in gt_targets[i] if g and g != '']
        if not gts: continue
        sn_top = sn[i][:max(ks)]
        ml = torch.tensor(np.isin(sn_top, gts), dtype=torch.uint8)
        prec = torch.cumsum(ml,0)*ml / torch.arange(1,len(sn_top)+1)
        for k in ks: maps[k].append(float(torch.sum(prec[:k])/min(len(gts),k)))
        single = torch.tensor(sn_top == str(target_names[i]))
        recalls[50].append(float(torch.sum(single[:50])))
    out = {f'mAP@{k}':round(np.mean(v)*100,2) for k,v in maps.items()}
    out[f'R@50'] = round(np.mean(recalls[50])*100,2)
    return out

bl = circo_metrics(orig_f, proxy_f, 1.0)
ens = circo_metrics(torch.nn.functional.normalize(BETA*orig_f+(1-BETA)*v7_f, dim=-1), proxy_f, ALPHA)
for m in ['mAP@5','mAP@10','mAP@25','mAP@50','R@50']:
    d=ens[m]-bl[m]; print(f"  {m}: BL={bl[m]:>7} Ens={ens[m]:>7} {d:>+7.2f} {'+' if d>0 else '-'}", flush=True)
del idx_feat, orig_f, v7_f, proxy_f; gc.collect(); torch.cuda.empty_cache()

# ===================== GeneCIS =====================
import datasets as ds_module

for name, label in [
    ('genecis_change_object','GeneCIS change_object'),
    ('genecis_focus_object','GeneCIS focus_object'),
    ('genecis_change_attribute','GeneCIS change_attribute'),
    ('genecis_focus_attribute','GeneCIS focus_attribute'),
]:
    print(f"\n{'='*60}\n  {label}\n{'='*60}", flush=True)
    rjson = os.path.join(D, 'results', f'{name}_full.json')
    if not os.path.exists(rjson): print("  SKIP: no baseline", flush=True); continue

    with open(rjson, encoding='utf-8') as f: all_s = json.load(f)
    random.seed(42)
    n2 = min(200, len(all_s))
    idx2 = sorted(random.sample(range(len(all_s)), n2))
    od2 = [all_s[i]['target_description'] for i in idx2]
    del all_s; gc.collect()

    v7p = os.path.join(D, 'v7_caches', f'{name}_v7_anti_hallucination_refine_cache_200s_seed42.json')
    if not os.path.exists(v7p): print("  SKIP: no v7", flush=True); continue
    with open(v7p, encoding='utf-8') as f: vd2 = json.load(f)['refined_descs']

    of2 = encode_texts(od2); vf2 = encode_texts(vd2)
    pp2 = [os.path.join(D,'proxy_cache',name,f'proxy_{i:05d}.jpg') for i in idx2]
    pf2 = encode_images(pp2)

    ds_split = '_'.join(name.split('_')[1:])
    prop = os.path.join(D,'datasets','GENECIS','genecis',ds_split+'.json')
    gbase = os.path.join(D,'datasets','GENECIS')
    if 'object' in name:
        dpath = os.path.join(gbase,'coco2017','val2017')
        eds = ds_module.COCOValSubset(root_dir=dpath, val_split_path=prop, data_split=ds_split, transform=preprocess)
    else:
        dpath = os.path.join(gbase,'Visual_Genome','VG_All')
        if not os.path.exists(dpath): dpath = os.path.join(gbase,'Visual_Genome','VG_ALL')
        eds = ds_module.VAWValSubset(image_dir=dpath, val_split_path=prop, data_split=ds_split, transform=preprocess)
    print(f"  Dataset: {dpath}", flush=True)

    topk=[1,2,3]
    @torch.no_grad()
    def eval_g(tf, pf, a):
        hits={k:[] for k in topk}
        for qi,idx in enumerate(idx2):
            s = eds[idx]
            if s is None: continue
            gf = torch.nn.functional.normalize(model.encode_image(s[3].to(DEVICE)).float().cpu(), dim=-1)
            t = torch.nn.functional.normalize(tf[qi].unsqueeze(0), dim=-1)
            sim = (t@gf.T).squeeze(0)
            if a<1: sim = a*sim + (1-a)*(torch.nn.functional.normalize(pf[qi].unsqueeze(0),dim=-1)@gf.T).squeeze(0)
            pos = (torch.argsort(sim,descending=True)==0).nonzero(as_tuple=True)[0].item()
            for k in topk: hits[k].append(1.0 if pos<k else 0.0)
            if qi%50==0: print(f"    [{qi}/{n2}]", flush=True)
        return {f'R@{k}':round(np.mean(v)*100,2) for k,v in hits.items()}

    print("  Baseline...", flush=True)
    bl2 = eval_g(of2, pf2, 1.0)
    print("  Ensemble...", flush=True)
    ef2 = torch.nn.functional.normalize(BETA*of2+(1-BETA)*vf2, dim=-1)
    er2 = eval_g(ef2, pf2, ALPHA)
    for k in topk:
        m=f'R@{k}'; d=er2[m]-bl2[m]
        print(f"  {m}: BL={bl2[m]:>6} Ens={er2[m]:>6} {d:>+6.1f} {'+' if d>0 else '-'}", flush=True)
    del of2,vf2,pf2; gc.collect(); torch.cuda.empty_cache()

print(f"\n{'='*60}\n  ALL DONE\n{'='*60}", flush=True)
