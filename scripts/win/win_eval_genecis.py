"""
GeneCIS Three-Way Fusion Evaluation - Windows GPU version.
Reads from Z: (Samba) or D:/osrcir_remote.
"""
import json, os, sys, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ALPHA = 0.9
BETA = 0.7

UNC_OSRCIR = r'\\sshfs.k\root@1.15.92.20\osrcir'
UNC_DATASETS = r'\\sshfs.k\root@1.15.92.20\data\disk\datasets'

ROOT = None
for p in [UNC_OSRCIR, 'Z:/', 'Z:/osrcir', 'D:/osrcir_remote']:
    if os.path.exists(os.path.join(p, 'outputs')):
        ROOT = p
        break
if ROOT is None:
    ROOT = '.'

GENECIS_PATH = None
for p in [os.path.join(UNC_DATASETS, 'GENECIS'),
          os.path.join(ROOT, 'datasets', 'GENECIS'),
          'Z:/datasets/GENECIS', 'Y:/', 'Y:/GENECIS',
          'D:/osrcir_remote/datasets/GENECIS']:
    if os.path.exists(os.path.join(p, 'genecis')):
        GENECIS_PATH = p
        break

LOGFILE = 'D:/osrcir_remote/genecis_gpu_eval.log'
os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOGFILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

DATASETS = {
    'genecis_change_object': {'ann': 'change_object.json', 'img': 'coco'},
    'genecis_focus_object':  {'ann': 'focus_object.json',  'img': 'coco'},
    'genecis_change_attribute': {'ann': 'change_attribute.json', 'img': 'vg'},
    'genecis_focus_attribute':  {'ann': 'focus_attribute.json',  'img': 'vg'},
}

def find(rel, dirs):
    for d in dirs:
        p = os.path.join(d, rel)
        if os.path.exists(p):
            return p
    return os.path.join(dirs[0], rel)

def gallery_path(img_id, img_type):
    if img_type == 'coco':
        return os.path.join(GENECIS_PATH, 'coco2017', 'val2017', f'{int(img_id):012d}.jpg')
    for sub in ['VG_All', 'VG_100K', 'VG_100K_2']:
        p = os.path.join(GENECIS_PATH, 'Visual_Genome', sub, f'{img_id}.jpg')
        if os.path.exists(p):
            return p
    return os.path.join(GENECIS_PATH, 'Visual_Genome', 'VG_All', f'{img_id}.jpg')

def load_clip():
    log("Loading CLIP ViT-L-14...")
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_path = r'C:\Users\12427\.cache\clip\ViT-L-14.pt'
    if os.path.exists(clip_path):
        model = torch.jit.load(clip_path, map_location=device).eval()
        import clip as clip_lib
        _, preprocess = clip_lib.load('ViT-L/14', device='cpu')
        tokenizer = lambda texts: clip_lib.tokenize(texts, truncate=True)
    else:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
    log(f"  CLIP on {device} in {time.time()-t0:.1f}s")
    return model, preprocess, tokenizer, device

def evaluate(name, cfg, model, preprocess, tokenizer, device):
    log(f"\n{'='*60}\n  {name}\n{'='*60}")
    out_dirs = [os.path.join(ROOT, 'outputs')]
    pipe_dirs = [os.path.join(ROOT, 'outputs', 'full_pipeline')]
    proxy_dirs = [os.path.join(ROOT, 'proxy_cache', name)]

    bl_path = find(f'{name}_full.json', out_dirs)
    rc_path = find(f'{name}_v7_refine_cache.json', pipe_dirs)
    proxy_dir = proxy_dirs[0]
    ann_path = os.path.join(GENECIS_PATH, 'genecis', cfg['ann']) if GENECIS_PATH else ''

    if not os.path.exists(bl_path):
        log(f"  SKIP: {bl_path} not found"); return None
    if not os.path.exists(ann_path):
        log(f"  SKIP: {ann_path} not found"); return None

    baseline = json.load(open(bl_path, encoding='utf-8'))
    annotation = json.load(open(ann_path, encoding='utf-8'))
    refine_map = {}
    if os.path.exists(rc_path):
        refine_map = {r['index']: r for r in json.load(open(rc_path, encoding='utf-8'))}
        log(f"  {len(refine_map)} refinements")
    else:
        log(f"  No refine, D2=D1")

    total = min(len(baseline), len(annotation))
    log(f"  Samples: {total}")

    hits_b = {1:0, 2:0, 3:0}
    hits_e = {1:0, 2:0, 3:0}
    hits_3 = {1:0, 2:0, 3:0}
    valid = proxy_used = 0
    t0 = time.time()

    with torch.no_grad():
        for idx in range(total):
            s = baseline[idx]; a = annotation[idx]
            d1 = s.get('target_description', '')
            if not d1: continue
            d2 = d1
            if idx in refine_map:
                d2 = refine_map[idx].get('refined_description', d1)

            gi = a.get('gallery', []); ti = a.get('target', {})
            if cfg['img'] == 'coco':
                tid = ti.get('val_image_id'); gids = [g['val_image_id'] for g in gi]
            else:
                tid = ti.get('image_id'); gids = [g['image_id'] for g in gi]
            if tid is None or not gids: continue
            if tid not in gids: gids.append(tid)
            trank = gids.index(tid)

            imgs = []; skip = False
            for gid in gids:
                gp = gallery_path(gid, cfg['img'])
                if not os.path.exists(gp): skip = True; break
                try: imgs.append(preprocess(Image.open(gp).convert('RGB')))
                except: skip = True; break
            if skip: continue

            gf = F.normalize(model.encode_image(torch.stack(imgs).to(device)).float(), dim=-1)
            d1f = F.normalize(model.encode_text(tokenizer([d1]).to(device)).float(), dim=-1)
            d2f = F.normalize(model.encode_text(tokenizer([d2]).to(device)).float(), dim=-1)
            ef = F.normalize(BETA * d1f + (1-BETA) * d2f, dim=-1)

            sb = (d1f @ gf.T).squeeze(0); se = (ef @ gf.T).squeeze(0)
            pb = (torch.argsort(sb, descending=True) == trank).nonzero(as_tuple=True)[0].item()
            pe = (torch.argsort(se, descending=True) == trank).nonzero(as_tuple=True)[0].item()

            for k in [1,2,3]:
                if pb < k: hits_b[k] += 1
                if pe < k: hits_e[k] += 1

            pp = os.path.join(proxy_dir, f'proxy_{idx:05d}.jpg')
            if os.path.exists(pp):
                try:
                    pf = F.normalize(model.encode_image(preprocess(Image.open(pp).convert('RGB')).unsqueeze(0).to(device)).float(), dim=-1)
                    s3 = ALPHA * se + (1-ALPHA) * (pf @ gf.T).squeeze(0)
                    p3 = (torch.argsort(s3, descending=True) == trank).nonzero(as_tuple=True)[0].item()
                    proxy_used += 1
                except: p3 = pe
            else: p3 = pe
            for k in [1,2,3]:
                if p3 < k: hits_3[k] += 1

            valid += 1
            if valid % 500 == 0:
                el = time.time()-t0; r = valid/el
                log(f"    [{valid}/{total}] {el:.0f}s {r:.1f}/s ETA={(total-idx)/r:.0f}s")

    if valid == 0: log("  ERROR: 0 valid"); return None
    log(f"  Valid: {valid}/{total}, proxy: {proxy_used}")

    mb = {f'R@{k}': hits_b[k]/valid*100 for k in [1,2,3]}
    me = {f'R@{k}': hits_e[k]/valid*100 for k in [1,2,3]}
    m3 = {f'R@{k}': hits_3[k]/valid*100 for k in [1,2,3]}

    log(f"  {'Metric':<8} {'Base':>8} {'Ens':>8} {'3Way':>8} {'Delta':>8}")
    log(f"  {'-'*45}")
    for k in ['R@1','R@2','R@3']:
        d = m3[k]-mb[k]; sg = '+' if d>0 else ''
        log(f"  {k:<8} {mb[k]:>8.2f} {me[k]:>8.2f} {m3[k]:>8.2f} {sg}{d:>7.2f}")
    log(f"  Time: {time.time()-t0:.0f}s")
    return {'dataset':name,'valid':valid,'total':total,'baseline':mb,'ensemble':me,'threeway':m3}

def main():
    open(LOGFILE,'w').write(f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log(f"ROOT={ROOT}\nGENECIS={GENECIS_PATH}")
    model, preprocess, tokenizer, device = load_clip()
    results = {}
    for n, c in DATASETS.items():
        r = evaluate(n, c, model, preprocess, tokenizer, device)
        if r: results[n] = r
    log(f"\n{'='*60}\n  SUMMARY beta={BETA} alpha={ALPHA}\n{'='*60}")
    for n, r in results.items():
        mb,me,m3 = r['baseline'],r['ensemble'],r['threeway']
        log(f"\n  {n} ({r['valid']}/{r['total']}):")
        for k in ['R@1','R@2','R@3']:
            d=m3[k]-mb[k]; sg='+' if d>0 else ''
            log(f"    {k}: base={mb[k]:.2f} ens={me[k]:.2f} 3way={m3[k]:.2f} ({sg}{d:.2f})")
    rp = 'D:/osrcir_remote/genecis_results.json'
    json.dump(results, open(rp,'w'), indent=2)
    log(f"\nSaved: {rp}")
    try:
        sp = os.path.join(ROOT,'outputs','full_pipeline','genecis_eval_summary.json')
        json.dump(results, open(sp,'w'), indent=2)
        log(f"Also: {sp}")
    except: pass

if __name__ == '__main__':
    main()
