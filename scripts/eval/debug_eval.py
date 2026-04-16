import json, pickle, random, sys, os
ROOT = '/home/haomingyang03/code/osrcir'

ann = json.load(open(ROOT+'/datasets/GENECIS/genecis/change_attribute.json'))
gd = pickle.load(open(ROOT+'/precomputed_cache/genecis/genecis_change_attribute_gallery.pkl','rb'))
gid_to_idx = {gid: i for i, gid in enumerate(gd['ids'])}

random.seed(42)
sample_indices = sorted(random.sample(range(2111), 200))

baseline_data = pickle.load(open(ROOT+'/precomputed_cache/precomputed/genecis_change_attribute_val_mods_mllm_structural_predictor_prompt_CoT_qwen-vl-max-latest.pkl', 'rb'))
all_samples = baseline_data['generated_results']

skip_reasons = {'no_proxy': 0, 'target_not_found': 0, 'few_gallery': 0, 'ok': 0}

for qi, si in enumerate(sample_indices[:20]):
    ann_item = ann[si]
    target = ann_item['target']
    tid = target.get('image_id')
    gallery = ann_item.get('gallery', [])
    g_ids = [g['image_id'] for g in gallery]
    
    local_indices = []
    target_local = -1
    for gi, gid in enumerate(g_ids):
        if gid in gid_to_idx:
            local_indices.append(gid_to_idx[gid])
            if gid == tid:
                target_local = len(local_indices) - 1
    
    proxy_path = os.path.join(ROOT, f'proxy_cache/genecis_change_attribute/proxy_{si:05d}.jpg')
    has_proxy = os.path.exists(proxy_path)
    
    if qi < 5:
        print(f"[{si}] tid={repr(tid)}, gallery_matches={len(local_indices)}/{len(g_ids)}, "
              f"target_local={target_local}, has_proxy={has_proxy}", flush=True)
    
    if target_local < 0:
        skip_reasons['target_not_found'] += 1
    elif len(local_indices) < 2:
        skip_reasons['few_gallery'] += 1
    elif not has_proxy:
        skip_reasons['no_proxy'] += 1
    else:
        skip_reasons['ok'] += 1

print(f"\nSkip reasons (first 20): {skip_reasons}", flush=True)

# Check proxy availability
proxy_count = sum(1 for si in sample_indices if os.path.exists(
    os.path.join(ROOT, f'proxy_cache/genecis_change_attribute/proxy_{si:05d}.jpg')))
print(f"Proxy available: {proxy_count}/{len(sample_indices)}", flush=True)
