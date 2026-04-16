import json
ROOT = '/home/haomingyang03/code/osrcir'

for ds in ['change_attribute', 'focus_attribute']:
    new_cache = json.load(open(f'{ROOT}/outputs/full_pipeline/genecis_{ds}_genecis_prompt_cache.json'))
    v7_cache = json.load(open(f'{ROOT}/outputs/full_pipeline/genecis_{ds}_v7_refine_cache.json'))
    v7_map = {e['index']: e for e in v7_cache}
    
    new_words = [len(e['refined_description'].split()) for e in new_cache]
    v7_descs = [v7_map.get(e['index'], {}).get('refined_description', '') for e in new_cache]
    v7_words = [len(d.split()) for d in v7_descs]
    
    print(f"\n=== {ds} ===", flush=True)
    print(f"  New prompt avg words: {sum(new_words)/len(new_words):.1f}", flush=True)
    print(f"  V7 avg words: {sum(v7_words)/len(v7_words):.1f}", flush=True)
    
    empty_new = sum(1 for e in new_cache if not e['refined_description'].strip())
    print(f"  Empty new: {empty_new}", flush=True)
    
    shown = 0
    for e in new_cache[:50]:
        idx = e['index']
        v7 = v7_map.get(idx, {}).get('refined_description', '')
        new_d = e['refined_description']
        if new_d and v7:
            orig = e.get('original_description', '')[:60]
            inst = e.get('instruction', '')
            print(f"\n  [{inst}] D1={orig}", flush=True)
            print(f"    V7:  {v7[:120]}", flush=True)
            print(f"    NEW: {new_d[:120]}", flush=True)
            shown += 1
            if shown >= 5:
                break
