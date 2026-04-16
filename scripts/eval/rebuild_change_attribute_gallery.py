"""
Download required VG images for genecis_change_attribute and rebuild gallery cache.

Steps:
  1. Parse annotation to find all needed VG image IDs (~15k)
  2. Download images from Stanford VG hosting (concurrent)
  3. Encode with ViT-L-14-quickgelu on GPU
  4. Save gallery cache pickle (same format as eval_genecis.py)

Usage:
  python3 -u scripts/eval/rebuild_change_attribute_gallery.py
"""
import json
import os
import pickle
import time
import gc
import sys
import urllib.request
import concurrent.futures
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENECIS_PATH = os.path.join(ROOT, 'datasets', 'GENECIS')
FEAT_CACHE_DIR = os.path.join(ROOT, 'precomputed_cache', 'genecis')
VG_DIR = '/tmp/vg_images'
ANNOTATION = os.path.join(GENECIS_PATH, 'genecis', 'change_attribute.json')
GALLERY_CACHE = os.path.join(FEAT_CACHE_DIR, 'genecis_change_attribute_gallery.pkl')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VG_URL_BASES = [
    "https://cs.stanford.edu/people/rak248/VG_100K_2",
    "https://cs.stanford.edu/people/rak248/VG_100K",
]

DOWNLOAD_WORKERS = 10
ENCODE_BATCH_SIZE = 64


def collect_gallery_ids():
    ann = json.load(open(ANNOTATION))
    ids = set()
    for a in ann:
        for img in a.get('gallery', []):
            ids.add(str(img['image_id']))
        ids.add(str(a['reference']['image_id']))
        ids.add(str(a['target']['image_id']))
    return sorted(ids)


def download_image(gid):
    dest = os.path.join(VG_DIR, f'{gid}.jpg')
    if os.path.exists(dest) and os.path.getsize(dest) > 100:
        return gid, True

    for attempt in range(3):
        for base_url in VG_URL_BASES:
            url = f"{base_url}/{gid}.jpg"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urllib.request.urlopen(req, timeout=20)
                data = resp.read()
                if len(data) > 100:
                    with open(dest, 'wb') as f:
                        f.write(data)
                    return gid, True
            except Exception:
                continue
        if attempt < 2:
            time.sleep(1 * (attempt + 1))
    return gid, False


def download_all(ids):
    os.makedirs(VG_DIR, exist_ok=True)
    already = sum(1 for gid in ids
                  if os.path.exists(os.path.join(VG_DIR, f'{gid}.jpg'))
                  and os.path.getsize(os.path.join(VG_DIR, f'{gid}.jpg')) > 100)
    need = len(ids) - already
    print(f"Total: {len(ids)}, already on disk: {already}, to download: {need}")

    if need == 0:
        print("All images already downloaded.")
        return len(ids), 0

    to_download = [gid for gid in ids
                   if not (os.path.exists(os.path.join(VG_DIR, f'{gid}.jpg'))
                           and os.path.getsize(os.path.join(VG_DIR, f'{gid}.jpg')) > 100)]

    ok, fail = already, 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futures = {ex.submit(download_image, gid): gid for gid in to_download}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            gid, success = future.result()
            if success:
                ok += 1
            else:
                fail += 1
            done = i + 1
            if done % 500 == 0 or done == len(to_download):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1)
                eta = (len(to_download) - done) / max(rate, 0.01) / 60
                print(f"  Download: {done}/{len(to_download)} "
                      f"(ok={ok}, fail={fail}) "
                      f"{rate:.1f}/s, ETA {eta:.0f}min", flush=True)

    elapsed = time.time() - t0
    print(f"Download done: {ok} ok, {fail} failed in {elapsed/60:.1f}min")
    return ok, fail


def load_model():
    import open_clip
    print(f"Loading ViT-L-14-quickgelu on {DEVICE}...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14-quickgelu', pretrained='openai')
    model = model.to(DEVICE).eval()
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    return model, preprocess


@torch.no_grad()
def encode_gallery(model, preprocess, ids):
    paths, valid_ids = [], []
    for gid in ids:
        p = os.path.join(VG_DIR, f'{gid}.jpg')
        if os.path.exists(p) and os.path.getsize(p) > 100:
            paths.append(p)
            valid_ids.append(gid)

    print(f"Encoding {len(valid_ids)}/{len(ids)} gallery images "
          f"(batch={ENCODE_BATCH_SIZE})...", flush=True)
    all_feats = []
    t0 = time.time()
    for i in range(0, len(paths), ENCODE_BATCH_SIZE):
        batch_paths = paths[i:i + ENCODE_BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                imgs.append(preprocess(img))
                img.close()
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        t = torch.stack(imgs).to(DEVICE)
        f = model.encode_image(t).float().cpu()
        all_feats.append(f)
        del t, f, imgs
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        done = min(i + ENCODE_BATCH_SIZE, len(paths))
        if done % 256 == 0 or done == len(paths):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1)
            eta = (len(paths) - done) / max(rate, 0.01)
            print(f"  Encoded: {done}/{len(paths)} "
                  f"({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)
            sys.stdout.flush()

    feats = torch.cat(all_feats, dim=0)
    feats = F.normalize(feats, dim=-1)
    elapsed = time.time() - t0
    print(f"Gallery encoding done: {feats.shape} in {elapsed:.0f}s "
          f"({len(valid_ids)/max(elapsed,1):.1f} img/s)", flush=True)
    return feats, valid_ids


def main():
    print("=" * 60)
    print("  Rebuild genecis_change_attribute gallery cache")
    print("=" * 60)

    ids = collect_gallery_ids()
    print(f"\nStep 1: Collect IDs -> {len(ids)} unique gallery images\n")

    print("Step 2: Download VG images")
    ok, fail = download_all(ids)
    if fail > len(ids) * 0.05:
        print(f"WARNING: {fail} images failed to download ({fail/len(ids)*100:.1f}%)")

    print(f"\nStep 3: Encode with CLIP")
    model, preprocess = load_model()
    feats, valid_ids = encode_gallery(model, preprocess, ids)

    os.makedirs(FEAT_CACHE_DIR, exist_ok=True)
    with open(GALLERY_CACHE, 'wb') as f:
        pickle.dump({'feats': feats, 'ids': valid_ids}, f)
    sz = os.path.getsize(GALLERY_CACHE) / 1024 / 1024
    print(f"\nStep 4: Saved -> {GALLERY_CACHE} ({sz:.1f}MB)")
    print(f"  IDs: {len(valid_ids)}, Shape: {feats.shape}")
    print("\nDone! Now run grid_search_genecis.py --datasets genecis_change_attribute")


if __name__ == '__main__':
    main()
