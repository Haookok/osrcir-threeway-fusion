#!/usr/bin/env python3
import argparse
import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_asins(split_dir, categories):
    asins = set()
    for cate in categories:
        for split in ("train", "val", "test"):
            split_path = os.path.join(split_dir, f"split.{cate}.{split}.json")
            if not os.path.exists(split_path):
                continue
            with open(split_path, "r", encoding="utf-8") as f:
                asins.update(json.load(f))
    return sorted(asins)


def load_official_url_map(meta_dir):
    asin_url = {}
    for cate in ("dress", "shirt", "toptee"):
        mapping_path = os.path.join(meta_dir, f"asin2url.{cate}.txt")
        if not os.path.exists(mapping_path):
            continue
        with open(mapping_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    asin_url[parts[0].strip()] = parts[-1].strip()
    return asin_url


def fetch_bytes(url, headers, timeout, proxy_url=None):
    req = urllib.request.Request(url, headers=headers)
    if proxy_url:
        proxy_handler = urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        opener = urllib.request.build_opener(proxy_handler)
        with opener.open(req, timeout=timeout) as resp:
            return resp.read()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_one(asin, asin_url, image_dir, min_bytes, timeout, retries, headers, proxy_url=None):
    dst = os.path.join(image_dir, f"{asin}.jpg")
    if os.path.exists(dst) and os.path.getsize(dst) >= min_bytes:
        return "skip"

    candidates = []
    official_url = asin_url.get(asin)
    if official_url:
        candidates.append(official_url)
        if official_url.startswith("http://"):
            candidates.append("https://" + official_url[len("http://"):])
    candidates.append(f"https://m.media-amazon.com/images/P/{asin}.01._SCLZZZZZZZ_.jpg")
    candidates.append(f"https://m.media-amazon.com/images/P/{asin}.jpg")

    for url in candidates:
        for _ in range(retries):
            try:
                data = fetch_bytes(url, headers=headers, timeout=timeout, proxy_url=proxy_url)
                if len(data) < min_bytes:
                    continue
                tmp = dst + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, dst)
                return "ok"
            except Exception:
                continue
    return "fail"


def main():
    parser = argparse.ArgumentParser(description="Download FashionIQ images from official metadata")
    parser.add_argument("--dataset-root", default="datasets/FASHIONIQ")
    parser.add_argument("--meta-root", default="tools/fashion-iq-metadata/image_url")
    parser.add_argument("--categories", default="dress,shirt,toptee")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-bytes", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--proxy", default="", help="Optional HTTP proxy host:port, e.g. oversea-squid1.jp.txyun:11080")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    split_dir = os.path.join(args.dataset_root, "image_splits")
    image_dir = os.path.join(args.dataset_root, "images")
    os.makedirs(image_dir, exist_ok=True)

    asins = load_asins(split_dir, categories)
    asin_url = load_official_url_map(args.meta_root)

    stats = {"ok": 0, "skip": 0, "fail": 0, "done": 0}
    lock = threading.Lock()
    headers = {"User-Agent": "Mozilla/5.0"}
    proxy_url = f"http://{args.proxy}" if args.proxy else None
    total = len(asins)
    print(f"Categories: {categories}")
    print(f"Total ASINs to process: {total}")
    print(f"Proxy: {proxy_url or 'none'}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                download_one, asin, asin_url, image_dir,
                args.min_bytes, args.timeout, args.retries, headers, proxy_url
            )
            for asin in asins
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with lock:
                stats[result] += 1
                stats["done"] += 1
                if stats["done"] % args.progress_every == 0 or stats["done"] == total:
                    print(
                        f"[{stats['done']}/{total}] "
                        f"ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}",
                        flush=True,
                    )

    print("Download finished.")


if __name__ == "__main__":
    main()
