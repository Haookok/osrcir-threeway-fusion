#!/bin/bash
set -e
cd "$(dirname "$0")/src"

echo "============================================"
echo "  Tonight's Pipeline - $(date)"
echo "  Budget: 70元 MiniMax + DashScope"
echo "============================================"

# Priority 1: CIRR (2650 proxy + 4181 V7 = ~95元, proxy部分66元)
echo ">>> [1/4] CIRR (大头)"
python3 -u run_full_pipeline.py \
    --dataset cirr \
    --dataset_path ../datasets/CIRR \
    --proxy_workers 3 \
    --refine_workers 3

# Priority 2: toptee补齐 (42 proxy + 98 V7 = ~1.7元)
echo ">>> [2/4] FIQ toptee 补齐"
python3 -u run_full_pipeline.py \
    --dataset fashioniq_toptee \
    --dataset_path ../datasets/FASHIONIQ \
    --proxy_workers 3 \
    --refine_workers 3

# Priority 3: dress补齐 (13 proxy + 70 V7 = ~0.8元)
echo ">>> [3/4] FIQ dress 补齐"
python3 -u run_full_pipeline.py \
    --dataset fashioniq_dress \
    --dataset_path ../datasets/FASHIONIQ \
    --proxy_workers 3 \
    --refine_workers 3

# Priority 4: shirt补齐 (8 proxy + 62 V7 = ~0.6元)
echo ">>> [4/4] FIQ shirt 补齐"
python3 -u run_full_pipeline.py \
    --dataset fashioniq_shirt \
    --dataset_path ../datasets/FASHIONIQ \
    --proxy_workers 3 \
    --refine_workers 3

echo ""
echo "============================================"
echo "  Tonight complete - $(date)"
echo "============================================"
