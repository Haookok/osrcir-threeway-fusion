#!/bin/bash
cd /root/osrcir/src

DATASET_PATH="/root/osrcir/datasets/GENECIS"

echo "============================================"
echo "  GeneCIS Resume - $(date)"
echo "============================================"

# 1. focus_attribute: 需要代理图 + 精炼
echo ">>> [1/4] genecis_focus_attribute (proxy + refine) at $(date)"
python3 -u run_full_pipeline.py \
    --dataset genecis_focus_attribute \
    --dataset_path "$DATASET_PATH" \
    --proxy_workers 2 \
    --refine_workers 3
echo ">>> Finished genecis_focus_attribute at $(date)"

# 2. change_attribute: 代理图大部分有了，需要补全 + 精炼
echo ">>> [2/4] genecis_change_attribute (proxy补全 + refine) at $(date)"
python3 -u run_full_pipeline.py \
    --dataset genecis_change_attribute \
    --dataset_path "$DATASET_PATH" \
    --proxy_workers 2 \
    --refine_workers 3
echo ">>> Finished genecis_change_attribute at $(date)"

# 3. change_object: 补全代理图(~250张) + 补精炼
echo ">>> [3/4] genecis_change_object (补全) at $(date)"
python3 -u run_full_pipeline.py \
    --dataset genecis_change_object \
    --dataset_path "$DATASET_PATH" \
    --proxy_workers 2 \
    --refine_workers 3
echo ">>> Finished genecis_change_object at $(date)"

# 4. focus_object: 补全代理图(~93张) + 补精炼
echo ">>> [4/4] genecis_focus_object (补全) at $(date)"
python3 -u run_full_pipeline.py \
    --dataset genecis_focus_object \
    --dataset_path "$DATASET_PATH" \
    --proxy_workers 2 \
    --refine_workers 3
echo ">>> Finished genecis_focus_object at $(date)"

echo "============================================"
echo "  GeneCIS Resume complete - $(date)"
echo "============================================"
