#!/bin/bash
cd /root/osrcir/src

echo "============================================"
echo "  GeneCIS Full Pipeline - $(date)"
echo "============================================"

DATASET_PATH="/root/osrcir/datasets/GENECIS"

for ds in genecis_change_object genecis_focus_object genecis_change_attribute genecis_focus_attribute; do
    echo ""
    echo ">>> Starting $ds at $(date)"
    python3 -u run_full_pipeline.py \
        --dataset "$ds" \
        --dataset_path "$DATASET_PATH" \
        --proxy_workers 3 \
        --refine_workers 3
    echo ">>> Finished $ds at $(date)"
    echo ""
done

echo "============================================"
echo "  GeneCIS complete - $(date)"
echo "============================================"
