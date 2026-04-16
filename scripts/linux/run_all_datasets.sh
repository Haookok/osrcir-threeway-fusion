#!/bin/bash
# Run full pipeline (proxy + V7 refine) for all 9 datasets sequentially.
# Each dataset runs one at a time to avoid overloading APIs.
# Skips already-cached items automatically.
#
# Usage: bash run_all_datasets.sh

set -e
cd "$(dirname "$0")/src"

DATASETS=(
    "fashioniq_dress:../datasets/FASHIONIQ"
    "fashioniq_shirt:../datasets/FASHIONIQ"
    "fashioniq_toptee:../datasets/FASHIONIQ"
    "circo:../datasets/CIRCO"
    "cirr:../datasets/CIRR"
    "genecis_change_object:../datasets/GENECIS"
    "genecis_focus_object:../datasets/GENECIS"
    "genecis_change_attribute:../datasets/GENECIS"
    "genecis_focus_attribute:../datasets/GENECIS"
)

echo "============================================"
echo "  Full Pipeline for ALL 9 Datasets"
echo "  Started: $(date)"
echo "============================================"
echo ""

TOTAL_START=$(date +%s)

for entry in "${DATASETS[@]}"; do
    IFS=':' read -r ds path <<< "$entry"
    echo ""
    echo ">>> Starting: $ds ($(date '+%H:%M:%S'))"
    python3 -u run_full_pipeline.py \
        --dataset "$ds" \
        --dataset_path "$path" \
        --proxy_workers 3 \
        --refine_workers 3
    echo ">>> Finished: $ds ($(date '+%H:%M:%S'))"
    echo ""
done

TOTAL_END=$(date +%s)
ELAPSED=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo "============================================"
echo "  ALL DATASETS COMPLETE"
echo "  Total time: ${HOURS}h ${MINS}m"
echo "  Finished: $(date)"
echo "============================================"
