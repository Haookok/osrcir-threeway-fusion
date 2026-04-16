#!/bin/bash
cd /home/haomingyang03/code/osrcir
exec /usr/bin/python3 -u scripts/eval/grid_search_genecis.py --datasets genecis_focus_attribute \
    > outputs/full_pipeline/focus_attr_grid.log 2>&1
