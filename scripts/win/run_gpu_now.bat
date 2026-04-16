@echo off
echo START %date% %time% > D:\osrcir_remote\gpu_log2.txt
D:\env-py311\python.exe -u D:\osrcir_remote\win_genecis_precompute.py >> D:\osrcir_remote\gpu_log2.txt 2>&1
echo END %date% %time% >> D:\osrcir_remote\gpu_log2.txt
