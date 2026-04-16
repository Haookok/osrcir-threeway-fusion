@echo off
echo === GeneCIS GPU Precompute + Eval === > D:\osrcir_remote\gpu_genecis_log.txt 2>&1
echo %date% %time% >> D:\osrcir_remote\gpu_genecis_log.txt 2>&1

D:\env-py311\python.exe D:\osrcir_remote\win_genecis_precompute.py >> D:\osrcir_remote\gpu_genecis_log.txt 2>&1

echo === Done === >> D:\osrcir_remote\gpu_genecis_log.txt 2>&1
echo %date% %time% >> D:\osrcir_remote\gpu_genecis_log.txt 2>&1
