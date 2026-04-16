@echo off
echo === GeneCIS v2 Eval Start === > D:\osrcir_remote\genecis_v2_log.txt 2>&1
echo %date% %time% >> D:\osrcir_remote\genecis_v2_log.txt 2>&1

D:\env-py311\python.exe D:\osrcir_remote\win_genecis_eval_v2.py >> D:\osrcir_remote\genecis_v2_log.txt 2>&1

echo === Done === >> D:\osrcir_remote\genecis_v2_log.txt 2>&1
echo %date% %time% >> D:\osrcir_remote\genecis_v2_log.txt 2>&1
