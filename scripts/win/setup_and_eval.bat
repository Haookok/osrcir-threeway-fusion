@echo off
echo === Setup drives and run GeneCIS eval === >> D:\osrcir_remote\setup_log.txt 2>&1
echo %date% %time% >> D:\osrcir_remote\setup_log.txt 2>&1

REM Map SSHFS mounts to drive letters
net use Z: /delete 2>nul
net use Y: /delete 2>nul
net use Z: "\\sshfs.k\root@1.15.92.20\osrcir" /persistent:no >> D:\osrcir_remote\setup_log.txt 2>&1
net use Y: "\\sshfs.k\root@1.15.92.20\data\disk\datasets" /persistent:no >> D:\osrcir_remote\setup_log.txt 2>&1

echo Drive Z: >> D:\osrcir_remote\setup_log.txt 2>&1
dir Z:\outputs /b >> D:\osrcir_remote\setup_log.txt 2>&1

echo Drive Y: >> D:\osrcir_remote\setup_log.txt 2>&1
dir Y:\ /b >> D:\osrcir_remote\setup_log.txt 2>&1

REM Run GeneCIS GPU evaluation
echo Starting GPU eval... >> D:\osrcir_remote\setup_log.txt 2>&1
D:\env-py311\python.exe Z:\win_eval_genecis.py >> D:\osrcir_remote\setup_log.txt 2>&1

echo === Done === >> D:\osrcir_remote\setup_log.txt 2>&1
echo %date% %time% >> D:\osrcir_remote\setup_log.txt 2>&1
