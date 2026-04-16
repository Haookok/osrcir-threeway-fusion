@echo off
echo === GeneCIS GPU Eval v2 === > D:\osrcir_remote\eval2_log.txt 2>&1
echo %date% %time% >> D:\osrcir_remote\eval2_log.txt 2>&1

REM Start SSH tunnel for SMB: local 44500 -> server 445
start /b ssh -o StrictHostKeyChecking=no -L 44500:127.0.0.1:445 -N root@1.15.92.20 >> D:\osrcir_remote\eval2_log.txt 2>&1

REM Wait for tunnel to establish
timeout /t 5 /nobreak >nul

REM Map drives via tunnel
net use Z: /delete 2>nul
net use Y: /delete 2>nul
net use Z: \\127.0.0.1@44500\osrcir /user:root osrcir123 /persistent:no >> D:\osrcir_remote\eval2_log.txt 2>&1
net use Y: \\127.0.0.1@44500\datasets /user:root osrcir123 /persistent:no >> D:\osrcir_remote\eval2_log.txt 2>&1

echo Testing Z: >> D:\osrcir_remote\eval2_log.txt 2>&1
dir Z:\outputs /b >> D:\osrcir_remote\eval2_log.txt 2>&1

REM Run evaluation from Z: drive (all data on server)
echo Running eval... >> D:\osrcir_remote\eval2_log.txt 2>&1
D:\env-py311\python.exe D:\osrcir_remote\win_eval_genecis.py >> D:\osrcir_remote\eval2_log.txt 2>&1

echo === Done === >> D:\osrcir_remote\eval2_log.txt 2>&1
echo %date% %time% >> D:\osrcir_remote\eval2_log.txt 2>&1
