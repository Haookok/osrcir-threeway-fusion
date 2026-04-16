@echo off
echo === GeneCIS GPU Eval v3 - User Session === > D:\osrcir_remote\eval3_log.txt 2>&1
echo %date% %time% >> D:\osrcir_remote\eval3_log.txt 2>&1
echo User: %USERNAME% >> D:\osrcir_remote\eval3_log.txt 2>&1

REM Check net use
net use >> D:\osrcir_remote\eval3_log.txt 2>&1

REM List all drives
echo All drives: >> D:\osrcir_remote\eval3_log.txt 2>&1
wmic logicaldisk get caption,providername,description >> D:\osrcir_remote\eval3_log.txt 2>&1

REM Try accessing SSHFS via UNC directly
echo Testing SSHFS UNC: >> D:\osrcir_remote\eval3_log.txt 2>&1
dir "\\sshfs.k\root@1.15.92.20\osrcir\outputs" /b >> D:\osrcir_remote\eval3_log.txt 2>&1
if errorlevel 1 (
    echo SSHFS UNC failed, trying pushd >> D:\osrcir_remote\eval3_log.txt 2>&1
    pushd "\\sshfs.k\root@1.15.92.20\osrcir"
    if errorlevel 1 (
        echo pushd also failed >> D:\osrcir_remote\eval3_log.txt 2>&1
    ) else (
        echo pushd success, cwd: >> D:\osrcir_remote\eval3_log.txt 2>&1
        cd >> D:\osrcir_remote\eval3_log.txt 2>&1
        dir /b >> D:\osrcir_remote\eval3_log.txt 2>&1
        popd
    )
)

echo === Done === >> D:\osrcir_remote\eval3_log.txt 2>&1
echo %date% %time% >> D:\osrcir_remote\eval3_log.txt 2>&1
