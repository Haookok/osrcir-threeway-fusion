@echo off
REM Extract proxy image archives on Windows
REM Run this after SCP transfer completes

cd /d D:\osrcir_remote

echo Extracting circo proxies...
tar xzf circo_proxy_full.tar.gz
echo Extracting dress proxies...
tar xzf fashioniq_dress_proxy_full.tar.gz
echo Extracting shirt proxies...
tar xzf fashioniq_shirt_proxy_full.tar.gz
echo Extracting toptee proxies...
tar xzf fashioniq_toptee_proxy_full.tar.gz

echo.
echo All extracted. Cleaning up archives...
del circo_proxy_full.tar.gz
del fashioniq_dress_proxy_full.tar.gz
del fashioniq_shirt_proxy_full.tar.gz
del fashioniq_toptee_proxy_full.tar.gz

echo Done!
echo.
echo Now run evaluation:
echo D:\env-py311\python.exe D:\osrcir_remote\win_eval_gpu.py
