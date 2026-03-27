@echo off
chcp 65001 >nul 2>&1
echo ========================================
echo   Track4World Platform Starting...
echo ========================================
echo.

cd /d "E:\bishe2"
set PATH=E:\Conda\envs\track4world\Scripts;E:\Conda\envs\track4world\Library\bin;%PATH%
E:\Conda\envs\track4world\python.exe platform\app.py

echo.
echo Press any key to exit...
pause >nul
