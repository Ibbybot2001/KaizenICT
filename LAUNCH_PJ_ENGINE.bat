@echo off
title PJ/ICT LIVE ENGINE (PRODUCTION V2.1)
color 0A
echo =============================================================
echo    PJ/ICT EXECUTION ENGINE - CERTIFIED PRODUCTION
echo =============================================================
echo.
echo [1/3] SETTING ENVIRONMENT...
set PROJECT_DIR=C:\Users\CEO\ICT reinforcement
cd /d %PROJECT_DIR%

echo [2/3] CLEANING PREVIOUS SESSIONS...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 >nul

echo [3/3] LAUNCHING LIVE SYSTEM...
echo.
echo -> Starting Dashboard (Monitoring)...
start "PJ_DASHBOARD" cmd /c "python live_dashboard.py"
timeout /t 1 >nul

echo -> Starting Engine (Execution Core)...
python run_live_engine.py

echo.
echo =============================================================
echo    ENGINE STOPPED. CHECK LOGS FOR DETAILS.
echo =============================================================
pause
