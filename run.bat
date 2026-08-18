@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ITC Belt Monitor — Run
echo ============================================================
echo.

:: ---- Check installation ----
if not exist "venv\Scripts\activate.bat" (
    echo   First-time setup detected. Running installation...
    echo.
    call install.bat
    if errorlevel 1 (
        echo   [ERROR] Installation failed. Fix the issues above and retry.
        pause
        exit /b 1
    )
    echo.
    echo ============================================================
    echo   Installation complete. Continuing to launch...
    echo ============================================================
    echo.
)

call venv\Scripts\activate.bat

:: ---- Check belt config ----
if not exist "config\belt_config_top.json" (
    echo   [WARN] No belt calibration found.
    echo   You need to calibrate the belt corners before the pipeline can run.
    echo.
    echo   Run calibrate_corners.bat with your RTSP URL first, then run this again.
    echo.
    pause
    exit /b 1
)

:: ---- Get RTSP URL ----
set "RTSP_URL=%~1"
if "!RTSP_URL!"=="" (
    echo   Enter the RTSP stream URL.
    echo.
    echo   Examples:
    echo     rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/101
    echo     rtsp://192.168.1.50:8554/live
    echo.
    set /p "RTSP_URL=  RTSP URL: "
)

if "!RTSP_URL!"=="" (
    echo   [ERROR] No RTSP URL provided.
    pause
    exit /b 1
)

:: ---- Launch dashboard in background ----
echo   Starting dashboard (background)...
start "ITC Dashboard" cmd /c "call venv\Scripts\activate.bat && python -m streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 2>nul"

:: Wait a moment for Streamlit to start
timeout /t 3 /nobreak >nul

echo   [OK] Dashboard: http://localhost:8501
echo.

:: ---- Launch pipeline (foreground) ----
set "CONFIG=config\belt_config_top.json"
set "BELT_SPEED=2.0"
set "EXTRA_ARGS=--record --stream --db output/live.db"

echo ============================================================
echo   Stream:     !RTSP_URL!
echo   Config:     %CONFIG%
echo   Belt speed: !BELT_SPEED!
echo   Dashboard:  http://localhost:8501
echo   MJPEG:      http://localhost:8503/live
echo.
echo   Press Ctrl+C to stop.
echo ============================================================
echo.

python live_pipeline.py --url "!RTSP_URL!" --config %CONFIG% --speed !BELT_SPEED! %EXTRA_ARGS%

echo.
echo   Pipeline stopped.
echo   NOTE: Dashboard may still be running in its window — close it manually.
pause
