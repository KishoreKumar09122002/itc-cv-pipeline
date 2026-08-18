@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ITC Belt Monitor — Live Pipeline
echo ============================================================
echo.

:: ---- Check installation ----
if not exist "venv\Scripts\activate.bat" (
    echo   [ERROR] Not installed. Run install.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

:: ---- Get RTSP URL ----
set "RTSP_URL=%~1"
if "!RTSP_URL!"=="" (
    echo   Enter the RTSP stream URL.
    echo.
    echo   Examples:
    echo     rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/101
    echo     rtsp://192.168.1.50:8554/live
    echo.
    echo   Tip: Run discover_cameras.bat to find URLs automatically.
    echo.
    set /p "RTSP_URL=  RTSP URL: "
)

if "!RTSP_URL!"=="" (
    echo   [ERROR] No RTSP URL provided.
    pause
    exit /b 1
)

:: ---- Check belt config ----
set "CONFIG=config\belt_config_top.json"
if not exist "%CONFIG%" (
    echo   [ERROR] Belt config not found: %CONFIG%
    echo   Run calibrate_corners.bat to create one.
    pause
    exit /b 1
)

:: ---- Parse optional arguments ----
set "BELT_SPEED=2.0"
set "EXTRA_ARGS=--record --stream --db output/live.db"

if not "%~2"=="" set "BELT_SPEED=%~2"

echo   Stream:     !RTSP_URL!
echo   Config:     %CONFIG%
echo   Belt speed: !BELT_SPEED!
echo   Database:   output/live.db
echo   MJPEG:      http://localhost:8503/live
echo   Dashboard:  Run start_dashboard.bat in another terminal
echo.
echo   Press Ctrl+C to stop the pipeline.
echo ============================================================
echo.

python live_pipeline.py --url "!RTSP_URL!" --config %CONFIG% --speed !BELT_SPEED! %EXTRA_ARGS%

echo.
echo   Pipeline stopped.
pause
