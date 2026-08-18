@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ITC Belt Monitor — Belt Corner Calibration
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo   [ERROR] Not installed. Run install.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

echo   This tool captures a frame from the camera and lets you
echo   click the 4 corners of the belt conveyor.
echo.
echo   Click order: top-left, top-right, bottom-right, bottom-left
echo   (as seen from the camera's perspective)
echo.

set "RTSP_URL=%~1"
if "!RTSP_URL!"=="" (
    echo   Enter the RTSP URL to capture a frame from:
    set /p "RTSP_URL=  RTSP URL: "
    echo.
)

if "!RTSP_URL!"=="" (
    echo   [ERROR] No RTSP URL provided.
    pause
    exit /b 1
)

python calibrate_belt.py "!RTSP_URL!"

echo.
echo   Config saved. You can now run start_pipeline.bat.
pause
