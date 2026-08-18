@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ITC Belt Monitor — Demo (Video File)
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo   First-time setup detected. Running installation...
    echo.
    call install.bat
    if errorlevel 1 (
        echo   [ERROR] Installation failed.
        pause
        exit /b 1
    )
    echo.
)

call venv\Scripts\activate.bat

:: ---- Check belt config ----
if not exist "config\belt_config_top.json" (
    echo   [WARN] No belt calibration found.
    echo   Run calibrate_corners.bat first.
    pause
    exit /b 1
)

:: ---- Get video path ----
set "VIDEO=%~1"
if "!VIDEO!"=="" (
    echo   Enter the path to the video file:
    echo.
    set /p "VIDEO=  Video path: "
)

if "!VIDEO!"=="" (
    echo   [ERROR] No video path provided.
    pause
    exit /b 1
)

if not exist "!VIDEO!" (
    echo   [ERROR] Video file not found: !VIDEO!
    pause
    exit /b 1
)

:: ---- Launch dashboard in background ----
echo   Starting dashboard (background)...
start "ITC Dashboard" cmd /c "call venv\Scripts\activate.bat && python -m streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 2>nul"
timeout /t 3 /nobreak >nul
echo   [OK] Dashboard: http://localhost:8501
echo.

:: ---- Launch pipeline ----
set "CONFIG=config\belt_config_top.json"
set "BELT_SPEED=2.0"

echo ============================================================
echo   Video:      !VIDEO!
echo   Config:     %CONFIG%
echo   Belt speed: !BELT_SPEED!
echo   Dashboard:  http://localhost:8501
echo   MJPEG:      http://localhost:8503/live
echo.
echo   Processing will run at full speed (not real-time).
echo   Dashboard auto-refreshes every 10 seconds.
echo   Press Ctrl+C to stop.
echo ============================================================
echo.

python live_pipeline.py --video "!VIDEO!" --config %CONFIG% --speed !BELT_SPEED! --stream --db output/live.db

echo.
echo   Processing complete.
echo   Dashboard may still be running — close its window manually.
pause
