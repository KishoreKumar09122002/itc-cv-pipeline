@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ITC Belt Monitor — Installation
echo ============================================================
echo.

:: ---- Check Python ----
python --version >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Python not found in PATH.
    echo.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   IMPORTANT: Check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo   [OK] %PYVER%

:: ---- Check Python version >= 3.10 ----
for /f "tokens=2 delims= " %%v in ("%PYVER%") do set PYVERNUM=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVERNUM%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if !PYMAJOR! LSS 3 (
    echo   [ERROR] Python 3.10+ required, found %PYVER%
    pause
    exit /b 1
)
if !PYMAJOR! EQU 3 if !PYMINOR! LSS 10 (
    echo   [ERROR] Python 3.10+ required, found %PYVER%
    pause
    exit /b 1
)

:: ---- Create virtual environment ----
if not exist "venv\Scripts\activate.bat" (
    echo   Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo   [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo   [OK] Virtual environment created.
) else (
    echo   [OK] Virtual environment already exists.
)

:: ---- Install packages ----
call venv\Scripts\activate.bat

if exist "tools\pip_packages" (
    echo   Installing packages (offline)...
    pip install --no-index --find-links=tools\pip_packages -r requirements.txt -q 2>nul
    if errorlevel 1 (
        echo   [WARN] Offline install failed, falling back to internet...
        pip install -r requirements.txt -q
    )
) else (
    echo   Installing packages (internet)...
    pip install -r requirements.txt -q
)
if errorlevel 1 (
    echo   [ERROR] Package installation failed.
    pause
    exit /b 1
)
echo   [OK] All packages installed.

:: ---- Verify imports ----
echo   Verifying imports...
python -c "import cv2; import numpy; import torch; import ultralytics; import streamlit; print('   Core packages: OK')"
if errorlevel 1 (
    echo   [ERROR] Core package verification failed.
    pause
    exit /b 1
)

python -c "import openvino; print('   OpenVINO:       OK')" 2>nul
if errorlevel 1 (
    echo   [WARN] OpenVINO not available — will use PyTorch CPU (slower on Intel)
)

:: ---- Check model weights ----
if exist "runs\pose\training_data\runs\overhead_pose_v1\weights\best_openvino_model\best.bin" (
    echo   [OK] Model: OpenVINO (optimized for Intel CPU)
) else if exist "runs\pose\training_data\runs\overhead_pose_v1\weights\best.pt" (
    echo   [OK] Model: PyTorch (best.pt)
) else (
    echo   [WARN] No model weights found — pipeline will download generic model
)

:: ---- Check config ----
if exist "config\belt_config_top.json" (
    echo   [OK] Belt config found
) else (
    echo   [INFO] No belt config yet — run calibrate_corners.bat after install
)

:: ---- Check ffmpeg ----
python -c "import imageio_ffmpeg; print('   ffmpeg:         ' + imageio_ffmpeg.get_ffmpeg_exe())" 2>nul
if errorlevel 1 (
    echo   [WARN] ffmpeg not available — video recording will be disabled
)

:: ---- Create output directories ----
if not exist "output" mkdir output
if not exist "output\live_recordings" mkdir output\live_recordings
if not exist "output\live_annotated" mkdir output\live_annotated
if not exist "output\moving_zones" mkdir output\moving_zones

echo.
echo ============================================================
echo   Installation complete!
echo.
echo   Next steps:
echo     1. Calibrate corners:  calibrate_corners.bat ^<RTSP_URL^>
echo     2. Start everything:   run.bat ^<RTSP_URL^>
echo ============================================================
echo.
pause
