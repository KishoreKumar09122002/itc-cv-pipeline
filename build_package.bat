@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ITC Belt Monitor — Build Deployment Package
echo ============================================================
echo.

:: ---- Configuration ----
set "PACKAGE_NAME=ITC_deploy_package"
set "DEST=%~dp0..\%PACKAGE_NAME%"
set "ZIP=%~dp0..\%PACKAGE_NAME%.zip"

:: ---- Files to include ----
:: Core pipeline
:: Config, model weights, tools, pip packages
:: Deployment scripts
:: Dashboard and utilities

echo   Building package: %DEST%
echo.

if exist "%DEST%" (
    echo   [WARN] Removing old package directory...
    rmdir /s /q "%DEST%"
)
mkdir "%DEST%"

:: ---- Copy core files ----
echo   Copying core pipeline files...
for %%f in (
    live_pipeline.py
    moving_zone_tracker.py
    dashboard.py
    generate_summary.py
    process_recording.py
    discover_cameras.py
    calibrate_belt.py
    supervisor.py
    watchdog_pipeline.py
    watchdog_disk.py
    run_overnight.py
    requirements.txt
) do (
    if exist "%%f" (
        copy /y "%%f" "%DEST%\%%f" >nul
    )
)

:: ---- Copy deployment scripts ----
echo   Copying deployment scripts...
for %%f in (
    install.bat
    start_pipeline.bat
    start_dashboard.bat
    discover_cameras.bat
    calibrate_corners.bat
    run.bat
    build_package.bat
) do (
    if exist "%%f" (
        copy /y "%%f" "%DEST%\%%f" >nul
    )
)

:: ---- Copy directories ----
echo   Copying src...
xcopy /s /e /i /q "src" "%DEST%\src" >nul

echo   Copying config...
xcopy /s /e /i /q "config" "%DEST%\config" >nul

echo   Copying .streamlit...
xcopy /s /e /i /q ".streamlit" "%DEST%\.streamlit" >nul

echo   Copying model weights...
set "WEIGHTS_SRC=runs\pose\training_data\runs\overhead_pose_v1\weights"
set "WEIGHTS_DST=%DEST%\runs\pose\training_data\runs\overhead_pose_v1\weights"
mkdir "%WEIGHTS_DST%" 2>nul
if exist "%WEIGHTS_SRC%\best.pt" copy /y "%WEIGHTS_SRC%\best.pt" "%WEIGHTS_DST%\" >nul
if exist "%WEIGHTS_SRC%\best_openvino_model" (
    xcopy /s /e /i /q "%WEIGHTS_SRC%\best_openvino_model" "%WEIGHTS_DST%\best_openvino_model" >nul
)

echo   Copying tools...
xcopy /s /e /i /q "tools" "%DEST%\tools" >nul

:: ---- Create output directories ----
mkdir "%DEST%\output" 2>nul
mkdir "%DEST%\output\live_recordings" 2>nul
mkdir "%DEST%\output\live_annotated" 2>nul
mkdir "%DEST%\output\moving_zones" 2>nul

:: ---- Calculate size ----
echo.
echo   Calculating package size...
set SIZE=0
for /f "tokens=3" %%s in ('dir /s "%DEST%" ^| findstr "File(s)"') do set SIZE=%%s
echo   Package size: %SIZE% bytes

echo.
echo ============================================================
echo   Package built: %DEST%
echo.
echo   To create a zip, use:
echo     PowerShell: Compress-Archive -Path "%DEST%" -DestinationPath "%ZIP%"
echo.
echo   To deploy on a new machine:
echo     1. Copy %PACKAGE_NAME%.zip to target machine
echo     2. Extract the zip
echo     3. Run install.bat
echo     4. Run start_pipeline.bat
echo ============================================================
echo.
pause
