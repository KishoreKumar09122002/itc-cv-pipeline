@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ITC Belt Monitor — Stream Video as RTSP
echo ============================================================
echo.
echo   This streams a local video file as an RTSP source so another
echo   machine on the network can process it with the live pipeline.
echo.
echo   Requires: mediamtx.exe and ffmpeg.exe in the tools\ folder
echo   or installed on the system. Run setup_streaming.bat first.
echo.

:: ---- Check tools ----
set "MEDIAMTX="
set "FFMPEG="

if exist "tools\mediamtx.exe" (
    set "MEDIAMTX=tools\mediamtx.exe"
) else (
    where mediamtx.exe >nul 2>&1
    if !errorlevel! equ 0 set "MEDIAMTX=mediamtx.exe"
)

if exist "tools\ffmpeg.exe" (
    set "FFMPEG=tools\ffmpeg.exe"
) else (
    where ffmpeg.exe >nul 2>&1
    if !errorlevel! equ 0 (
        set "FFMPEG=ffmpeg.exe"
    ) else (
        if exist "venv\Scripts\activate.bat" (
            call venv\Scripts\activate.bat
            for /f "tokens=*" %%p in ('python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2^>nul') do set "FFMPEG=%%p"
        )
    )
)

if "!MEDIAMTX!"=="" (
    echo   [ERROR] mediamtx not found. Run setup_streaming.bat first.
    pause
    exit /b 1
)
if "!FFMPEG!"=="" (
    echo   [ERROR] ffmpeg not found. Run setup_streaming.bat first.
    pause
    exit /b 1
)

echo   [OK] mediamtx: !MEDIAMTX!
echo   [OK] ffmpeg:   !FFMPEG!
echo.

:: ---- Get video path ----
set "VIDEO=%~1"
if "!VIDEO!"=="" (
    echo   Enter the path to the video file:
    set /p "VIDEO=  Video path: "
    echo.
)

if "!VIDEO!"=="" (
    echo   [ERROR] No video path provided.
    pause
    exit /b 1
)

if not exist "!VIDEO!" (
    echo   [ERROR] File not found: !VIDEO!
    pause
    exit /b 1
)

:: ---- Get this machine's IP ----
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set "IP=%%a"
    set "IP=!IP: =!"
)

echo ============================================================
echo   Streaming: !VIDEO!
echo   RTSP URL:  rtsp://!IP!:8554/live
echo.
echo   Use this URL on the processing machine:
echo     run.bat rtsp://!IP!:8554/live
echo.
echo   Press Ctrl+C to stop streaming.
echo ============================================================
echo.

:: ---- Start mediamtx in background ----
start "MediaMTX" cmd /c "!MEDIAMTX!"
timeout /t 2 /nobreak >nul

:: ---- Stream video with ffmpeg (-re = real-time speed, -stream_loop = loop forever) ----
"!FFMPEG!" -re -stream_loop -1 -i "!VIDEO!" -c:v libx264 -preset ultrafast -tune zerolatency -crf 23 -an -f rtsp rtsp://localhost:8554/live

echo.
echo   Streaming stopped.
taskkill /fi "WINDOWTITLE eq MediaMTX" >nul 2>&1
pause
