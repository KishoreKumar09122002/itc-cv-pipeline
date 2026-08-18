@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ITC Belt Monitor — Setup Streaming Tools
echo ============================================================
echo.
echo   Downloads mediamtx and ffmpeg for streaming a video file
echo   as an RTSP source. Only needed on the streaming machine.
echo.

if not exist "tools" mkdir tools

:: ---- Check mediamtx ----
if exist "tools\mediamtx.exe" (
    echo   [OK] mediamtx already present
) else (
    echo   Downloading mediamtx...
    curl.exe -L -o tools\mediamtx.zip https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_windows_amd64.zip
    if errorlevel 1 (
        echo   [ERROR] Failed to download mediamtx
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path tools\mediamtx.zip -DestinationPath tools\ -Force"
    del tools\mediamtx.zip 2>nul
    if exist "tools\mediamtx.exe" (
        echo   [OK] mediamtx downloaded
    ) else (
        echo   [ERROR] mediamtx.exe not found after extraction
        pause
        exit /b 1
    )
)

:: ---- Check ffmpeg ----
if exist "tools\ffmpeg.exe" (
    echo   [OK] ffmpeg already present
) else (
    echo   Checking ffmpeg via pip...
    if exist "venv\Scripts\activate.bat" (
        call venv\Scripts\activate.bat
        python -c "import imageio_ffmpeg; print('[OK] ffmpeg available via imageio_ffmpeg:', imageio_ffmpeg.get_ffmpeg_exe())" 2>nul
        if !errorlevel! equ 0 goto :ffmpeg_done
    )
    echo   Downloading ffmpeg...
    curl.exe -L -o tools\ffmpeg.zip https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
    if errorlevel 1 (
        echo   [ERROR] Failed to download ffmpeg
        pause
        exit /b 1
    )
    powershell -Command "$z = 'tools\ffmpeg.zip'; $d = 'tools\ffmpeg-temp'; Expand-Archive $z $d -Force; $bin = Get-ChildItem $d -Recurse -Filter 'ffmpeg.exe' | Select -First 1; Copy-Item $bin.FullName 'tools\ffmpeg.exe'; Remove-Item $d -Recurse -Force; Remove-Item $z"
    if exist "tools\ffmpeg.exe" (
        echo   [OK] ffmpeg downloaded
    ) else (
        echo   [ERROR] ffmpeg.exe not found after extraction
        pause
        exit /b 1
    )
)

:ffmpeg_done
echo.
echo ============================================================
echo   Streaming tools ready.
echo   Run stream_video.bat to start streaming a video file.
echo ============================================================
echo.
pause
