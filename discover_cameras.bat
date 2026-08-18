@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ITC Belt Monitor — Camera Discovery (ONVIF)
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo   [ERROR] Not installed. Run install.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

:: ---- Get NVR details ----
set "HOST=%~1"
set "USER=%~2"
set "PASS=%~3"

if "!HOST!"=="" (
    echo   Enter NVR / camera details:
    echo.
    set /p "HOST=  IP address: "
    set /p "USER=  Username:   "
    set /p "PASS=  Password:   "
    echo.
)

if "!HOST!"=="" (
    echo   [ERROR] No IP address provided.
    pause
    exit /b 1
)

python discover_cameras.py --host "!HOST!" --user "!USER!" --pass "!PASS!"

echo.
pause
