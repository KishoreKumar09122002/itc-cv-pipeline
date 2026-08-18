@echo off
echo.
echo ============================================================
echo   ITC Belt Monitor — Dashboard
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo   [ERROR] Not installed. Run install.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

echo   Starting dashboard...
echo   Open in browser: http://localhost:8501
echo.
echo   Press Ctrl+C to stop.
echo ============================================================
echo.

python -m streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0

echo.
echo   Dashboard stopped.
pause
