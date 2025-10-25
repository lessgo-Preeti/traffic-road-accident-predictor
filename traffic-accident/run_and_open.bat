@echo off
echo ========================================
echo Starting Traffic Accident Prediction API
echo ========================================
echo.
echo Server will start at: http://127.0.0.1:8000
echo Your browser will open automatically...
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d "e:\traffic-accident\traffic-accident"

REM Wait 3 seconds then open browser
start /B timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000

REM Start the Flask server
E:\traffic-accident\.venv\Scripts\python.exe app.py
