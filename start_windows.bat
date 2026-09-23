@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo        Overwatch Vision Launcher
echo ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found in PATH.
    echo Install Python 3.11+ and make sure "Add Python to PATH" is enabled.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment.
        pause
        exit /b 1
    )

    echo [2/3] Installing project dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 (
        echo ERROR: pip upgrade failed.
        pause
        exit /b 1
    )

    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment found.
    echo [2/3] Dependencies already set up.
)

echo [3/3] Starting Overwatch Vision...
echo.
echo Open Overwatch before using the detector.
echo Q = quit   D = toggle debug   R = reset tracker
echo.

".venv\Scripts\python.exe" run.py
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
    echo Overwatch Vision exited with code %EXITCODE%.
) else (
    echo Overwatch Vision closed normally.
)

pause
exit /b %EXITCODE%
