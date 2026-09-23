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
    echo [1/5] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Virtual environment found.
)

echo [2/5] Making sure dependencies are current...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo [3/5] Checking OpenCV window support...
".venv\Scripts\python.exe" -c "import cv2,re,sys; b=cv2.getBuildInformation(); sys.exit(0 if re.search(r'GUI:\s+(WIN32|QT|GTK|COCOA)', b, re.I) else 1)"
if errorlevel 1 (
    echo OpenCV GUI support is missing. Repairing the Windows OpenCV build...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --force-reinstall --no-deps opencv-python==4.10.0.84
    if errorlevel 1 (
        echo ERROR: Could not repair OpenCV GUI support.
        pause
        exit /b 1
    )

    ".venv\Scripts\python.exe" -c "import cv2,re,sys; b=cv2.getBuildInformation(); sys.exit(0 if re.search(r'GUI:\s+(WIN32|QT|GTK|COCOA)', b, re.I) else 1)"
    if errorlevel 1 (
        echo ERROR: OpenCV still does not have window support after repair.
        echo Try deleting the .venv folder and running this launcher again.
        pause
        exit /b 1
    )
) else (
    echo OpenCV GUI support is available.
)

echo [4/5] Starting vision services...
echo First launch note: OCR may download its recognition model.
echo.

echo [5/5] Starting Overwatch Vision...
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
