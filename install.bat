@echo off
echo Installing application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if setup.py exists
if not exist "setup.py" (
    echo Error: setup.py not found in current directory
    pause
    exit /b 1
)

REM Run setup.py
echo Running setup.py...
python setup.py install

if errorlevel 1 (
    echo.
    echo Installation failed!
    pause
    exit /b 1
) else (
    echo.
    echo Installation completed successfully!
    pause
)