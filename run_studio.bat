@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   NEURALWAVE AI -- AI MUSIC GENERATION STUDIO
echo ============================================================
echo.

where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=py"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_CMD=python"
    ) else (
        echo [ERROR] Python was not found in PATH!
        echo Please ensure Python 3.10+ is installed from python.org or Microsoft Store.
        pause
        exit /b 1
    )
)

echo [INFO] Using Python interpreter: !PY_CMD!
echo [INFO] Launching NeuralWave AI Studio in your browser...
echo.

!PY_CMD! -m streamlit run app.py --browser.gatherUsageStats false

pause
