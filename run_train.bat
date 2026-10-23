@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   NEURALWAVE AI -- MODEL TRAINING PIPELINE
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
        echo [ERROR] Python was not found!
        pause
        exit /b 1
    )
)

echo [INFO] Step 1: Preprocessing MIDI files...
!PY_CMD! scripts\train.py --preprocess-only

echo.
echo [INFO] Step 2: Training NeuralWave LSTM (10 epochs)...
!PY_CMD! scripts\train.py --epochs 10

echo.
echo [SUCCESS] Training pipeline completed! Check models\best_model.pt
pause
