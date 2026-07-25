@echo off
TITLE Iris KNN Classifier Pipeline
cls
echo ============================================================
echo   Launching Iris KNN Classifier (Interactive Mode)...
echo ============================================================
echo.
set PYTHONIOENCODING=utf-8
python main.py
echo.
echo Press any key to exit...
pause > nul
