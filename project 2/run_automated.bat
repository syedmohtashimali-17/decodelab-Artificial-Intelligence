@echo off
TITLE Iris KNN Classifier Pipeline (Automated)
cls
echo ============================================================
echo   Running Iris KNN Pipeline & Generating Visual Outputs...
echo ============================================================
echo.
set PYTHONIOENCODING=utf-8
python main.py --no-interactive
echo.
echo ============================================================
echo   Pipeline complete! All plots saved to ./outputs/
echo ============================================================
echo.
pause
