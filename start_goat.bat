@echo off
REM Launch (or relaunch) the expert-iteration training run in a named window.
REM Idempotent: gracefully stops any prior goat-train instance first, so the
REM resume state is saved and --resume picks it straight back up.

cd /d "%~dp0"

call "%~dp0stop_goat.bat"

echo Starting goat-train (expert iteration, dashboard: gui/alphazero/index.html)
REM Window closes on exit so stop_goat.bat's wait loop can see it is gone;
REM progress/state live in loss_logs + models/expert_iter_v2, not the console.
start "goat-train" cmd /c "set CUBLAS_WORKSPACE_CONFIG=:4096:8&& .venv\Scripts\python -m scripts.expert_iter --resume"

echo.
echo Running check:  tasklist /FI "WINDOWTITLE eq goat-train*"
echo Stop with:      stop_goat.bat
