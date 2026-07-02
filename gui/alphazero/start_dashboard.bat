@echo off
setlocal
cd /d "%~dp0..\.."

REM ============================================================
REM  start_dashboard.bat -- AlphaZero training dashboard (viewer only)
REM
REM  This does NOT start training. Start training separately:
REM    .venv\Scripts\python -m scripts.train_alphazero --network medium --value_tanh ...
REM
REM  Dashboard: http://[ip]:7654/gui/alphazero/
REM  It just reads loss_logs/metrics_log.jsonl -- zero training overhead.
REM
REM  (Arena GUI is separate and NOT needed for AZ runs: start-arena.bat)
REM
REM  Idempotent -- kills a prior instance first.
REM  Stop with:  taskkill /FI "WINDOWTITLE eq az-dashboard" /T /F
REM ============================================================

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found.
  exit /b 1
)

taskkill /FI "WINDOWTITLE eq az-dashboard" /T /F >nul 2>&1
start "az-dashboard" .venv\Scripts\python -m http.server 7654

echo.
echo AZ dashboard started (viewer only -- does not start training).
echo.
echo   Local     : http://localhost:7654/gui/alphazero/
echo   Tailscale : http://[your-tailscale-ip]:7654/gui/alphazero/
echo.
echo Stop: taskkill /FI "WINDOWTITLE eq az-dashboard" /T /F
echo.
endlocal
