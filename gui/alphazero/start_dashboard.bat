@echo off
setlocal
cd /d "%~dp0..\.."

REM ============================================================
REM  start_dashboard.bat -- AlphaZero training dashboard (viewer only)
REM
REM  This does NOT start training. Start training separately: start_goat.bat
REM  (wraps python -m scripts.expert_iter --resume).
REM
REM  Dashboard: http://[ip]:7654/gui/alphazero/
REM  Serves via gui.alphazero.dashboard_server: same static file serving as
REM  the old http.server (reads loss_logs/metrics_log.jsonl + state.json,
REM  zero training overhead) plus /api/play, the on-demand watch-a-game
REM  endpoint (one CPU game per click, torch loaded lazily on first use).
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
start "az-dashboard" .venv\Scripts\python -m gui.alphazero.dashboard_server --port 7654

echo.
echo AZ dashboard started (viewer only -- does not start training).
echo.
echo   Local     : http://localhost:7654/gui/alphazero/
echo   Tailscale : http://[your-tailscale-ip]:7654/gui/alphazero/
echo.
echo Stop: taskkill /FI "WINDOWTITLE eq az-dashboard" /T /F
echo.
endlocal
