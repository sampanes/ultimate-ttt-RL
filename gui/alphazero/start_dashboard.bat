@echo off
setlocal
cd /d "%~dp0..\.."

REM ============================================================
REM  start_dashboard.bat  -- start both monitoring dashboards
REM
REM  Arena GUI    (Flask, port 5050)  : http://[ip]:5050/
REM  AZ dashboard (static, port 7654) : http://[ip]:7654/gui/alphazero/
REM
REM  Both are idempotent -- kills prior instances first.
REM  Stop with:  taskkill /FI "WINDOWTITLE eq uttt-arena" /T /F
REM              taskkill /FI "WINDOWTITLE eq az-dashboard" /T /F
REM ============================================================

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found.
  exit /b 1
)

REM -- Arena GUI (Flask) --
taskkill /FI "WINDOWTITLE eq uttt-arena" /T /F >nul 2>&1
start "uttt-arena" cmd /c ".venv\Scripts\python -m arena.gui_server --port 5050 || pause"

REM -- AZ training dashboard (http.server, zero overhead) --
taskkill /FI "WINDOWTITLE eq az-dashboard" /T /F >nul 2>&1
start "az-dashboard" .venv\Scripts\python -m http.server 7654

echo.
echo Both dashboards started.
echo.
echo   Arena GUI    : http://localhost:5050/
echo   AZ dashboard : http://localhost:7654/gui/alphazero/
echo.
echo Via Tailscale -- replace localhost with your Tailscale IP.
echo.
endlocal
