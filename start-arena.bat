@echo off
setlocal
cd /d "%~dp0"

REM ============================================================
REM  start-arena.bat - launch the Arena GUI in a window NAMED
REM  "uttt-arena" so it is easy to find and kill.
REM  Idempotent: tears down any prior instance before starting.
REM  Uses port 5050 to avoid colliding with another local service on :5000.
REM  Stop it with: stop-arena.bat   (or Ctrl+C in the window)
REM ============================================================

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Create it first:  python -m venv .venv
  pause
  exit /b 1
)

REM --- tear down any previous instance first ---
REM  exact title (no wildcard) so it does NOT also kill "uttt-arena-train".
taskkill /FI "WINDOWTITLE eq uttt-arena" /T /F >nul 2>&1

echo Starting "uttt-arena" on http://127.0.0.1:5050 ...
start "uttt-arena" cmd /c "cd /d %~dp0 && .venv\Scripts\python -m arena.gui_server --port 5050 || pause"

echo.
echo   Window title : uttt-arena
echo   URL          : http://127.0.0.1:5050/
echo   To stop      : run stop-arena.bat  (or Ctrl+C in that window)
echo.
endlocal
