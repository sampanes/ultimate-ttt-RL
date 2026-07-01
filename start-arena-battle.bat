@echo off
setlocal
cd /d "%~dp0"

REM ============================================================
REM  start-arena-battle.bat - ONE-CLICK "watch it battle":
REM  starts the trainer AND the GUI, then opens the dashboard.
REM
REM    * trainer -> window "uttt-arena-train"  (evolves the population)
REM    * GUI     -> window "uttt-arena"  on http://127.0.0.1:5050/
REM
REM  Both windows are titled so the ports dashboard / Task Manager can
REM  name them, and both child launchers are idempotent (re-running
REM  this never stacks duplicates). GUI is started first so its exact-
REM  title teardown can't clip the trainer.
REM
REM  Stop everything with: stop-arena-battle.bat
REM ============================================================

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Create it first:  python -m venv .venv
  pause
  exit /b 1
)

call "%~dp0start-arena.bat"
call "%~dp0start-arena-train.bat"

echo Waiting ~8s for the dashboard to spin up (torch + engine load) ...
timeout /t 8 /nobreak >nul
start "" "http://127.0.0.1:5050/"

echo.
echo   Trainer window  : uttt-arena-train
echo   GUI window      : uttt-arena
echo   This PC         : http://127.0.0.1:5050/
echo   Phone/Tailscale : http://^<your-tailscale-ip^>:5050/   (find it: tailscale ip -4)
echo   (If the page is blank, give it a few more seconds and refresh.)
echo.
echo   To stop BOTH    : stop-arena-battle.bat
echo.
endlocal
