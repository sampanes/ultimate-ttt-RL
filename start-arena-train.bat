@echo off
setlocal
cd /d "%~dp0"

REM ============================================================
REM  start-arena-train.bat - launch the Arena TRAINER in a window
REM  NAMED "uttt-arena-train" so it is easy to find and kill.
REM
REM  This is the process that actually plays games and evolves the
REM  population (writes models\arena\arena_state.json). It binds NO
REM  port - the GUI (start-arena.bat) just reads that same file.
REM
REM  Idempotent: tears down any prior trainer before starting, so
REM  re-running this never stacks duplicate trainers.
REM  Stop it with: stop-arena-train.bat   (or Ctrl+C in the window)
REM ============================================================

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Create it first:  python -m venv .venv
  pause
  exit /b 1
)

REM --- tear down any previous trainer (precise: won't touch the GUI window) ---
taskkill /FI "WINDOWTITLE eq uttt-arena-train*" /T /F >nul 2>&1

echo Starting "uttt-arena-train" (runs forever; Ctrl+C in its window to stop) ...
start "uttt-arena-train" cmd /c "cd /d %~dp0 && .venv\Scripts\python -X utf8 -m arena.train_arena || pause"

echo.
echo   Window title : uttt-arena-train
echo   Writes       : models\arena\arena_state.json
echo   To stop      : run stop-arena-train.bat  (or Ctrl+C in that window)
echo.
endlocal
