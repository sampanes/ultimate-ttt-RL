@echo off
REM ============================================================
REM  stop-arena.bat - tear down the "uttt-arena" GUI server.
REM  1) kills the named window (and its python child tree)
REM  2) falls back to whatever is LISTENING on :5050
REM ============================================================

echo Stopping "uttt-arena" ...
REM  exact title (no wildcard) so it does NOT also kill "uttt-arena-train";
REM  the :5050 port fallback below backs this up if the title ever misses.
taskkill /FI "WINDOWTITLE eq uttt-arena" /T /F
if not errorlevel 1 goto done

echo No matching window - checking port 5050 ...
set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :5050 ^| findstr LISTENING') do (
  set "FOUND=1"
  echo   killing PID %%p
  taskkill /PID %%p /F
)
if not defined FOUND echo Nothing listening on :5050 - already stopped.

:done
echo Done.
