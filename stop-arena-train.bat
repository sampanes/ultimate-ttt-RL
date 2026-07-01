@echo off
REM ============================================================
REM  stop-arena-train.bat - tear down the "uttt-arena-train" trainer.
REM  The trainer binds no port, so we match it by window title only.
REM ============================================================

echo Stopping "uttt-arena-train" ...
taskkill /FI "WINDOWTITLE eq uttt-arena-train*" /T /F
if errorlevel 1 echo No "uttt-arena-train" window found - already stopped.
echo Done.
