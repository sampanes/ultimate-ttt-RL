@echo off
REM Gracefully stop the goat-train run: sends Ctrl+C (NOT taskkill /F) so the
REM trainer's finally block saves optimizer/state and --resume loses nothing.

cd /d "%~dp0"

set FOUND=0
for /f "tokens=2" %%p in ('tasklist /FI "WINDOWTITLE eq goat-train*" /FO TABLE /NH 2^>nul ^| findstr /B "cmd.exe"') do (
  set FOUND=1
  echo Sending Ctrl+C to goat-train console ^(pid %%p^)...
  .venv\Scripts\python scripts\send_ctrl_c.py %%p
)

if "%FOUND%"=="0" (
  echo No goat-train window running.
  goto :eof
)

echo Waiting for goat-train to exit (state save can take a moment)...
REM Bounded loop: ~2 min cap so a wedged trainer can never hang stop_goat
REM (and start_goat, which calls this first) forever. ~3s/iteration * 40 = 2min.
set /a WAITS=0
:wait
REM ping, not timeout: Git Bash puts a Unix timeout.exe on PATH that shadows
REM cmd's and errors out instantly, turning this loop into a busy-spin that can
REM declare "stopped" before the state save finishes. ping is unshadowed.
ping -n 4 127.0.0.1 >nul
set /a WAITS+=1
REM Liveness by BOTH the console window AND a real process check: an orphaned
REM goat_supervisor/expert_iter python (window closed, process survived) would
REM slip past a title-only check and get double-started by the next start_goat.
tasklist /FI "WINDOWTITLE eq goat-train*" /FO TABLE /NH 2>nul | findstr /B "cmd.exe" >nul
if not errorlevel 1 goto stillrunning
wmic process where "name='python.exe' and commandline like '%%goat_supervisor%%'" get processid 2>nul | findstr /R "[0-9]" >nul
if not errorlevel 1 goto stillrunning
wmic process where "name='python.exe' and commandline like '%%scripts.expert_iter%%'" get processid 2>nul | findstr /R "[0-9]" >nul
if not errorlevel 1 goto stillrunning
echo goat-train stopped.
goto :eof
:stillrunning
if %WAITS% GEQ 40 (
  echo [!] goat-train still running after ~2 min -- it may be wedged.
  echo     Check loss_logs\goat_console.log; if truly stuck, kill it manually:
  echo       taskkill /F /IM python.exe   ^(WARNING: loses unsaved resume state^)
  goto :eof
)
goto wait
