@echo off
REM Gracefully stop the goat-train run.
REM
REM Primary mechanism: write a STOP sentinel (models\expert_iter_v2\STOP) that
REM expert_iter polls at the top of every block. It breaks out of the loop, its
REM finally-block saves optimizer/state (so --resume loses nothing), and it
REM exits 0; goat_supervisor sees the same file and does NOT relaunch.
REM
REM This is reliable where a console Ctrl+C is NOT: with --eval_server the
REM trainer's multiprocessing actors swallow the console CTRL_C_EVENT on Windows,
REM so the main loop never sees a KeyboardInterrupt. We still fire a Ctrl+C too
REM as a bonus, but the sentinel is what actually stops it.
REM
REM NEVER taskkill /F here -- that can kill the child mid state-save and corrupt
REM the resume payload this graceful path exists to protect.

cd /d "%~dp0"

set "STOPFILE=models\expert_iter_v2\STOP"

REM --- is a run actually active? (process check, not just the window title,
REM     since an orphaned python with the window closed must still be caught) ---
set RUNNING=0
tasklist /FI "WINDOWTITLE eq goat-train*" /FO TABLE /NH 2>nul | findstr /B "cmd.exe" >nul
if not errorlevel 1 set RUNNING=1
wmic process where "name='python.exe' and commandline like '%%goat_supervisor%%'" get processid 2>nul | findstr /R "[0-9]" >nul
if not errorlevel 1 set RUNNING=1
wmic process where "name='python.exe' and commandline like '%%scripts.expert_iter%%'" get processid 2>nul | findstr /R "[0-9]" >nul
if not errorlevel 1 set RUNNING=1

if "%RUNNING%"=="0" (
  echo No goat-train run active.
  if exist "%STOPFILE%" del /f /q "%STOPFILE%"
  goto :eof
)

echo Requesting graceful stop (writing %STOPFILE%)...
type nul > "%STOPFILE%"

REM Bonus: also send a real Ctrl+C to the console window if we can find it.
for /f "tokens=2" %%p in ('tasklist /FI "WINDOWTITLE eq goat-train*" /FO TABLE /NH 2^>nul ^| findstr /B "cmd.exe"') do (
  echo Also sending Ctrl+C to goat-train console ^(pid %%p^)...
  .venv\Scripts\python scripts\send_ctrl_c.py %%p
)

echo Waiting for goat-train to save state and exit...
REM Bounded loop: ~3 min cap so a wedged trainer can never hang stop_goat (and
REM start_goat, which calls this first) forever. ~3s/iteration * 60 = 3min. A
REM block that is mid-promotion-gauntlet can take ~70s before it re-checks STOP,
REM so the cap allows for that plus the state save.
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
REM Stopped. The supervisor removes STOP on a clean stop; delete it here too so a
REM leftover sentinel can never silently halt the next run.
if exist "%STOPFILE%" del /f /q "%STOPFILE%"
echo goat-train stopped.
goto :eof
:stillrunning
if %WAITS% GEQ 60 (
  echo [!] goat-train still running after ~3 min -- the STOP file is set, so it
  echo     will exit at the next block boundary; a promotion gauntlet block can
  echo     take ^~70s. Check loss_logs\goat_console.log. Do NOT taskkill /F
  echo     unless truly wedged ^(that risks the unsaved resume state^).
  goto :eof
)
goto wait
