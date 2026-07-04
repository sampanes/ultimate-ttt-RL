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
:wait
timeout /t 3 /nobreak >nul
tasklist /FI "WINDOWTITLE eq goat-train*" /FO TABLE /NH 2>nul | findstr /B "cmd.exe" >nul
if not errorlevel 1 goto wait
echo goat-train stopped.
