@echo off
REM ============================================================
REM  stop-arena-battle.bat - tear down BOTH the trainer and the GUI
REM  that start-arena-battle.bat launched.
REM ============================================================
call "%~dp0stop-arena-train.bat"
call "%~dp0stop-arena.bat"
echo.
echo All arena processes stopped.
