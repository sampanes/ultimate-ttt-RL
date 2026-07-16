@echo off
REM Launch (or relaunch) the expert-iteration training run in a named window.
REM Idempotent: gracefully stops any prior goat-train instance first, so the
REM resume state is saved and --resume picks it straight back up.

cd /d "%~dp0"

call "%~dp0stop_goat.bat"

echo Starting goat-train (expert iteration, dashboard: gui/alphazero/index.html)
REM Window closes on exit so stop_goat.bat's wait loop can see it is gone;
REM progress/state live in loss_logs + models/expert_iter_v2, not the console.
REM Console output (gate/promotion lines, crash tracebacks) is mirrored to
REM loss_logs\goat_console.log so an unattended crash is never silent. -u keeps
REM prints unbuffered so the log tail is current.
if not exist loss_logs mkdir loss_logs
REM goat_supervisor auto-restarts expert_iter after a crash (nonzero exit,
REM 60s backoff, capped); a clean/Ctrl+C exit stops everything, so
REM stop_goat.bat behaves exactly as before.
REM S1 segment (2026-07-13, STRENGTH_NEXT/PENDING runbook): gregory-d2 slice
REM at 0.10, donated equally from opp_mix (0.35->0.30) and rnd_mix (0.15->0.10)
REM so pure self-play stays 0.50. Baseline before enabling: raw gen-6 teacher
REM scored 0.138 vs d2 AND d3 (300 games, seed 8801) -- huge headroom, ENABLE.
REM S4 (2026-07-15): --actors 8 runs generation in 8 worker PROCESSES. Measured
REM on an idle box vs the sequential path, gen-8 teacher, this exact mix:
REM   sequential 775 games/hr | 4 actors 1784 (2.30x) | 8 actors 1909 (2.46x)
REM   12 actors 2021-2173 (2.6-2.8x, GPU pegged at 95%)
REM 8 is the knee: ~2.5x for ~2.1 GiB of extra VRAM. Distribution-preserving
REM (the parent still draws the opponent mix), NOT byte-identical. Drop to
REM --actors 0 to fall back to the sequential path exactly as it was.
start "goat-train" cmd /c "set CUBLAS_WORKSPACE_CONFIG=:4096:8&& .venv\Scripts\python -u -m scripts.goat_supervisor --resume --greg_mix 0.10 --opp_mix 0.30 --rnd_mix 0.10 --actors 8 >> loss_logs\goat_console.log 2>&1"

echo.
echo Running check:  tasklist /FI "WINDOWTITLE eq goat-train*"
echo Live log:       type loss_logs\goat_console.log
echo Stop with:      stop_goat.bat
