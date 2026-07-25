@echo off
REM Launch (or relaunch) the expert-iteration training run in a named window.
REM Idempotent: gracefully stops any prior goat-train instance first, so the
REM resume state is saved and --resume picks it straight back up.

cd /d "%~dp0"

call "%~dp0stop_goat.bat"

REM Fresh slate: a leftover STOP sentinel from a prior graceful stop must not
REM immediately halt the new run before it makes any progress. (stop_goat and
REM the supervisor both remove it, but delete here too as belt-and-suspenders.)
if exist "models\expert_iter_v2\STOP" del /f /q "models\expert_iter_v2\STOP"

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
REM S5+S8 (2026-07-16): --eval_server routes every actor's forward through ONE
REM GPU-owning server (batched, single CUDA context); --actors 16 pure-CPU game
REM workers. S8 (C++ fill_planes) cuts each actor's plane build 54x. A/B on the
REM gen-9 teacher, this exact mix, GENERATION games/hr vs the sequential anchor:
REM   sequential 714 (1.00x) | first-cut a12 +S8 2491 (3.49x)
REM   eval-server +S8: a8 9518 (13.3x) | a12 10462 (14.6x)
REM   a16 11509 (16.1x, knee) | a24 10875 (15.2x, past the knee)
REM 16 actors match the 16-game block (one parallel wave). Distribution-preserving,
REM NOT byte-identical. REVERT: drop --eval_server (first-cut context per actor)
REM or --actors 0 (sequential path exactly as it was). See RESULT_S5.md.
start "goat-train" cmd /c "set CUBLAS_WORKSPACE_CONFIG=:4096:8&& .venv\Scripts\python -u -m scripts.goat_supervisor --resume --greg_mix 0.10 --opp_mix 0.30 --rnd_mix 0.10 --actors 16 --eval_server >> loss_logs\goat_console.log 2>&1"

echo.
echo Running check:  tasklist /FI "WINDOWTITLE eq goat-train*"
echo Live log:       type loss_logs\goat_console.log
echo Stop with:      stop_goat.bat
