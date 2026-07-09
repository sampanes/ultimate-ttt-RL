"""Crash supervisor for the unattended expert-iteration run.

Runs scripts.expert_iter as a child process and restarts it ONLY when it dies
with a nonzero exit code (CUDA hiccup, transient OOM, unhandled exception).
A clean exit -- including Ctrl+C, which expert_iter catches, saves state for,
and exits 0 from -- ends the supervisor too, so stop_goat.bat keeps working
exactly as before: one Ctrl+C to the console stops everything gracefully.

Restart policy: 60s backoff between restarts, and at most MAX_RESTARTS_PER_DAY
nonzero exits in any rolling 24h window. A hard bug would otherwise crash-loop
forever and grind the GPU on startup/teardown; past the cap the supervisor
gives up loudly and exits nonzero so the console log shows exactly why.

Usage (start_goat.bat wraps this):
    set CUBLAS_WORKSPACE_CONFIG=:4096:8
    .venv\\Scripts\\python -u -m scripts.goat_supervisor --resume
All arguments are passed through to scripts.expert_iter unchanged.
"""

import subprocess
import sys
import time

BACKOFF_SECS = 60
MAX_RESTARTS_PER_DAY = 20


def main():
    child_cmd = [sys.executable, "-u", "-m", "scripts.expert_iter"] + sys.argv[1:]
    crash_times = []
    attempt = 0
    while True:
        attempt += 1
        print(f"[supervisor] launching expert_iter (attempt {attempt}): "
              f"{' '.join(child_cmd)}", flush=True)
        # Child shares this console, so a Ctrl+C sent to the console (what
        # stop_goat.bat does) reaches the child directly; expert_iter saves
        # resume state and exits 0. We must NOT let subprocess.call's 0.25s
        # KeyboardInterrupt grace TerminateProcess the child mid state-save
        # (that corrupts the resume payload the graceful-stop design protects).
        # So: explicit Popen, then wait() and swallow every Ctrl+C, giving the
        # child unbounded time to finish its finally-block save and exit itself.
        proc = subprocess.Popen(child_cmd)
        while True:
            try:
                rc = proc.wait()
                break
            except KeyboardInterrupt:
                # The child got the same Ctrl+C and is saving state. Keep
                # waiting on it rather than killing it.
                print("[supervisor] Ctrl+C -- waiting for child to save state "
                      "and exit...", flush=True)
        if rc == 0:
            print("[supervisor] expert_iter exited cleanly -- done.", flush=True)
            return 0
        now = time.time()
        crash_times.append(now)
        crash_times = [t for t in crash_times if now - t < 24 * 3600]
        print(f"[supervisor] expert_iter died with exit code {rc} "
              f"({len(crash_times)} crash(es) in the last 24h).", flush=True)
        if len(crash_times) >= MAX_RESTARTS_PER_DAY:
            print(f"[supervisor] [X] {MAX_RESTARTS_PER_DAY} crashes in 24h -- "
                  f"this is a real bug, not a transient. Giving up.", flush=True)
            return 1
        print(f"[supervisor] restarting with --resume in {BACKOFF_SECS}s...",
              flush=True)
        try:
            time.sleep(BACKOFF_SECS)
        except KeyboardInterrupt:
            print("[supervisor] interrupted during backoff -- not restarting.",
                  flush=True)
            return 0
        if "--resume" not in child_cmd:
            child_cmd.append("--resume")


if __name__ == "__main__":
    sys.exit(main())
