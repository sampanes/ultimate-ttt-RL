"""A single-instance lock for anything that measures wall time.

WHY THIS EXISTS. On 2026-08-09 two runs of `tools/profile_kernels.py` overlapped
-- a launcher's child had outlived the shell believed to have killed it -- and
both wrote complete-looking results to the same file. Neither crashed. The k=8
kernel time differed by 1.9x between them because they were sharing the GPU,
and the only tell was that two reads of the same path disagreed.

A timing measurement has no internal check that it had the machine to itself,
and a partially-overwritten JSON still parses. So every tool whose output is a
duration takes this lock.

    with single_instance("kernels"):
        ...

Stale locks are not auto-cleared. A lock left behind by a killed process is a
prompt to check what happened, and clearing it automatically would restore
exactly the failure mode this prevents.
"""

import contextlib
import os
import time

LOCK_DIR = os.path.join("results", "_locks")


def lock_path(name):
    return os.path.join(LOCK_DIR, "%s.lock" % name)


def acquire(name):
    os.makedirs(LOCK_DIR, exist_ok=True)
    path = lock_path(name)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            with open(path) as fh:
                who = fh.read().strip()
        except OSError:
            who = "unknown"
        raise SystemExit(
            "[X] another run holds %s (%s).\n"
            "    Concurrent runs share the GPU and both results are then\n"
            "    contended -- which has already invalidated one study. Wait\n"
            "    for it; if it is dead, delete the lock file and re-run."
            % (path, who))
    with os.fdopen(fd, "w") as fh:
        fh.write("pid %d started %s"
                 % (os.getpid(), time.strftime("%Y-%m-%d %H:%M:%S")))
    return path


def release(name):
    try:
        os.remove(lock_path(name))
    except OSError:
        pass


@contextlib.contextmanager
def single_instance(name):
    acquire(name)
    try:
        yield
    finally:
        release(name)


def gpu_baseline(samples=5, gap=0.4):
    """Whatever else is using the GPU, sampled before a run starts.

    Recorded rather than acted on: this is a desktop card and the compositor is
    always doing something. It belongs in the result so a later reader can tell
    a quiet box from a busy one.
    """
    import subprocess
    vals = []
    for _ in range(samples):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            vals.append(float(out.stdout.strip().splitlines()[0]))
        except Exception:
            return None
        time.sleep(gap)
    return {"samples": vals, "mean_pct": sum(vals) / len(vals),
            "max_pct": max(vals)}
