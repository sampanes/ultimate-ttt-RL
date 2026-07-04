"""Send Ctrl+C to a console process so it can exit gracefully (finally blocks
run, resume state gets saved). taskkill /F cannot do this -- it hard-kills and
training scripts lose their optimizer/buffer resume payloads.

Usage:
    python scripts/send_ctrl_c.py <pid>

The PID may be the cmd window's or the python child's -- the event goes to
every process attached to that console. This helper detaches from its own
console first, so run it from a DIFFERENT console than the target.
"""
import ctypes
import sys


def main():
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        sys.exit("usage: python scripts/send_ctrl_c.py <pid>")
    pid = int(sys.argv[1])
    k = ctypes.windll.kernel32
    k.FreeConsole()
    if not k.AttachConsole(pid):
        sys.exit("AttachConsole(%d) failed, err=%d "
                 "(process gone or no console?)" % (pid, k.GetLastError()))
    k.SetConsoleCtrlHandler(None, 1)   # do not kill ourselves with the event
    if not k.GenerateConsoleCtrlEvent(0, 0):   # CTRL_C to the whole console
        sys.exit("GenerateConsoleCtrlEvent failed, err=%d" % k.GetLastError())
    print("ctrl-c sent to console of pid %d" % pid)


if __name__ == "__main__":
    main()
