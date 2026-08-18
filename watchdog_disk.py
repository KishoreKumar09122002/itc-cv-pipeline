"""Disk space watchdog — runs independently, no Claude tokens.

Checks C: and E: drive free space every 30 minutes.
Logs to output/disk_watchdog.txt. Auto-exits when pipeline process dies.
"""
import time
import os
import ctypes
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
LOG = "output/disk_watchdog.txt"
CHECK_INTERVAL = 1800  # 30 minutes
C_WARN_GB = 5.0
E_WARN_GB = 50.0


def get_free_gb(drive):
    free = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        drive, None, None, ctypes.pointer(free))
    return free.value / (1024 ** 3)


def is_pid_alive(pid):
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    return False


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (ts, msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    pipeline_pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    log("DISK WATCHDOG STARTED (pipeline PID: %s)" % pipeline_pid)
    log("  C: warn < %.1f GB | E: warn < %.1f GB | interval: %ds" % (
        C_WARN_GB, E_WARN_GB, CHECK_INTERVAL))

    while True:
        c_free = get_free_gb("C:\\")
        e_free = get_free_gb("E:\\")

        c_status = "OK" if c_free >= C_WARN_GB else "WARNING — LOW"
        e_status = "OK" if e_free >= E_WARN_GB else "WARNING — LOW"

        log("C: %.1f GB free [%s] | E: %.1f GB free [%s]" % (
            c_free, c_status, e_free, e_status))

        if c_free < C_WARN_GB:
            log("*** ALERT: C: drive below %.1f GB! ***" % C_WARN_GB)
        if e_free < E_WARN_GB:
            log("*** ALERT: E: drive below %.1f GB! ***" % E_WARN_GB)

        if pipeline_pid and not is_pid_alive(pipeline_pid):
            log("Pipeline process (PID %d) no longer running — exiting." % pipeline_pid)
            break

        time.sleep(CHECK_INTERVAL)

    log("DISK WATCHDOG STOPPED")
