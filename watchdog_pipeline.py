"""Pipeline health watchdog — runs independently, no Claude tokens.

Checks pipeline process status every 30 minutes.
Reads last progress line, GPU memory, elapsed time.
Logs to output/pipeline_watchdog.txt. Auto-exits when pipeline dies.
"""
import time
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
LOG = "output/pipeline_watchdog.txt"
PIPELINE_LOG = "output/pipeline_log.txt"
CHECK_INTERVAL = 1800  # 30 minutes


def is_pid_alive(pid):
    import ctypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    return False


def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip()
        parts = [x.strip() for x in out.split(",")]
        return "GPU: %s%% util | %s/%s MB | %s°C" % tuple(parts)
    except Exception:
        return "GPU: unavailable"


def get_last_progress():
    try:
        with open(PIPELINE_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 4096)
            f.seek(-read_size, 2)
            lines = f.read().decode("utf-8", errors="replace").strip().split("\n")
            for line in reversed(lines):
                if "Frame" in line or "%" in line or "fps" in line:
                    return line.strip()
            return lines[-1].strip() if lines else "no output"
    except FileNotFoundError:
        return "log file not found"
    except Exception as e:
        return "error reading log: %s" % e


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (ts, msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    pipeline_pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    start_time = time.time()

    log("PIPELINE WATCHDOG STARTED (pipeline PID: %s)" % pipeline_pid)
    log("  Check interval: %ds" % CHECK_INTERVAL)

    check_num = 0
    while True:
        time.sleep(CHECK_INTERVAL)
        check_num += 1
        elapsed_min = (time.time() - start_time) / 60

        alive = is_pid_alive(pipeline_pid) if pipeline_pid else "unknown"
        gpu = get_gpu_info()
        progress = get_last_progress()

        log("CHECK #%d (%.0f min elapsed)" % (check_num, elapsed_min))
        log("  Pipeline alive: %s" % alive)
        log("  %s" % gpu)
        log("  Last progress: %s" % progress)

        if pipeline_pid and not alive:
            log("Pipeline process (PID %d) EXITED — watchdog stopping." % pipeline_pid)
            total_hrs = elapsed_min / 60
            log("Total runtime: %.1f hours" % total_hrs)
            break

    log("PIPELINE WATCHDOG STOPPED")
