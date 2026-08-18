"""Unified pipeline supervisor — instant crash detection + auto-restart.

Spawns the pipeline as a child process and uses process.wait() for
instant crash detection (no polling delay). Background thread monitors
disk space every 5 minutes and logs full health every 30 minutes.

On crash: checks if restart is safe (disk space ok, retries remaining),
waits 10 seconds, then restarts the pipeline from the beginning.

Usage:
  Offline video:
    python supervisor.py --video recording.avi --config config/belt_config_top.json --speed 2.0

  Live RTSP:
    python supervisor.py --live --url rtsp://admin:pass@192.168.1.100:554/stream --config config/belt_config_top.json --speed 2.0

All output goes to output/supervisor_log.txt.
Pipeline stdout goes to output/pipeline_log.txt.
"""
import argparse
import subprocess
import threading
import time
import ctypes
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def build_pipeline_cmd(args):
    if args.live:
        cmd = [
            sys.executable, "-u", "live_pipeline.py",
            "--url", args.url,
            "--config", args.config,
            "--speed", str(args.speed),
        ]
        if args.db:
            cmd.extend(["--db", args.db])
        if args.record:
            cmd.append("--record")
        if args.stream:
            cmd.extend(["--stream", "--stream-port", str(args.stream_port)])
        return cmd
    else:
        cmd = [
            sys.executable, "-u", "process_recording.py",
            "--video", args.video,
            "--config", args.config,
            "--speed", str(args.speed), "--full",
        ]
        return cmd

MAX_RETRIES = 3
DISK_INTERVAL = 300       # 5 minutes
HEALTH_INTERVAL = 1800    # 30 minutes
C_CRITICAL_GB = 1.0
C_WARN_GB = 3.0
E_WARN_GB = 50.0

SUP_LOG = "output/supervisor_log.txt"
PIPE_LOG = "output/pipeline_log.txt"

os.makedirs("output", exist_ok=True)


def get_free_gb(drive):
    free = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        drive, None, None, ctypes.pointer(free))
    return free.value / (1024 ** 3)


def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip()
        parts = [x.strip() for x in out.split(",")]
        return "GPU %s%% util | %s/%s MB VRAM | %s C" % tuple(parts)
    except Exception:
        return "GPU info unavailable"


def get_pipeline_progress():
    try:
        with open(PIPE_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return "no output yet"
            read_size = min(size, 8192)
            f.seek(-read_size, 2)
            lines = f.read().decode("utf-8", errors="replace").strip().split("\n")
            for line in reversed(lines):
                if "Frame" in line or "%" in line or "fps" in line:
                    return line.strip()
            return lines[-1].strip() if lines else "no output"
    except FileNotFoundError:
        return "log not created yet"
    except Exception as e:
        return "error: %s" % e


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (ts, msg)
    print(line, flush=True)
    with open(SUP_LOG, "a") as f:
        f.write(line + "\n")


class Supervisor:
    def __init__(self, pipeline_cmd):
        self.pipeline_cmd = pipeline_cmd
        self.pipeline = None
        self.attempt = 0
        self.stop_event = threading.Event()
        self.pipeline_start_time = None

    def start_pipeline(self):
        self.attempt += 1
        log("LAUNCHING PIPELINE (attempt %d/%d)" % (self.attempt, MAX_RETRIES))
        log("  cmd: %s" % " ".join(self.pipeline_cmd))

        pipe_out = open(PIPE_LOG, "w")
        self.pipeline = subprocess.Popen(
            self.pipeline_cmd,
            stdout=pipe_out,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)))
        self.pipeline_start_time = time.time()
        log("  PID: %d" % self.pipeline.pid)

    def check_disks(self):
        c_free = get_free_gb("C:\\")
        e_free = get_free_gb("E:\\")
        c_tag = "CRITICAL" if c_free < C_CRITICAL_GB else (
            "LOW" if c_free < C_WARN_GB else "OK")
        e_tag = "LOW" if e_free < E_WARN_GB else "OK"
        log("  DISK  C: %.1f GB [%s] | E: %.1f GB [%s]" % (
            c_free, c_tag, e_free, e_tag))
        return c_free

    def full_health_check(self, check_num):
        elapsed = time.time() - self.pipeline_start_time
        elapsed_min = elapsed / 60
        log("HEALTH CHECK #%d (%.0f min into attempt %d)" % (
            check_num, elapsed_min, self.attempt))
        self.check_disks()
        log("  %s" % get_gpu_info())
        log("  Progress: %s" % get_pipeline_progress())

    def monitor_loop(self):
        """Background: disk every 5 min, full health every 30 min."""
        tick = 0
        while not self.stop_event.is_set():
            self.stop_event.wait(DISK_INTERVAL)
            if self.stop_event.is_set():
                break
            tick += 1
            if tick % (HEALTH_INTERVAL // DISK_INTERVAL) == 0:
                self.full_health_check(tick // (HEALTH_INTERVAL // DISK_INTERVAL))
            else:
                c_free = self.check_disks()
                if c_free < C_CRITICAL_GB:
                    log("*** CRITICAL: C: below %.1f GB — pipeline may fail ***" %
                        C_CRITICAL_GB)

    def run(self):
        log("=" * 60)
        log("SUPERVISOR STARTED")
        log("  Max retries: %d | Disk check: %ds | Health: %ds" % (
            MAX_RETRIES, DISK_INTERVAL, HEALTH_INTERVAL))
        log("=" * 60)

        monitor = threading.Thread(target=self.monitor_loop, daemon=True)
        monitor.start()

        while self.attempt < MAX_RETRIES:
            c_free = get_free_gb("C:\\")
            if c_free < C_CRITICAL_GB:
                log("C: drive at %.1f GB — too low to start pipeline. Aborting." %
                    c_free)
                break

            self.start_pipeline()

            try:
                exit_code = self.pipeline.wait()
            except KeyboardInterrupt:
                log("Ctrl+C received — stopping pipeline gracefully.")
                self.pipeline.terminate()
                try:
                    self.pipeline.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.pipeline.kill()
                break

            elapsed = (time.time() - self.pipeline_start_time) / 60

            if exit_code == 0:
                log("=" * 60)
                log("PIPELINE COMPLETED SUCCESSFULLY in %.0f min" % elapsed)
                log("=" * 60)
                break
            else:
                log("=" * 60)
                log("PIPELINE CRASHED — exit code %d after %.0f min" % (
                    exit_code, elapsed))
                log("  Last progress: %s" % get_pipeline_progress())
                log("=" * 60)

                c_free = get_free_gb("C:\\")
                if c_free < C_CRITICAL_GB:
                    log("C: at %.1f GB — not safe to restart. Stopping." % c_free)
                    break

                if self.attempt < MAX_RETRIES:
                    log("Restarting in 10 seconds... (attempt %d/%d next)" % (
                        self.attempt + 1, MAX_RETRIES))
                    time.sleep(10)
                else:
                    log("Max retries (%d) reached. Giving up." % MAX_RETRIES)

        self.stop_event.set()
        log("SUPERVISOR STOPPED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Supervisor")
    parser.add_argument("--live", action="store_true",
                        help="Run live RTSP pipeline instead of offline video")
    parser.add_argument("--url", default="",
                        help="RTSP stream URL (required with --live)")
    parser.add_argument("--video", default="",
                        help="Video file for offline mode")
    parser.add_argument("--config", default="config/belt_config_top.json",
                        help="Belt config JSON path")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Belt speed in pixels/frame")
    parser.add_argument("--db", default="",
                        help="SQLite database path (live mode only)")
    parser.add_argument("--record", action="store_true",
                        help="Record RTSP stream for gap video replay")
    parser.add_argument("--stream", action="store_true",
                        help="Enable MJPEG live stream + annotated video saving")
    parser.add_argument("--stream-port", type=int, default=8503,
                        help="MJPEG stream port (default: 8503)")
    args = parser.parse_args()

    if args.live and not args.url:
        parser.error("--url is required with --live")
    if not args.live and not args.video:
        parser.error("--video is required for offline mode")

    cmd = build_pipeline_cmd(args)
    Supervisor(cmd).run()
