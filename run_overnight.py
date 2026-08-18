"""Overnight pipeline: JSON generation + annotated video re-render.

Runs both steps sequentially, logging progress and timestamps.

Usage:
  python run_overnight.py --video recording.avi --config config/belt_config_top.json --speed 2.0
"""
import argparse
import subprocess
import sys
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
LOG = "output/pipeline_log.txt"
os.makedirs("output", exist_ok=True)


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (ts, msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run_step(label, cmd):
    log("START: %s" % label)
    log("  cmd: %s" % " ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    with open(LOG, "a") as f:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
    proc.wait()
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else "FAILED (exit %d)" % proc.returncode
    log("END: %s — %s in %.0f min" % (label, status, elapsed / 60))
    return proc.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight pipeline")
    parser.add_argument("--video", required=True, help="Video file to process")
    parser.add_argument("--config", required=True, help="Belt config JSON")
    parser.add_argument("--speed", type=float, default=2.0, help="Belt speed")
    args = parser.parse_args()

    log("=" * 60)
    log("OVERNIGHT PIPELINE")
    log("  video:  %s" % args.video)
    log("  config: %s" % args.config)
    log("  speed:  %s" % args.speed)
    log("=" * 60)

    ok = run_step(
        "JSON Generation (generate_summary.py)",
        [sys.executable, "generate_summary.py",
         "--video", args.video,
         "--config", args.config,
         "--speed", str(args.speed)])

    if not ok:
        log("JSON generation failed — skipping video pipeline")
        sys.exit(1)

    log("")
    run_step(
        "Video Pipeline (process_recording.py --full)",
        [sys.executable, "process_recording.py",
         "--video", args.video,
         "--config", args.config,
         "--speed", str(args.speed), "--full"])

    log("")
    log("OVERNIGHT PIPELINE COMPLETE")
