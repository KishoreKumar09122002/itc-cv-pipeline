"""Generate detailed JSON summary from a recorded video.

Runs YOLO inference + zone tracking (no video encoding) to produce
zone_events and per_second data matching the dashboard schema.
Uses threaded frame reader for speed.

Usage:
  python generate_summary.py --video recording.mp4 --config config/belt_config_top.json --speed 2.0
  python generate_summary.py --video recording.mp4 --config config/belt_config_top.json --speed 2.0 --output summary.json
"""
import argparse
import cv2
import numpy as np
import os
import sys
import time
import json
import threading
import queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.setup import compute_transforms, load_config
from src.frame_processor import FrameProcessor, StationMapper
from moving_zone_tracker import MovingZoneTracker, BELT_LENGTH

parser = argparse.ArgumentParser(description="Generate JSON summary from recorded video")
parser.add_argument("--video", required=True, help="Path to recorded video file")
parser.add_argument("--config", required=True, help="Belt config JSON path")
parser.add_argument("--speed", type=float, default=2.0, help="Belt speed (default: 2.0)")
parser.add_argument("--output", default=None, help="Output JSON path (default: auto)")
_args = parser.parse_args()

VIDEO_PATH = _args.video
CONFIG_PATH = _args.config
BELT_SPEED = _args.speed

_weights_dir = "runs/pose/training_data/runs/overhead_pose_v1/weights"
_openvino = os.path.join(_weights_dir, "best_openvino_model")
FINETUNED_MODEL = _openvino if os.path.isdir(_openvino) else os.path.join(_weights_dir, "best.pt")

DOWNSCALE = 0.5
ZONE_WIDTH = 166.67
FRAME_SKIP = 2
MAX_REACH_PX = 60
MOVE_THRESHOLD = 12.0

config = load_config(CONFIG_PATH)
corners = np.array(config["pixel_corners"], dtype=np.float32)
belt_height = config.get("belt_height", 400.0)
n_workers = config.get("n_workers", 4)

cap = cv2.VideoCapture(VIDEO_PATH)
fps_val = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_w = int(full_w * DOWNSCALE)
out_h = int(full_h * DOWNSCALE)
effective_fps = fps_val / FRAME_SKIP
duration_sec = int(total_frames / fps_val)
cap.release()

scale_corners = corners * DOWNSCALE
p2b_s, b2p_s = compute_transforms(scale_corners, belt_height=belt_height)
belt_poly = scale_corners.reshape((-1, 1, 2)).astype(np.float32)
mid_y = belt_height / 2.0

processor = FrameProcessor(model_path=FINETUNED_MODEL, confidence_threshold=0.3)
processor.enable_tracking()
station_mapper = StationMapper(belt_height, BELT_LENGTH, n_workers)

tracker = MovingZoneTracker(
    belt_speed=BELT_SPEED * FRAME_SKIP,
    zone_width=ZONE_WIDTH,
    belt_height=belt_height,
    split_zones=True,
)

os.makedirs("output/moving_zones", exist_ok=True)

# --- Threaded frame reader ---
QUEUE_SIZE = 32
frame_queue = queue.Queue(maxsize=QUEUE_SIZE)
reader_done = threading.Event()


def frame_reader():
    local_cap = cv2.VideoCapture(VIDEO_PATH)
    count = 0
    while True:
        ret, frame = local_cap.read()
        if not ret:
            break
        count += 1
        if count % FRAME_SKIP != 0:
            continue
        small = cv2.resize(frame, (out_w, out_h))
        frame_queue.put((count, small))
    local_cap.release()
    reader_done.set()


reader_thread = threading.Thread(target=frame_reader, daemon=True)

# --- Per-second tracking state ---
prev_wrist_pos = {}
current_second = -1
sec_data = None
per_second_log = []
cumulative_picks = {}
cumulative_active = {}
cumulative_detected = {}
cumulative_zones_done = 0
cumulative_zones_picked = 0
cumulative_zones_missed = 0
cumulative_zones_picked_L = 0
cumulative_zones_picked_R = 0
cumulative_zones_picked_both = 0
prev_completed_count = 0

est_frames = total_frames // FRAME_SKIP

print("=" * 70)
print("  JSON SUMMARY GENERATION — %s" % os.path.basename(VIDEO_PATH))
print("=" * 70)
print("  Model: %s" % FINETUNED_MODEL)
print("  Source: %dx%d @ %.0ffps, %d frames (%.1f hrs)" % (
    full_w, full_h, fps_val, total_frames, duration_sec / 3600))
print("  Processing: %dx%d, FRAME_SKIP=%d, %d workers" % (
    out_w, out_h, FRAME_SKIP, n_workers))
print("  Est. frames to process: %d" % est_frames)
print("=" * 70)
sys.stdout.flush()

processed = 0
t_start = time.time()
reader_thread.start()

while True:
    try:
        item = frame_queue.get(timeout=5.0)
    except queue.Empty:
        if reader_done.is_set() and frame_queue.empty():
            break
        continue

    frames_read, small = item
    ts = frames_read / fps_val
    this_second = int(ts)

    # --- Worker detection ---
    workers = processor.get_worker_wrists(small, fingertip_extend=0.3)
    near_belt = processor.filter_belt_proximity(
        workers, scale_corners, p2b_s, belt_height, n_workers=n_workers)
    near_belt = station_mapper.map(near_belt)

    per_worker_wrists = []
    for w in near_belt:
        wid = w["worker_id"]
        w_side = "L" if w.get("_belt_y", 0) < mid_y else "R"
        pw = {"worker_id": wid, "left_pixel": None,
              "right_pixel": None, "worker_side": w_side}

        for hand_key, pixel_key in [("left", "left_pixel"),
                                    ("right", "right_pixel")]:
            h = w[hand_key]
            if h is None:
                continue

            wx, wy = float(h["x"]), float(h["y"])
            tx, ty = float(h["tip_x"]), float(h["tip_y"])
            arm_dx, arm_dy = tx - wx, ty - wy
            arm_len = (arm_dx**2 + arm_dy**2)**0.5

            hkey = (wid, hand_key)
            hand_moved = False
            if hkey in prev_wrist_pos:
                ppx, ppy = prev_wrist_pos[hkey]
                move_dist = ((wx - ppx)**2 + (wy - ppy)**2)**0.5
                hand_moved = move_dist > MOVE_THRESHOLD
            prev_wrist_pos[hkey] = (wx, wy)

            h["_moved"] = hand_moved

            if cv2.pointPolygonTest(belt_poly, (tx, ty), False) >= 0:
                pw[pixel_key] = (int(tx), int(ty))

        per_worker_wrists.append(pw)

    # --- Tracker update ---
    frame_info = tracker.update(b2p_s, per_worker_wrists)

    # --- Per-frame activity tracking ---
    any_worker_in_zone = False
    worker_active_this_frame = {}
    worker_present_this_frame = {}

    for w in near_belt:
        wid = w["worker_id"]
        cumulative_detected[wid] = cumulative_detected.get(wid, 0) + 1
        worker_present_this_frame[wid] = True

        is_active = False
        for hand in ["left", "right"]:
            h = w[hand]
            if h and h.get("_moved"):
                is_active = True
                break

        if is_active:
            cumulative_active[wid] = cumulative_active.get(wid, 0) + 1
        worker_active_this_frame[wid] = is_active

        in_zone_now = any(
            v["status"].startswith("IN") and ("W%d" % wid) in k
            for k, v in frame_info.items()
        )
        if in_zone_now:
            any_worker_in_zone = True
            cumulative_picks[wid] = cumulative_picks.get(wid, 0) + 1

    # --- Zone completion tracking ---
    new_completed = len(tracker.completed_zones)
    if new_completed > prev_completed_count:
        for z in tracker.completed_zones[prev_completed_count:]:
            cumulative_zones_done += 1
            if z.picked:
                cumulative_zones_picked += 1
            else:
                cumulative_zones_missed += 1
            if z.picked_L:
                cumulative_zones_picked_L += 1
            if z.picked_R:
                cumulative_zones_picked_R += 1
            if z.picked_L and z.picked_R:
                cumulative_zones_picked_both += 1
        prev_completed_count = new_completed

    # --- Per-second snapshot ---
    if this_second > current_second:
        if sec_data is not None:
            per_second_log.append(sec_data)
        sec_data = {"t": this_second}
        for wid in range(n_workers):
            det = cumulative_detected.get(wid, 0)
            act = cumulative_active.get(wid, 0)
            sec_data["w%d_present" % wid] = False
            sec_data["w%d_active" % wid] = False
            sec_data["w%d_picks" % wid] = cumulative_picks.get(wid, 0)
            sec_data["w%d_active_pct" % wid] = (
                round(act / det * 100, 1) if det > 0 else 0)
        sec_data["any_picking"] = False
        sec_data["zones_done"] = cumulative_zones_done
        sec_data["zones_picked"] = cumulative_zones_picked
        sec_data["zones_missed"] = cumulative_zones_missed
        sec_data["zones_picked_L"] = cumulative_zones_picked_L
        sec_data["zones_picked_R"] = cumulative_zones_picked_R
        sec_data["zones_picked_both"] = cumulative_zones_picked_both
        sec_data["coverage_pct"] = (
            round(cumulative_zones_picked / cumulative_zones_done * 100, 1)
            if cumulative_zones_done > 0 else 0)
        sec_data["coverage_L_pct"] = (
            round(cumulative_zones_picked_L / cumulative_zones_done * 100, 1)
            if cumulative_zones_done > 0 else 0)
        sec_data["coverage_R_pct"] = (
            round(cumulative_zones_picked_R / cumulative_zones_done * 100, 1)
            if cumulative_zones_done > 0 else 0)
        sec_data["full_coverage_pct"] = (
            round(cumulative_zones_picked_both / cumulative_zones_done * 100, 1)
            if cumulative_zones_done > 0 else 0)
        sec_data["gap_active"] = False
        current_second = this_second

    if sec_data is not None:
        for wid in range(n_workers):
            if worker_present_this_frame.get(wid, False):
                sec_data["w%d_present" % wid] = True
            if worker_active_this_frame.get(wid, False):
                sec_data["w%d_active" % wid] = True
        if any_worker_in_zone:
            sec_data["any_picking"] = True

    processed += 1

    # --- Progress reporting ---
    if processed % 1000 == 0:
        elapsed = time.time() - t_start
        fps_proc = processed / elapsed
        remaining = (est_frames - processed) / fps_proc
        pct = 100 * processed / est_frames
        hours = int(ts // 3600)
        mins = int((ts % 3600) // 60)
        secs = int(ts % 60)
        q_size = frame_queue.qsize()
        print("  [%.1f%%] %d:%02d:%02d | FR %d/%d | %.1f fps | ~%.0f min left "
              "| Q:%d | L:%.0f%% R:%.0f%% | W: %s" % (
                  pct, hours, mins, secs, processed, est_frames,
                  fps_proc, remaining / 60, q_size,
                  sec_data.get("coverage_L_pct", 0) if sec_data else 0,
                  sec_data.get("coverage_R_pct", 0) if sec_data else 0,
                  ", ".join("W%d=%d" % (wid, cumulative_detected.get(wid, 0))
                            for wid in sorted(cumulative_detected.keys()))))
        sys.stdout.flush()

# Flush last second
if sec_data is not None:
    per_second_log.append(sec_data)

reader_thread.join(timeout=5)

elapsed = time.time() - t_start
report = tracker.get_report()

# --- Build zone_events ---
zone_events = []
for z in tracker.completed_zones:
    zone_events.append({
        "batch_id": z.batch_id,
        "spawn_time": round(z.born_frame / effective_fps, 1),
        "retire_time": round((z.born_frame + z.frames_alive) / effective_fps, 1),
        "picked": z.picked,
        "picked_L": z.picked_L,
        "picked_R": z.picked_R,
        "picked_by": sorted(z.picked_by) if z.picked else [],
        "picked_by_L": sorted(z.picked_by_L) if z.picked_L else [],
        "picked_by_R": sorted(z.picked_by_R) if z.picked_R else [],
        "first_pick_time": round(z.first_pick_frame / effective_fps, 1) if z.first_pick_frame >= 0 else None,
        "first_pick_worker": z.first_pick_worker if z.first_pick_worker >= 0 else None,
    })

# --- Build summary JSON ---
summary = {
    "video": os.path.basename(VIDEO_PATH),
    "angle": "top",
    "duration_seconds": duration_sec,
    "fps": fps_val,
    "effective_fps": effective_fps,
    "belt_speed": BELT_SPEED,
    "zone_width": ZONE_WIDTH,
    "belt_height": belt_height,
    "belt_traverse_time_seconds": int(BELT_LENGTH / (BELT_SPEED * fps_val / FRAME_SKIP)),
    "split_zones": True,
    "n_workers": n_workers,
    "zone_events": zone_events,
    "per_second": per_second_log,
}

if _args.output:
    json_path = _args.output
else:
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    json_path = "output/moving_zones/%s_summary.json" % video_name
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

# --- Print results ---
print()
print("=" * 70)
print("  SUMMARY GENERATION — RESULTS")
print("=" * 70)
print("  Processed %d frames in %.0f min (%.1f fps)" % (
    processed, elapsed / 60, processed / elapsed))
print("  Completed zones: %d" % report["completed_batches"])
print("  Picked: %d (%.1f%%)" % (report["picked"], report["pickup_rate_percent"]))
print("  Missed: %d" % report["missed"])
print("  L-side: %.1f%% | R-side: %.1f%% | Both: %.1f%%" % (
    report.get("coverage_L_pct", 0),
    report.get("coverage_R_pct", 0),
    report.get("full_coverage_pct", 0)))
print()
for wid in range(n_workers):
    det = cumulative_detected.get(wid, 0)
    act = cumulative_active.get(wid, 0)
    pct = round(act / det * 100, 1) if det > 0 else 0
    picks = cumulative_picks.get(wid, 0)
    side = "Near" if wid < n_workers // 2 else "Far"
    print("  Worker %d (%s): active %.1f%% | %d pick frames | %d detected" % (
        wid, side, pct, picks, det))
print()
print("  per_second entries: %d" % len(per_second_log))
print("  zone_events entries: %d" % len(zone_events))
print("  JSON: %s" % json_path)
print("=" * 70)
sys.stdout.flush()
