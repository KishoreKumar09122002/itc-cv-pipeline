"""Process a recorded video — annotated output + JSON summary.

Workers on both sides of the belt are tracked. Each zone has L (near-side)
and R (far-side) sub-zones. Coverage requires both sides to be inspected.

Usage:
  python process_recording.py --video recording.mp4 --config config/belt_config_top.json --speed 2.0 --smoke
  python process_recording.py --video recording.mp4 --config config/belt_config_top.json --speed 2.0 --full
"""

import argparse
import cv2
import numpy as np
import os
import subprocess
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.setup import compute_transforms, load_config, belt_to_pixel_quad
from src.frame_processor import FrameProcessor, StationMapper
from moving_zone_tracker import MovingZoneTracker, BELT_LENGTH

ZONE_WIDTH = 166.67
MOVE_THRESHOLD = 12.0
DOWNSCALE = 0.5
MAX_REACH_PX = 60


def run(video_path, config_path, belt_speed, mode="smoke",
        output_dir="output/moving_zones", start_sec=0):
    config = load_config(config_path)
    corners = np.array(config["pixel_corners"], dtype=np.float32)
    belt_height = config.get("belt_height", 400.0)
    n_workers = config.get("n_workers", 4)

    p2b, b2p = compute_transforms(corners, belt_height=belt_height)

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "runs", "pose", "training_data", "runs",
                               "overhead_pose_v1", "weights")
    openvino_dir = os.path.join(weights_dir, "best_openvino_model")
    pt_path = os.path.join(weights_dir, "best.pt")
    if os.path.isdir(openvino_dir):
        model = openvino_dir
    elif os.path.exists(pt_path):
        model = pt_path
    else:
        model = "yolov8n-pose.pt"
    processor = FrameProcessor(model_path=model,
                               confidence_threshold=0.3)
    processor.enable_tracking()
    station_mapper = StationMapper(belt_height, BELT_LENGTH, n_workers)

    cap = cv2.VideoCapture(video_path)
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_skip = 2 if mode == "full" else 1
    tracker = MovingZoneTracker(
        belt_speed=belt_speed * frame_skip,
        zone_width=ZONE_WIDTH,
        belt_height=belt_height,
        split_zones=True,
    )
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    effective_fps = fps_val / frame_skip

    out_w = int(width * DOWNSCALE)
    out_h = int(height * DOWNSCALE)
    scale_corners = corners * DOWNSCALE
    p2b_s, b2p_s = compute_transforms(scale_corners, belt_height=belt_height)

    belt_poly = scale_corners.reshape((-1, 1, 2)).astype(np.float32)
    belt_center = np.mean(scale_corners, axis=0)
    near_edge_vec = scale_corners[1] - scale_corners[0]
    near_normal = np.array([-near_edge_vec[1], near_edge_vec[0]], dtype=np.float32)
    if np.dot(near_normal, belt_center - scale_corners[0]) < 0:
        near_normal = -near_normal
    near_normal = near_normal / np.linalg.norm(near_normal)
    far_edge_vec = scale_corners[2] - scale_corners[3]
    far_normal = np.array([-far_edge_vec[1], far_edge_vec[0]], dtype=np.float32)
    if np.dot(far_normal, belt_center - scale_corners[3]) < 0:
        far_normal = -far_normal
    far_normal = far_normal / np.linalg.norm(far_normal)
    mid_y = belt_height / 2.0

    if start_sec > 0:
        start_frame = int(start_sec * fps_val)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print("  Starting from %ds (frame %d)" % (start_sec, start_frame))

    os.makedirs(output_dir, exist_ok=True)

    writer = None
    ffmpeg_proc = None
    if mode == "full":
        out_path = os.path.join(output_dir, "%s_moving.mp4" % video_name)
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_cmd = [
                ffmpeg_exe, "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", "%dx%d" % (out_w, out_h),
                "-pix_fmt", "bgr24", "-r", str(effective_fps),
                "-i", "-",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                out_path,
            ]
            ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, effective_fps,
                                     (out_w, out_h))

    print("=" * 60)
    print("  TOP ANGLE — %s (%s)" % (video_name, mode.upper()))
    print("  Speed: %.3f/frame | Zones: %d | Width: %.1f | Split: L/R" % (
        belt_speed, int(1000 / ZONE_WIDTH), ZONE_WIDTH))
    print("  Downscale: %.0f%% (%dx%d)" % (DOWNSCALE * 100, out_w, out_h))
    print("=" * 60)

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

    frame_num = 0
    processed = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "full" and frame_num % frame_skip != 0:
            frame_num += 1
            continue

        small = cv2.resize(frame, (out_w, out_h))

        workers = processor.get_worker_wrists(small, fingertip_extend=0.3)
        near_belt = processor.filter_belt_proximity(
            workers, scale_corners, p2b_s, belt_height,
            n_workers=n_workers)
        near_belt = station_mapper.map(near_belt)

        per_worker_wrists = []
        for w in near_belt:
            w_side = "L" if w.get("_belt_y", 0) < mid_y else "R"
            pw = {"worker_id": w["worker_id"],
                  "left_pixel": None, "right_pixel": None,
                  "worker_side": w_side}
            for hand_key, pixel_key in [("left", "left_pixel"),
                                        ("right", "right_pixel")]:
                h = w[hand_key]
                if h is None:
                    continue
                wx, wy = float(h["x"]), float(h["y"])
                tx, ty = float(h["tip_x"]), float(h["tip_y"])
                arm_dx, arm_dy = tx - wx, ty - wy
                arm_len = (arm_dx**2 + arm_dy**2)**0.5

                hkey = (w["worker_id"], hand_key)
                hand_moved = False
                if hkey in prev_wrist_pos:
                    ppx, ppy = prev_wrist_pos[hkey]
                    move_dist = ((wx - ppx)**2 + (wy - ppy)**2)**0.5
                    hand_moved = move_dist > MOVE_THRESHOLD
                prev_wrist_pos[hkey] = (wx, wy)

                detected = False
                method = None

                if cv2.pointPolygonTest(belt_poly, (tx, ty), False) >= 0:
                    pw[pixel_key] = (int(tx), int(ty))
                    detected = True
                    method = "primary"

                h["_detected"] = detected
                h["_method"] = method
                h["_moved"] = hand_moved
            per_worker_wrists.append(pw)

        frame_info = tracker.update(b2p_s, per_worker_wrists)

        ts = frame_num / fps_val
        this_second = int(ts)

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

        # --- Annotate frame (grid-style with L/R sub-zones) ---
        vis = small.copy()

        for z in tracker.active_zones:
            if z.x < 0 or z.x + z.width > BELT_LENGTH:
                continue

            pts_L = np.array([[[z.x, 0], [z.x + z.width, 0],
                               [z.x + z.width, mid_y], [z.x, mid_y]]],
                             dtype=np.float32)
            pts_R = np.array([[[z.x, mid_y], [z.x + z.width, mid_y],
                               [z.x + z.width, belt_height],
                               [z.x, belt_height]]], dtype=np.float32)
            quad_L = cv2.perspectiveTransform(pts_L, b2p_s)[0].astype(np.int32)
            quad_R = cv2.perspectiveTransform(pts_R, b2p_s)[0].astype(np.int32)

            col_L = (0, 200, 0) if z.picked_L else (0, 0, 160)
            col_R = (0, 200, 0) if z.picked_R else (0, 0, 160)

            overlay = vis.copy()
            cv2.fillPoly(overlay, [quad_L], col_L)
            cv2.fillPoly(overlay, [quad_R], col_R)
            cv2.addWeighted(overlay, 0.30, vis, 0.70, 0, vis)

            quad_full = belt_to_pixel_quad(z.x, z.width, b2p_s,
                                           belt_height=belt_height)
            cv2.polylines(vis, [quad_full], True, (0, 0, 220), 2)
            pt_m1 = cv2.perspectiveTransform(
                np.array([[[z.x, mid_y]]], dtype=np.float32), b2p_s)[0][0]
            pt_m2 = cv2.perspectiveTransform(
                np.array([[[z.x + z.width, mid_y]]], dtype=np.float32),
                b2p_s)[0][0]
            cv2.line(vis, (int(pt_m1[0]), int(pt_m1[1])),
                     (int(pt_m2[0]), int(pt_m2[1])), (255, 255, 255), 3)

        for w in near_belt:
            wid = w["worker_id"]
            w_side = "L" if w.get("_belt_y", 0) < mid_y else "R"
            for hand, side in [("left", "L"), ("right", "R")]:
                h = w[hand]
                if h is None:
                    continue
                wx, wy = int(h["x"]), int(h["y"])
                method = h.get("_method")
                if method == "primary":
                    tx, ty = int(h["tip_x"]), int(h["tip_y"])
                    wkey = "W%d-%s" % (wid, side)
                    status = frame_info.get(wkey, {}).get("status", "N/A")
                    color = (0, 255, 0) if status.startswith("IN") else (0, 0, 255)
                    cv2.circle(vis, (wx, wy), 4, (0, 0, 255), -1)
                    cv2.circle(vis, (tx, ty), 6, color, -1)
                    cv2.line(vis, (wx, wy), (tx, ty), (0, 255, 255), 1)
                elif method == "secondary":
                    px, py = int(h["_reach_tip_x"]), int(h["_reach_tip_y"])
                    wkey = "W%d-%s" % (wid, side)
                    status = frame_info.get(wkey, {}).get("status", "N/A")
                    color = (0, 255, 0) if status.startswith("IN") else (0, 0, 255)
                    cv2.circle(vis, (wx, wy), 4, (0, 0, 255), -1)
                    cv2.circle(vis, (px, py), 6, color, -1)
                    cv2.line(vis, (wx, wy), (px, py), (255, 200, 0), 2)
                else:
                    cv2.circle(vis, (wx, wy), 3, (128, 128, 128), -1)

            if w.get("shoulder_mid"):
                sx, sy = int(w["shoulder_mid"][0]), int(w["shoulder_mid"][1])
                cv2.putText(vis, "W%d(%s)" % (wid, w_side),
                            (sx + 8, sy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        report = tracker.get_report()
        cv2.putText(vis,
                    "F%d (%.0fs) | Done:%d Active:%d | Any:%.1f%% L:%.1f%% R:%.1f%% Both:%.1f%%" % (
                        frame_num, ts,
                        report["completed_batches"], report["active_batches"],
                        report["pickup_rate_percent"],
                        report.get("coverage_L_pct", 0),
                        report.get("coverage_R_pct", 0),
                        report.get("full_coverage_pct", 0)),
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        if ffmpeg_proc:
            ffmpeg_proc.stdin.write(vis.tobytes())
        elif writer:
            writer.write(vis)

        processed += 1
        if mode == "full" and processed % 500 == 0:
            elapsed = time.time() - t_start
            fps_proc = processed / elapsed
            remaining = (total / frame_skip - processed) / fps_proc
            print("  Frame %d/%d | %.1f fps | ~%.0fs remaining" % (
                frame_num, total, fps_proc, remaining))

        if mode == "smoke" and processed >= 750:
            print("  Smoke test: processed %d frames, stopping." % processed)
            break

        frame_num += 1

    if sec_data is not None:
        per_second_log.append(sec_data)

    cap.release()
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    elif writer:
        writer.release()

    elapsed = time.time() - t_start
    report = tracker.get_report()
    duration_sec = int(frame_num / fps_val)

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

    summary = {
        "video": os.path.basename(video_path),
        "angle": "top",
        "duration_seconds": duration_sec,
        "fps": fps_val,
        "effective_fps": effective_fps,
        "belt_speed": belt_speed,
        "zone_width": ZONE_WIDTH,
        "belt_height": belt_height,
        "split_zones": True,
        "n_workers": n_workers,
        "zone_events": zone_events,
        "per_second": per_second_log,
    }

    json_path = os.path.join(output_dir, "%s_moving_summary.json" % video_name)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("  RESULTS — %s (TOP ANGLE)" % video_name)
    print("=" * 60)
    print("  Coverage: %d/%d batches (%.1f%%)" % (
        report["picked"], report["completed_batches"],
        report["pickup_rate_percent"]))
    print("  L-side: %.1f%% | R-side: %.1f%% | Both: %.1f%%" % (
        report.get("coverage_L_pct", 0),
        report.get("coverage_R_pct", 0),
        report.get("full_coverage_pct", 0)))
    for wid in range(n_workers):
        det = cumulative_detected.get(wid, 0)
        act = cumulative_active.get(wid, 0)
        pct = round(act / det * 100, 1) if det > 0 else 0
        picks = cumulative_picks.get(wid, 0)
        side = "Near" if wid < n_workers // 2 else "Far"
        print("  Worker %d (%s): active %.1f%% | %d pick frames" % (
            wid, side, pct, picks))
    print("  Processed %d frames in %.1fs" % (processed, elapsed))
    print("=" * 60)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process recorded video with annotation + JSON")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--start", type=int, default=0,
                        help="Start offset in seconds")
    args = parser.parse_args()

    mode = "smoke" if args.smoke else ("full" if args.full else "smoke")
    run(args.video, args.config, args.speed, mode=mode, start_sec=args.start)
