"""Live pipeline — production-grade frame-by-frame processing.

Reads from an RTSP camera stream or a local video file,
runs YOLO + ByteTrack + StationMapper + MovingZoneTracker,
and writes results to SQLite in real time.

Usage:
  Live RTSP:
    python live_pipeline.py --url rtsp://admin:pass@192.168.1.100:554/stream --config config/belt_config_top.json --speed 2.0
    python live_pipeline.py --url rtsp://... --config ... --speed 2.0 --db output/live.db --record --stream

  Local video (demo):
    python live_pipeline.py --video recording.avi --config config/belt_config_top.json --speed 2.0 --stream
"""
import argparse
import cv2
import numpy as np
import os
import subprocess
import sys
import time
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn, TCPServer
import http.server


class ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.setup import compute_transforms, load_config, belt_to_pixel_quad
from src.frame_processor import FrameProcessor, StationMapper
from moving_zone_tracker import MovingZoneTracker, BELT_LENGTH

ZONE_WIDTH = 166.67
MOVE_THRESHOLD = 12.0
DOWNSCALE = 0.5
FRAME_SKIP = 2
RECONNECT_DELAY = 5
MAX_RECONNECT_ATTEMPTS = 50
RECORDING_DIR = "output/live_recordings"
ANNOTATED_DIR = "output/live_annotated"


# --------------- MJPEG Stream Server ---------------

class MJPEGHandler(BaseHTTPRequestHandler):
    latest_frame = None
    frame_lock = threading.Lock()

    def do_GET(self):
        if self.path == "/":
            page = (
                '<!doctype html><html><head><title>ITC Belt Monitor</title>'
                '<style>*{margin:0;padding:0}body{background:#000;'
                'display:flex;align-items:center;justify-content:center;'
                'height:100vh}img{width:100%;height:100%;object-fit:contain}'
                '</style></head><body>'
                '<img src="/live" alt="Live Feed"/></body></html>'
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(page.encode())
            return
        if self.path != "/live":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=--frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                with MJPEGHandler.frame_lock:
                    jpeg = MJPEGHandler.latest_frame
                if jpeg is None:
                    time.sleep(0.04)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(("Content-Length: %d\r\n\r\n" % len(jpeg)).encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        pass


def start_mjpeg_server(port):
    server = ThreadingHTTPServer(("0.0.0.0", port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def update_mjpeg_frame(frame):
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with MJPEGHandler.frame_lock:
        MJPEGHandler.latest_frame = jpeg.tobytes()


def start_annotated_writer(ffmpeg_exe, out_w, out_h, fps, output_path):
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "%dx%d" % (out_w, out_h),
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+frag_keyframe+empty_moov",
        "-an", output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------- Database ---------------

def init_db(db_path):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            camera_url TEXT NOT NULL,
            config_path TEXT NOT NULL,
            belt_speed REAL,
            zone_width REAL,
            n_workers INTEGER,
            belt_height REAL,
            fps REAL,
            status TEXT DEFAULT 'running',
            recording_path TEXT,
            annotated_path TEXT
        );
        CREATE TABLE IF NOT EXISTS per_second (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            t INTEGER NOT NULL,
            wall_time TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS zone_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            batch_id INTEGER,
            spawn_time REAL,
            retire_time REAL,
            picked INTEGER,
            picked_L INTEGER,
            picked_R INTEGER,
            picked_by TEXT,
            picked_by_L TEXT,
            picked_by_R TEXT,
            first_pick_time REAL,
            first_pick_worker INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_ps_session ON per_second(session_id, t);
        CREATE INDEX IF NOT EXISTS idx_ze_session ON zone_events(session_id);
    """)
    # Migrate: add columns if missing (existing DBs)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
    if "recording_path" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN recording_path TEXT")
    if "annotated_path" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN annotated_path TEXT")
    return conn


def create_session(conn, camera_url, config_path, belt_speed, n_workers,
                   belt_height, fps, recording_path=None, annotated_path=None):
    cur = conn.execute(
        """INSERT INTO sessions
           (start_time, camera_url, config_path, belt_speed, zone_width,
            n_workers, belt_height, fps, status, recording_path, annotated_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)""",
        (time.strftime("%Y-%m-%d %H:%M:%S"), camera_url, config_path,
         belt_speed, ZONE_WIDTH, n_workers, belt_height, fps,
         recording_path, annotated_path))
    return cur.lastrowid


def write_per_second(conn, session_id, t, data):
    conn.execute(
        "INSERT INTO per_second (session_id, t, wall_time, data) VALUES (?, ?, ?, ?)",
        (session_id, t, time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(data)))


def write_zone_event(conn, session_id, z, effective_fps):
    conn.execute(
        """INSERT INTO zone_events
           (session_id, batch_id, spawn_time, retire_time, picked,
            picked_L, picked_R, picked_by, picked_by_L, picked_by_R,
            first_pick_time, first_pick_worker)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, z.batch_id,
         round(z.born_frame / effective_fps, 1),
         round((z.born_frame + z.frames_alive) / effective_fps, 1),
         int(z.picked), int(z.picked_L), int(z.picked_R),
         json.dumps(sorted(z.picked_by) if z.picked else []),
         json.dumps(sorted(z.picked_by_L) if z.picked_L else []),
         json.dumps(sorted(z.picked_by_R) if z.picked_R else []),
         round(z.first_pick_frame / effective_fps, 1) if z.first_pick_frame >= 0 else None,
         z.first_pick_worker if z.first_pick_worker >= 0 else None))


def end_session(conn, session_id):
    conn.execute(
        "UPDATE sessions SET status = 'completed' WHERE id = ?",
        (session_id,))


# --------------- RTSP Connection ---------------

class ThreadedCapture:
    """Continuously reads frames in a background thread so the HEVC decoder
    sees every frame (preserving the reference-frame chain).  The main loop
    calls read() to grab the latest fully-decoded frame."""

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._stopped = False
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()

    def _grab(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
            if not ret:
                break

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        self._stopped = True
        self._thread.join(timeout=3)
        self._cap.release()

    def isOpened(self):
        return self._cap.isOpened() and not self._stopped


def connect_stream(url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None, 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    threaded = ThreadedCapture(cap)
    return threaded, fps, w, h


def connect_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h


def reconnect(url, attempt=0):
    while attempt < MAX_RECONNECT_ATTEMPTS:
        attempt += 1
        print("  [RECONNECT] Attempt %d/%d — waiting %ds..." % (
            attempt, MAX_RECONNECT_ATTEMPTS, RECONNECT_DELAY), flush=True)
        time.sleep(RECONNECT_DELAY)
        cap, fps, w, h = connect_stream(url)
        if cap is not None:
            print("  [RECONNECT] Success!", flush=True)
            return cap, fps, w, h
    return None, 0, 0, 0


# --------------- Main Pipeline ---------------

def run(args):
    config = load_config(args.config)
    corners = np.array(config["pixel_corners"], dtype=np.float32)
    belt_height = config.get("belt_height", 400.0)
    n_workers = config.get("n_workers", 4)
    belt_speed = args.speed
    frame_skip = args.frame_skip

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "runs", "pose", "training_data", "runs",
                               "overhead_pose_v1", "weights")
    if args.model:
        model_path = args.model
    else:
        openvino_dir = os.path.join(weights_dir, "best_openvino_model")
        pt_path = os.path.join(weights_dir, "best.pt")
        if os.path.isdir(openvino_dir):
            model_path = openvino_dir
        elif os.path.exists(pt_path):
            model_path = pt_path
        else:
            model_path = "yolov8n-pose.pt"

    processor = FrameProcessor(model_path=model_path, confidence_threshold=0.3)
    processor.enable_tracking()
    station_mapper = StationMapper(belt_height, BELT_LENGTH, n_workers)

    # Connect to source
    is_video = bool(getattr(args, "video", None))
    source = args.video if is_video else args.url
    print("=" * 60)
    print("  %s PIPELINE — %s" % (
        "VIDEO" if is_video else "LIVE",
        "Loading %s" % source if is_video else "Connecting to %s" % source))
    print("=" * 60, flush=True)

    if is_video:
        cap, fps_val, full_w, full_h = connect_video(source)
    else:
        cap, fps_val, full_w, full_h = connect_stream(source)
    if cap is None:
        print("ERROR: Cannot open %s" % source)
        sys.exit(1)

    out_w = int(full_w * DOWNSCALE)
    out_h = int(full_h * DOWNSCALE)
    scale_corners = corners * DOWNSCALE
    p2b_s, b2p_s = compute_transforms(scale_corners, belt_height=belt_height)
    belt_poly = scale_corners.reshape((-1, 1, 2)).astype(np.float32)
    mid_y = belt_height / 2.0


    tracker = MovingZoneTracker(
        belt_speed=belt_speed * frame_skip,
        zone_width=ZONE_WIDTH,
        belt_height=belt_height,
        split_zones=True,
    )

    # Recording (separate ffmpeg process — RTSP mode only)
    recorder = None
    recording_path = None
    if args.record and not is_video:
        os.makedirs(RECORDING_DIR, exist_ok=True)
        recording_path = os.path.join(
            RECORDING_DIR,
            "session_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = "ffmpeg"
        try:
            recorder = subprocess.Popen(
                [ffmpeg_exe, "-y", "-rtsp_transport", "tcp",
                 "-i", args.url, "-an", "-c:v", "copy",
                 "-movflags", "+frag_keyframe+empty_moov", recording_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            print("  Recording: %s (PID %d)" % (recording_path, recorder.pid),
                  flush=True)
        except FileNotFoundError:
            print("  [WARN] ffmpeg not found — recording disabled", flush=True)
            recorder = None
            recording_path = None

    # MJPEG live stream server
    mjpeg_server = None
    stream_port = getattr(args, "stream_port", 8503)
    if args.stream:
        try:
            mjpeg_server = start_mjpeg_server(stream_port)
            print("  MJPEG stream: http://0.0.0.0:%d/live" % stream_port,
                  flush=True)
        except OSError as e:
            print("  [WARN] MJPEG server failed to start: %s" % e, flush=True)

    # Annotated video writer (ffmpeg pipe)
    ann_writer = None
    annotated_path = None
    if args.stream:
        os.makedirs(ANNOTATED_DIR, exist_ok=True)
        annotated_path = os.path.join(
            ANNOTATED_DIR,
            "session_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))
        try:
            import imageio_ffmpeg
            ann_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ann_ffmpeg = "ffmpeg"
        try:
            ann_writer = start_annotated_writer(
                ann_ffmpeg, out_w, out_h, int(fps_val), annotated_path)
            print("  Annotated video: %s (PID %d)" % (
                annotated_path, ann_writer.pid), flush=True)
        except FileNotFoundError:
            print("  [WARN] ffmpeg not found — annotated video disabled",
                  flush=True)
            ann_writer = None
            annotated_path = None

    # Database
    conn = init_db(args.db)
    session_id = create_session(
        conn, source, args.config, belt_speed, n_workers,
        belt_height, fps_val, recording_path=recording_path,
        annotated_path=annotated_path)

    effective_fps = fps_val / frame_skip
    print("  Stream: %dx%d @ %.0f fps → %dx%d (process every %d%s frame = %.1f fps)" % (
        full_w, full_h, fps_val, out_w, out_h,
        frame_skip, "nd" if frame_skip == 2 else "rd" if frame_skip == 3 else "th",
        effective_fps))
    print("  Model: %s" % model_path)
    print("  Workers: %d | Belt speed: %.1f | Zones: %.0f" % (
        n_workers, belt_speed, ZONE_WIDTH))
    print("  Database: %s (session %d)" % (args.db, session_id))
    if recording_path:
        print("  Recording (raw): %s" % recording_path)
    if annotated_path:
        print("  Recording (annotated): %s" % annotated_path)
    if mjpeg_server:
        print("  Live view: http://localhost:%d/live" % stream_port)
    print("  Press Ctrl+C to stop")
    print("=" * 60, flush=True)

    # Processing state
    prev_wrist_pos = {}
    current_second = -1
    sec_data = None
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

    processed = 0
    frame_count = 0
    t_start = time.time()
    stream_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_video:
                    print("  [END] Video file finished.", flush=True)
                    break
                if not cap.isOpened():
                    print("  [WARN] Stream disconnected — attempting reconnect...",
                          flush=True)
                    cap.release()
                    cap, fps_val, _, _ = reconnect(args.url)
                    if cap is None:
                        print("  [ERROR] All reconnect attempts failed. Exiting.",
                              flush=True)
                        break
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            small = cv2.resize(frame, (out_w, out_h))
            if is_video:
                ts = frame_count / fps_val
            else:
                ts = time.time() - stream_start
            this_second = int(ts)

            # --- Detection + Tracking ---
            workers = processor.get_worker_wrists(small, fingertip_extend=0.3)
            near_belt = processor.filter_belt_proximity(
                workers, scale_corners, p2b_s, belt_height,
                n_workers=n_workers)
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

                    hkey = (wid, hand_key)
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

            # --- Zone tracking ---
            frame_info = tracker.update(b2p_s, per_worker_wrists)

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

            # --- Zone completion ---
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
                    write_zone_event(conn, session_id, z, fps_val)
                prev_completed_count = new_completed

            # --- Per-second snapshot ---
            if this_second > current_second:
                if sec_data is not None:
                    write_per_second(conn, session_id, current_second, sec_data)

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

            # --- Annotate + stream + save (full visualization) ---
            if args.stream:
                try:
                    ann = small.copy()

                    for z in tracker.active_zones:
                        if z.x < 0 or z.x + z.width > BELT_LENGTH:
                            continue
                        pts_L = np.array([[[z.x, 0], [z.x + z.width, 0],
                                           [z.x + z.width, mid_y],
                                           [z.x, mid_y]]], dtype=np.float32)
                        pts_R = np.array([[[z.x, mid_y],
                                           [z.x + z.width, mid_y],
                                           [z.x + z.width, belt_height],
                                           [z.x, belt_height]]],
                                         dtype=np.float32)
                        quad_L = cv2.perspectiveTransform(
                            pts_L, b2p_s)[0].astype(np.int32)
                        quad_R = cv2.perspectiveTransform(
                            pts_R, b2p_s)[0].astype(np.int32)
                        col_L = (0, 200, 0) if z.picked_L else (0, 0, 160)
                        col_R = (0, 200, 0) if z.picked_R else (0, 0, 160)
                        overlay = ann.copy()
                        cv2.fillPoly(overlay, [quad_L], col_L)
                        cv2.fillPoly(overlay, [quad_R], col_R)
                        cv2.addWeighted(overlay, 0.30, ann, 0.70, 0, ann)
                        quad_full = belt_to_pixel_quad(
                            z.x, z.width, b2p_s, belt_height=belt_height)
                        cv2.polylines(ann, [quad_full], True, (0, 0, 220), 2)
                        pt_m1 = cv2.perspectiveTransform(
                            np.array([[[z.x, mid_y]]], dtype=np.float32),
                            b2p_s)[0][0]
                        pt_m2 = cv2.perspectiveTransform(
                            np.array([[[z.x + z.width, mid_y]]],
                                     dtype=np.float32), b2p_s)[0][0]
                        cv2.line(ann, (int(pt_m1[0]), int(pt_m1[1])),
                                 (int(pt_m2[0]), int(pt_m2[1])),
                                 (255, 255, 255), 3)

                    for w in near_belt:
                        wid = w["worker_id"]
                        w_side = ("L" if w.get("_belt_y", 0) < mid_y
                                  else "R")
                        for hand, side in [("left", "L"), ("right", "R")]:
                            h = w[hand]
                            if h is None:
                                continue
                            wx, wy = int(h["x"]), int(h["y"])
                            method = h.get("_method")
                            if method == "primary":
                                tx = int(h["tip_x"])
                                ty = int(h["tip_y"])
                                wkey = "W%d-%s" % (wid, side)
                                status = frame_info.get(
                                    wkey, {}).get("status", "N/A")
                                color = ((0, 255, 0)
                                         if status.startswith("IN")
                                         else (0, 0, 255))
                                cv2.circle(ann, (wx, wy), 4,
                                           (0, 0, 255), -1)
                                cv2.circle(ann, (tx, ty), 6, color, -1)
                                cv2.line(ann, (wx, wy), (tx, ty),
                                         (0, 255, 255), 1)
                            else:
                                cv2.circle(ann, (wx, wy), 3,
                                           (128, 128, 128), -1)
                        if w.get("shoulder_mid"):
                            sx = int(w["shoulder_mid"][0])
                            sy = int(w["shoulder_mid"][1])
                            cv2.putText(ann, "W%d(%s)" % (wid, w_side),
                                        (sx + 8, sy - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 255, 0), 1)

                    report = tracker.get_report()
                    cv2.putText(
                        ann,
                        "F%d (%.0fs) | Done:%d Active:%d | "
                        "Any:%.1f%% L:%.1f%% R:%.1f%% Both:%.1f%%" % (
                            processed, ts,
                            report["completed_batches"],
                            report["active_batches"],
                            report["pickup_rate_percent"],
                            report.get("coverage_L_pct", 0),
                            report.get("coverage_R_pct", 0),
                            report.get("full_coverage_pct", 0)),
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 255, 255), 1)

                    update_mjpeg_frame(ann)
                    if ann_writer and ann_writer.poll() is None:
                        try:
                            ann_writer.stdin.write(ann.tobytes())
                        except (BrokenPipeError, OSError):
                            ann_writer = None
                except Exception as e:
                    if processed <= 5 or processed % 500 == 0:
                        print("  [WARN] Annotation error (frame %d): %s"
                              % (processed, e), flush=True)

            # --- Progress (every 5 seconds) ---
            if this_second > 0 and this_second % 5 == 0 and this_second != getattr(run, '_last_report', -1):
                run._last_report = this_second
                elapsed = time.time() - t_start
                fps_proc = processed / elapsed if elapsed > 0 else 0
                mins = this_second // 60
                secs = this_second % 60
                print("  [LIVE %02d:%02d] %d frames | %.1f fps | "
                      "L:%.0f%% R:%.0f%% | Workers: %s" % (
                          mins, secs, processed, fps_proc,
                          sec_data.get("coverage_L_pct", 0) if sec_data else 0,
                          sec_data.get("coverage_R_pct", 0) if sec_data else 0,
                          ", ".join("W%d=%d" % (wid, cumulative_detected.get(wid, 0))
                                   for wid in sorted(cumulative_detected.keys()))),
                      flush=True)

    except KeyboardInterrupt:
        print("\n  [STOP] Ctrl+C received — shutting down gracefully.", flush=True)

    # Flush last second
    if sec_data is not None:
        write_per_second(conn, session_id, current_second, sec_data)

    cap.release()

    # Stop annotated video writer
    if ann_writer and ann_writer.poll() is None:
        print("  [STOP] Stopping annotated writer...", flush=True)
        ann_writer.stdin.close()
        try:
            ann_writer.wait(timeout=15)
        except subprocess.TimeoutExpired:
            ann_writer.kill()
        print("  [STOP] Annotated video saved: %s" % annotated_path, flush=True)

    # Stop MJPEG server
    if mjpeg_server:
        mjpeg_server.shutdown()

    # Stop recorder gracefully
    if recorder and recorder.poll() is None:
        print("  [STOP] Stopping recorder...", flush=True)
        recorder.stdin.write(b"q")
        recorder.stdin.flush()
        try:
            recorder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            recorder.kill()
        print("  [STOP] Recording saved: %s" % recording_path, flush=True)

    end_session(conn, session_id)

    elapsed = time.time() - t_start
    report = tracker.get_report()

    print()
    print("=" * 60)
    print("  LIVE SESSION ENDED — Session %d" % session_id)
    print("=" * 60)
    print("  Duration: %d seconds | Frames: %d | %.1f fps" % (
        int(elapsed), processed, processed / elapsed if elapsed > 0 else 0))
    print("  Zones completed: %d | Picked: %d (%.1f%%)" % (
        report["completed_batches"], report["picked"],
        report["pickup_rate_percent"]))
    print("  L: %.1f%% | R: %.1f%% | Both: %.1f%%" % (
        report.get("coverage_L_pct", 0),
        report.get("coverage_R_pct", 0),
        report.get("full_coverage_pct", 0)))
    for wid in range(n_workers):
        det = cumulative_detected.get(wid, 0)
        act = cumulative_active.get(wid, 0)
        pct = round(act / det * 100, 1) if det > 0 else 0
        side = "Near" if wid < n_workers // 2 else "Far"
        print("  W%d (%s): active %.1f%% | %d picks | %d detected" % (
            wid, side, pct, cumulative_picks.get(wid, 0), det))
    print("  Database: %s" % args.db)
    print("=" * 60, flush=True)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Pipeline")
    parser.add_argument("--url", default=None,
                        help="RTSP stream URL (e.g. rtsp://admin:pass@192.168.1.100:554/stream)")
    parser.add_argument("--video", default=None,
                        help="Local video file path (alternative to --url for demo/offline)")
    parser.add_argument("--config", required=True,
                        help="Belt config JSON path")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Belt speed in pixels/frame (default: 2.0)")
    parser.add_argument("--db", default="output/live.db",
                        help="SQLite database path (default: output/live.db)")
    parser.add_argument("--record", action="store_true",
                        help="Record RTSP stream to video file for gap replay")
    parser.add_argument("--stream", action="store_true",
                        help="Enable MJPEG live stream + annotated video saving")
    parser.add_argument("--stream-port", type=int, default=8503,
                        help="MJPEG stream port (default: 8503)")
    parser.add_argument("--model", default=None,
                        help="Model path (default: auto-detect OpenVINO > best.pt)")
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP,
                        help="Process every Nth frame (default: %d)" % FRAME_SKIP)
    args = parser.parse_args()
    if not args.url and not args.video:
        parser.error("Either --url (RTSP) or --video (file) is required")
    if args.url and args.video:
        parser.error("Use --url or --video, not both")
    run(args)
