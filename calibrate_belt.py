"""Interactive belt corner calibration.

Click 4 corners of the belt in this order:
  1. Entry edge, NEAR side (left side of belt, closest to camera)
  2. Exit edge, NEAR side
  3. Exit edge, FAR side (right side, near glass partition)
  4. Entry edge, FAR side

After clicking all 4: press 'c' to confirm, 'r' to reset.
ESC to cancel (no accidental quit from single keypress).
"""

import cv2
import numpy as np
import json
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python calibrate_belt.py <RTSP_URL or video_path>")
    sys.exit(1)
VIDEO_PATH = sys.argv[1]
CONFIG_PATH = "config/belt_config_top.json"
FRAME_TIME_SEC = 30 * 60

CORNER_LABELS = ["1:Entry-Near", "2:Exit-Near", "3:Exit-Far", "4:Entry-Far"]
CORNER_COLORS = [(0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 200)]

click_points = []
confirmed = False
state = "clicking"  # clicking | review


def mouse_callback(event, x, y, flags, param):
    global click_points, state
    if state != "clicking":
        return
    if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 4:
        click_points.append([x, y])
        print("  Corner %d/4 (%s): (%d, %d)" % (
            len(click_points), CORNER_LABELS[len(click_points) - 1], x, y))
        if len(click_points) == 4:
            state = "review"
            print()
            print("  All 4 corners marked!")
            print("  Press 'c' to CONFIRM, 'r' to RESET and redo.")


def draw_overlay(display, frame):
    h, w = frame.shape[:2]

    if state == "clicking":
        next_idx = len(click_points)
        if next_idx < 4:
            txt = "Click corner %d: %s" % (next_idx + 1,
                                            CORNER_LABELS[next_idx])
            cv2.putText(display, txt, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(display, "Press 'r' to reset", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    elif state == "review":
        cv2.putText(display, "REVIEW: Press 'c' to CONFIRM  |  'r' to RESET",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    for i, pt in enumerate(click_points):
        color = CORNER_COLORS[i]
        cv2.circle(display, tuple(pt), 10, color, -1)
        cv2.circle(display, tuple(pt), 12, (255, 255, 255), 2)
        cv2.putText(display, CORNER_LABELS[i], (pt[0] + 15, pt[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if len(click_points) >= 2:
        for i in range(len(click_points) - 1):
            cv2.line(display, tuple(click_points[i]),
                     tuple(click_points[i + 1]), (0, 255, 0), 2)
    if len(click_points) == 4:
        cv2.line(display, tuple(click_points[3]),
                 tuple(click_points[0]), (0, 255, 0), 2)

        if state == "review":
            overlay = display.copy()
            pts = np.array(click_points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)


def main():
    global click_points, confirmed, state

    is_rtsp = VIDEO_PATH.lower().startswith("rtsp://")
    if is_rtsp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG if is_rtsp else cv2.CAP_ANY)
    if not cap.isOpened():
        print("Cannot open %s" % VIDEO_PATH)
        return

    if not is_rtsp:
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(FRAME_TIME_SEC * fps))

    frame = None
    for i in range(30):
        ret, f = cap.read()
        if ret and f is not None and f.size > 0:
            frame = f
            if i >= 4:
                break
    cap.release()

    if frame is None:
        print("Cannot read a stable frame from %s" % VIDEO_PATH)
        return

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    window = "Belt Corner Selection - Top Angle"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, mouse_callback)

    print()
    print("=" * 55)
    print("  TOP ANGLE — BELT CORNER SELECTION")
    print("=" * 55)
    print("  Click 4 corners in order:")
    print("    1. Entry-Near  (hopper end, LEFT side)")
    print("    2. Exit-Near   (exit end, LEFT side)")
    print("    3. Exit-Far    (exit end, RIGHT side)")
    print("    4. Entry-Far   (hopper end, RIGHT side)")
    print()
    print("  Entry = hopper/top area")
    print("  Exit  = bottom area where belt goes out")
    print("  Near  = left (walkway/camera side)")
    print("  Far   = right (glass partition side)")
    print()
    print("  'r' = reset    'c' = confirm    ESC = cancel")
    print("=" * 55)

    while True:
        display = frame.copy()
        draw_overlay(display, frame)
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('r'):
            click_points = []
            state = "clicking"
            confirmed = False
            print("  Reset. Click 4 corners again.")

        elif key == ord('c') and state == "review":
            confirmed = True
            break

        elif key == 27:  # ESC only
            cv2.destroyAllWindows()
            print("  Cancelled by user.")
            return

    cv2.destroyAllWindows()

    corners = np.array(click_points, dtype=np.float32)
    config = {
        "source": VIDEO_PATH,
        "pixel_corners": corners.tolist(),
        "belt_length": 1000.0,
        "belt_height": 400.0,
        "belt_mode": "full_width",
        "n_workers": 4,
    }

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print()
    print("  Corners saved to %s:" % CONFIG_PATH)
    for i, label in enumerate(CORNER_LABELS):
        print("    %s: (%d, %d)" % (label, corners[i][0], corners[i][1]))
    print()


if __name__ == "__main__":
    main()
