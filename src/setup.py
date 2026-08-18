import cv2
import numpy as np
import json
import os

BELT_LENGTH = 1000.0
BELT_HEIGHT = 200.0  # default for single-side angles; top angle uses 400

_click_points = []
_click_done = False


def _mouse_callback(event, x, y, flags, param):
    global _click_points, _click_done
    if event == cv2.EVENT_LBUTTONDOWN and len(_click_points) < 4:
        _click_points.append([x, y])
        print(f"  Corner {len(_click_points)}/4: ({x}, {y})")
        if len(_click_points) == 4:
            _click_done = True


def define_belt_zone(video_path):
    """Open first frame, let user click 4 corners of the far-side belt column.

    Click order:
      1. Entry edge, far side (edge nearest far-side workers)
      2. Exit edge, far side
      3. Exit edge, belt center (where column meets the near half)
      4. Entry edge, belt center

    Example (belt moves bottom-right → top-left):
      1 = bottom-right, upper belt edge
      2 = top-left, upper belt edge
      3 = top-left, belt center
      4 = bottom-right, belt center
    """
    global _click_points, _click_done
    _click_points = []
    _click_done = False

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from {video_path}")

    window = "Click 4 belt column corners (see console for order)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, _mouse_callback)

    print("\n=== Belt Column Corner Selection ===")
    print("Click 4 corners of the FAR-SIDE belt column in this order:")
    print("  1. Entry edge, far side (near far-side workers)")
    print("  2. Exit edge, far side")
    print("  3. Exit edge, belt center")
    print("  4. Entry edge, belt center")
    print("Press 'r' to reset, 'q' to quit.\n")

    while True:
        display = frame.copy()
        for i, pt in enumerate(_click_points):
            cv2.circle(display, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(_click_points) >= 2:
            for i in range(len(_click_points) - 1):
                cv2.line(display, tuple(_click_points[i]),
                         tuple(_click_points[i + 1]), (0, 255, 0), 2)
        if len(_click_points) == 4:
            cv2.line(display, tuple(_click_points[3]),
                     tuple(_click_points[0]), (0, 255, 0), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('r'):
            _click_points = []
            _click_done = False
            print("Reset. Click 4 corners again.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif _click_done:
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            return np.array(_click_points, dtype=np.float32)


def compute_transforms(pixel_corners, belt_height=None):
    """Compute pixel<->belt perspective transform matrices.

    pixel_corners: 4 points mapping to belt rectangle corners.
    Maps to belt space: [(0,0), (L,0), (L,H), (0,H)]
    where x=along movement (0=entry, L=exit), y=across belt width.
    """
    bh = belt_height if belt_height is not None else BELT_HEIGHT
    belt_corners = np.array([
        [0, 0],
        [BELT_LENGTH, 0],
        [BELT_LENGTH, bh],
        [0, bh],
    ], dtype=np.float32)

    pixel_to_belt = cv2.getPerspectiveTransform(pixel_corners, belt_corners)
    belt_to_pixel = cv2.getPerspectiveTransform(belt_corners, pixel_corners)
    return pixel_to_belt, belt_to_pixel


def pixel_to_belt_point(px_point, transform_matrix):
    pt = np.array([[[px_point[0], px_point[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, transform_matrix)
    return float(result[0][0][0]), float(result[0][0][1])


def belt_to_pixel_point(belt_point, inverse_transform):
    pt = np.array([[[belt_point[0], belt_point[1]]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, inverse_transform)
    return float(result[0][0][0]), float(result[0][0][1])


def belt_to_pixel_quad(zone_x, zone_width, belt_to_pixel_matrix, belt_height=None):
    """Convert a belt-space zone rectangle to 4 pixel-space corners."""
    bh = belt_height if belt_height is not None else BELT_HEIGHT
    belt_pts = np.array([[
        [zone_x, 0],
        [zone_x + zone_width, 0],
        [zone_x + zone_width, bh],
        [zone_x, bh],
    ]], dtype=np.float32)
    pixel_pts = cv2.perspectiveTransform(belt_pts, belt_to_pixel_matrix)
    return pixel_pts[0].astype(np.int32)


def save_config(pixel_corners, belt_speed_per_frame, video_path, config_path):
    config = {
        "source": video_path,
        "pixel_corners": pixel_corners.tolist() if isinstance(pixel_corners, np.ndarray) else pixel_corners,
        "belt_length": BELT_LENGTH,
        "belt_height": BELT_HEIGHT,
        "belt_speed_per_frame": belt_speed_per_frame,
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    return config


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    config["pixel_corners"] = np.array(config["pixel_corners"], dtype=np.float32)
    print(f"Config loaded from {config_path}")
    return config
