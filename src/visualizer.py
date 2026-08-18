import cv2
import numpy as np
from src.setup import belt_to_pixel_quad


def draw_zones_split(frame, active_zones, belt_to_pixel_matrix,
                     belt_height, mid_y):
    """Draw zones with independent L/R half coloring and mid-line divider."""
    from moving_zone_tracker import BELT_LENGTH
    for z in active_zones:
        if z.x < 0 or z.x + z.width > BELT_LENGTH:
            continue
        pts_L = np.array([[[z.x, 0], [z.x + z.width, 0],
                           [z.x + z.width, mid_y], [z.x, mid_y]]],
                         dtype=np.float32)
        pts_R = np.array([[[z.x, mid_y], [z.x + z.width, mid_y],
                           [z.x + z.width, belt_height],
                           [z.x, belt_height]]], dtype=np.float32)
        quad_L = cv2.perspectiveTransform(
            pts_L, belt_to_pixel_matrix)[0].astype(np.int32)
        quad_R = cv2.perspectiveTransform(
            pts_R, belt_to_pixel_matrix)[0].astype(np.int32)
        col_L = (0, 200, 0) if z.picked_L else (0, 0, 160)
        col_R = (0, 200, 0) if z.picked_R else (0, 0, 160)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [quad_L], col_L)
        cv2.fillPoly(overlay, [quad_R], col_R)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
        quad_full = belt_to_pixel_quad(z.x, z.width, belt_to_pixel_matrix,
                                       belt_height=belt_height)
        cv2.polylines(frame, [quad_full], True, (0, 0, 220), 2)
        pt_m1 = cv2.perspectiveTransform(
            np.array([[[z.x, mid_y]]], dtype=np.float32),
            belt_to_pixel_matrix)[0][0]
        pt_m2 = cv2.perspectiveTransform(
            np.array([[[z.x + z.width, mid_y]]], dtype=np.float32),
            belt_to_pixel_matrix)[0][0]
        cv2.line(frame, (int(pt_m1[0]), int(pt_m1[1])),
                 (int(pt_m2[0]), int(pt_m2[1])), (255, 255, 255), 3)


def draw_hands(frame, workers, frame_info, mid_y):
    """Draw hand tracking with fingertip lines, status colors, and labels."""
    for w in workers:
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
                cv2.circle(frame, (wx, wy), 4, (0, 0, 255), -1)
                cv2.circle(frame, (tx, ty), 6, color, -1)
                cv2.line(frame, (wx, wy), (tx, ty), (0, 255, 255), 1)
            else:
                cv2.circle(frame, (wx, wy), 3, (128, 128, 128), -1)
        if w.get("shoulder_mid"):
            sx, sy = int(w["shoulder_mid"][0]), int(w["shoulder_mid"][1])
            cv2.putText(frame, "W%d(%s)" % (wid, w_side),
                        (sx + 8, sy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


def draw_stats_bar(frame, report, frame_num, ts):
    """Draw detailed stats overlay with coverage breakdown."""
    cv2.putText(
        frame,
        "F%d (%.0fs) | Done:%d Active:%d | "
        "Any:%.1f%% L:%.1f%% R:%.1f%% Both:%.1f%%" % (
            frame_num, ts,
            report["completed_batches"],
            report["active_batches"],
            report["pickup_rate_percent"],
            report.get("coverage_L_pct", 0),
            report.get("coverage_R_pct", 0),
            report.get("full_coverage_pct", 0)),
        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)


def draw_belt_outline(frame, pixel_corners):
    pts = pixel_corners.astype(np.int32)
    cv2.polylines(frame, [pts], True, (255, 255, 0), 2)
