from ultralytics import YOLO
import numpy as np
import cv2


class StationMapper:
    """Maps ByteTrack track_ids to fixed station IDs (W0, W1, ...) using
    spatial zones in belt-space coordinates.

    Each station is a rectangular region in belt-space. A track is assigned
    to the station whose center is nearest to the worker's shoulder position.
    Hysteresis prevents flicker at zone boundaries.
    """

    def __init__(self, belt_height, belt_length, n_workers, hysteresis_px=15):
        self.belt_height = belt_height
        self.belt_length = belt_length
        self.n_workers = n_workers
        self.hysteresis_px = hysteresis_px
        self.track_to_station = {}
        self.stations = self._build_stations(n_workers, belt_length, belt_height)

    def _build_stations(self, n_workers, belt_length, belt_height):
        mid_y = belt_height / 2.0
        if n_workers <= 2:
            return [
                {"id": 0, "label": "Near", "cx": belt_length / 2, "cy": mid_y * 0.5,
                 "y_min": 0, "y_max": mid_y},
                {"id": 1, "label": "Far", "cx": belt_length / 2, "cy": mid_y * 1.5,
                 "y_min": mid_y, "y_max": belt_height},
            ]
        per_side = n_workers // 2
        x_step = belt_length / per_side
        stations = []
        for i in range(per_side):
            stations.append({
                "id": i, "label": "Near-%d" % i,
                "cx": x_step * (i + 0.5), "cy": mid_y * 0.5,
                "y_min": 0, "y_max": mid_y,
                "x_min": x_step * i, "x_max": x_step * (i + 1),
            })
        for i in range(per_side):
            stations.append({
                "id": per_side + i, "label": "Far-%d" % i,
                "cx": x_step * (i + 0.5), "cy": mid_y * 1.5,
                "y_min": mid_y, "y_max": belt_height,
                "x_min": x_step * i, "x_max": x_step * (i + 1),
            })
        return stations

    def map(self, workers):
        """Assign station-based worker_id to each worker using track_id + position."""
        claimed = set()
        result = []

        for w in workers:
            bx = w.get("_belt_x", 0)
            by = w.get("_belt_y", 0)
            track_id = w.get("_track_id")

            prev_station = self.track_to_station.get(track_id)

            best_station = None
            best_dist = float("inf")
            for s in self.stations:
                if s["id"] in claimed:
                    continue
                dx = bx - s["cx"]
                dy = by - s["cy"]
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if prev_station == s["id"]:
                    dist -= self.hysteresis_px
                if dist < best_dist:
                    best_dist = dist
                    best_station = s["id"]

            if best_station is not None:
                w["worker_id"] = best_station
                claimed.add(best_station)
                if track_id is not None:
                    self.track_to_station[track_id] = best_station
                result.append(w)

        result.sort(key=lambda w: w["worker_id"])
        return result


class FrameProcessor:
    def __init__(self, model_path="yolov8n-pose.pt", confidence_threshold=0.3):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self._tracking_enabled = False

    def get_wrists(self, frame):
        """Run YOLO-Pose and extract all wrist detections.

        Returns list of dicts, each with:
          x, y (pixel coords), confidence, side ("L"/"R"), person (index)
        """
        results = self.model(frame, verbose=False)
        wrists = []
        if not results or results[0].keypoints is None:
            return wrists

        kps = results[0].keypoints.data.cpu().numpy()
        for person_idx, person in enumerate(kps):
            left_wrist = person[9]
            right_wrist = person[10]
            if left_wrist[2] > self.confidence_threshold:
                wrists.append({
                    "x": float(left_wrist[0]),
                    "y": float(left_wrist[1]),
                    "confidence": float(left_wrist[2]),
                    "side": "L",
                    "person": person_idx,
                })
            if right_wrist[2] > self.confidence_threshold:
                wrists.append({
                    "x": float(right_wrist[0]),
                    "y": float(right_wrist[1]),
                    "confidence": float(right_wrist[2]),
                    "side": "R",
                    "person": person_idx,
                })
        return wrists

    def enable_tracking(self, tracker="bytetrack.yaml"):
        """Enable ByteTrack-based persistent tracking across frames."""
        self._tracking_enabled = True
        self._tracker_config = tracker

    def get_worker_wrists(self, frame, fingertip_extend=0.3):
        """Run YOLO-Pose, return per-person wrist pairs plus shoulder midpoint.

        When tracking is enabled (via enable_tracking()), uses ByteTrack for
        persistent track_id assignment across frames. Otherwise falls back to
        per-frame person_idx.

        Each wrist entry includes an estimated fingertip position computed by
        extending the elbow→wrist vector by `fingertip_extend` (default 30%
        of forearm length). Falls back to shoulder→wrist if elbow is missing.

        Returns list of dicts, one per detected person:
          { person, _track_id, left: {x,y,conf,tip_x,tip_y}|None,
            right: {x,y,conf,tip_x,tip_y}|None,
            shoulder_mid: (x,y)|None }
        """
        if self._tracking_enabled:
            results = self.model.track(
                frame, persist=True, tracker=self._tracker_config, verbose=False)
        else:
            results = self.model(frame, verbose=False)

        workers = []
        if not results or results[0].keypoints is None:
            return workers

        kps = results[0].keypoints.data.cpu().numpy()
        track_ids = None
        if self._tracking_enabled and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for person_idx, person in enumerate(kps):
            lw = person[9]   # left wrist
            rw = person[10]  # right wrist
            le = person[7]   # left elbow
            re = person[8]   # right elbow
            ls = person[5]   # left shoulder
            rs = person[6]   # right shoulder
            tid = int(track_ids[person_idx]) if track_ids is not None and person_idx < len(track_ids) else None
            entry = {"person": person_idx, "_track_id": tid,
                     "left": None, "right": None, "shoulder_mid": None}

            for wrist, elbow, shoulder, side in [
                (lw, le, ls, "left"), (rw, re, rs, "right")
            ]:
                if wrist[2] <= self.confidence_threshold:
                    continue
                wx, wy = float(wrist[0]), float(wrist[1])
                tip_x, tip_y = wx, wy
                if elbow[2] > self.confidence_threshold:
                    dx = wx - float(elbow[0])
                    dy = wy - float(elbow[1])
                    tip_x = wx + dx * fingertip_extend
                    tip_y = wy + dy * fingertip_extend
                elif shoulder[2] > self.confidence_threshold:
                    dx = wx - float(shoulder[0])
                    dy = wy - float(shoulder[1])
                    arm_len = max((dx**2 + dy**2)**0.5, 1.0)
                    tip_x = wx + dx / arm_len * arm_len * fingertip_extend
                    tip_y = wy + dy / arm_len * arm_len * fingertip_extend
                entry[side] = {
                    "x": wx, "y": wy,
                    "confidence": float(wrist[2]),
                    "tip_x": tip_x, "tip_y": tip_y,
                }

            if ls[2] > 0.3 and rs[2] > 0.3:
                entry["shoulder_mid"] = (
                    float((ls[0] + rs[0]) / 2),
                    float((ls[1] + rs[1]) / 2),
                )
            elif ls[2] > 0.3:
                entry["shoulder_mid"] = (float(ls[0]), float(ls[1]))
            elif rs[2] > 0.3:
                entry["shoulder_mid"] = (float(rs[0]), float(rs[1]))
            workers.append(entry)
        return workers

    def get_all_keypoints(self, frame):
        """Run YOLO-Pose and return raw keypoints for all detected persons.
        Returns list of (17, 3) arrays — each row is (x, y, confidence).
        """
        results = self.model(frame, verbose=False)
        if not results or results[0].keypoints is None:
            return []
        return results[0].keypoints.data.cpu().numpy()

    def _deduplicate(self, workers, min_dist=30):
        """Merge detections whose shoulders are within min_dist pixels."""
        keep = []
        for w in workers:
            if w["shoulder_mid"] is None:
                continue
            sx, sy = w["shoulder_mid"]
            is_dup = False
            for k in keep:
                kx, ky = k["shoulder_mid"]
                if ((sx - kx)**2 + (sy - ky)**2)**0.5 < min_dist:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(w)
        return keep

    def filter_belt_proximity(self, workers, pixel_corners, p2b, belt_height,
                                 n_workers=4, margin_px=200):
        """Keep workers whose shoulder is near the belt polygon in pixel space.

        Deduplicates overlapping detections first, then uses pixel-space
        distance to the belt polygon. Projects to belt-space for side assignment.
        """
        workers = self._deduplicate(workers)
        poly = pixel_corners.reshape((-1, 1, 2)).astype(np.float32)
        candidates = []
        for w in workers:
            if w["shoulder_mid"] is None:
                continue
            sx, sy = w["shoulder_mid"]
            dist = cv2.pointPolygonTest(poly, (float(sx), float(sy)), True)
            if dist < -margin_px:
                continue
            belt_pt = cv2.perspectiveTransform(
                np.array([[[float(sx), float(sy)]]], dtype=np.float32), p2b
            )[0][0]
            w["_belt_x"] = float(belt_pt[0])
            w["_belt_y"] = float(belt_pt[1])
            w["_pixel_dist"] = dist
            candidates.append(w)

        candidates.sort(key=lambda w: abs(w["_pixel_dist"]))
        return candidates[:n_workers]

    def assign_worker_ids_by_side(self, workers, belt_height):
        """Assign stable IDs: near-side workers first (W0,W1), then far-side (W2,W3).

        Near side = belt_y < belt_height/2, far side = belt_y >= belt_height/2.
        Within each side, sorted by belt_x (entry→exit).
        """
        mid_y = belt_height / 2.0
        near = [w for w in workers if w.get("_belt_y", 0) < mid_y]
        far = [w for w in workers if w.get("_belt_y", 0) >= mid_y]
        near.sort(key=lambda w: w.get("_belt_x", 0))
        far.sort(key=lambda w: w.get("_belt_x", 0))
        combined = near + far
        for i, w in enumerate(combined):
            w["worker_id"] = i
        return combined

    def filter_far_side(self, workers, pixel_corners, n_workers=2):
        """Keep the n far-side workers closest to the belt zone.

        Works purely in pixel space — no corner-order dependency.
        Far side = above the belt's top edge in the frame.
        Filters to workers whose shoulder x overlaps the belt zone,
        then picks the n closest to the belt edge.
        """
        belt_min_y = float(np.min(pixel_corners[:, 1]))
        belt_x_min = float(np.min(pixel_corners[:, 0]))
        belt_x_max = float(np.max(pixel_corners[:, 0]))
        x_margin = (belt_x_max - belt_x_min) * 0.15

        candidates = []
        for w in workers:
            if w["shoulder_mid"] is None:
                continue
            sx, sy = w["shoulder_mid"]
            if sy >= belt_min_y:
                continue
            if sx < belt_x_min - x_margin or sx > belt_x_max + x_margin:
                continue
            w["_dist_to_belt"] = belt_min_y - sy
            candidates.append(w)

        candidates.sort(key=lambda w: w["_dist_to_belt"])
        return candidates[:n_workers]

    def assign_worker_ids(self, far_side_workers):
        """Assign stable worker IDs based on horizontal position.

        Leftmost wrist x → Worker 0, next → Worker 1.
        Returns list sorted by x-position with 'worker_id' added.
        """
        def avg_x(w):
            xs = []
            if w["left"]:
                xs.append(w["left"]["x"])
            if w["right"]:
                xs.append(w["right"]["x"])
            return sum(xs) / len(xs) if xs else 0

        sorted_workers = sorted(far_side_workers, key=avg_x)
        for i, w in enumerate(sorted_workers):
            w["worker_id"] = i
        return sorted_workers
