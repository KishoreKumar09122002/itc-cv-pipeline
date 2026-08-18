"""Moving zone tracker — zones travel with the belt material.

Each zone represents a physical batch of material. Born at entry (x=0),
moves forward by belt_speed each frame, dies at exit (x=BELT_LENGTH).
A zone is "picked" if any worker's fingertip entered it during its lifetime.

Split-zone mode: each zone has L (near) and R (far) halves. Detection
uses the full zone polygon (strict containment — fingertip must be INSIDE).
L/R attribution is by worker side, not by where the fingertip lands.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field

BELT_LENGTH = 1000.0
BELT_HEIGHT = 200.0


@dataclass
class MovingZone:
    batch_id: int
    x: float
    width: float
    born_frame: int
    picked: bool = False
    picked_by: set = field(default_factory=set)
    frames_alive: int = 0
    pick_count: int = 0
    first_pick_frame: int = -1
    first_pick_worker: int = -1
    picked_L: bool = False
    picked_R: bool = False
    picked_by_L: set = field(default_factory=set)
    picked_by_R: set = field(default_factory=set)


def _get_zone_polygon(zone_x, zone_width, b2p, belt_height=None):
    bh = belt_height if belt_height is not None else BELT_HEIGHT
    pts = np.array([[[zone_x, 0.0], [zone_x + zone_width, 0.0],
                     [zone_x + zone_width, float(bh)],
                     [zone_x, float(bh)]]], dtype=np.float32)
    return cv2.perspectiveTransform(pts, b2p)[0]


def point_in_zone(px, py, zone_x, zone_width, b2p, belt_height=None):
    bh = belt_height if belt_height is not None else BELT_HEIGHT
    if zone_x + zone_width > BELT_LENGTH or zone_x < 0:
        return False
    poly = _get_zone_polygon(zone_x, zone_width, b2p, bh)
    return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0


class MovingZoneTracker:
    def __init__(self, belt_speed, zone_width=250.0, belt_height=None,
                 split_zones=False, **kwargs):
        self.belt_speed = belt_speed
        self.zone_width = zone_width
        self.belt_height = belt_height if belt_height is not None else BELT_HEIGHT
        self.split_zones = split_zones
        self.active_zones = []
        self.completed_zones = []
        self.next_batch_id = 1
        self.frame_count = 0
        self._spawn_accumulator = 0.0

    def update(self, b2p, per_worker_wrists, p2b=None):
        self.frame_count += 1

        for z in self.active_zones:
            z.x += self.belt_speed
            z.frames_alive += 1

        self._spawn_accumulator += self.belt_speed
        while self._spawn_accumulator >= self.zone_width:
            self._spawn_accumulator -= self.zone_width
            new_zone = MovingZone(
                batch_id=self.next_batch_id,
                x=0.0,
                width=self.zone_width,
                born_frame=self.frame_count,
            )
            self.active_zones.append(new_zone)
            self.next_batch_id += 1

        still_active = []
        for z in self.active_zones:
            if z.x >= BELT_LENGTH:
                self.completed_zones.append(z)
            else:
                still_active.append(z)
        self.active_zones = still_active

        frame_info = {}

        for pw in per_worker_wrists:
            wid = pw["worker_id"]
            worker_side = pw.get("worker_side")

            for hand, pixel_key in [("L", "left_pixel"), ("R", "right_pixel")]:
                pixel = pw.get(pixel_key)
                wrist_key = "W%d-%s" % (wid, hand)

                if pixel is None:
                    frame_info[wrist_key] = {"status": "N/A", "batch_id": None, "px": 0, "py": 0}
                    continue

                px, py = float(pixel[0]), float(pixel[1])
                hit_zone = None

                for z in self.active_zones:
                    if point_in_zone(px, py, z.x, z.width, b2p, self.belt_height):
                        hit_zone = z
                        break

                if hit_zone:
                    z = hit_zone
                    if not z.picked:
                        z.first_pick_frame = self.frame_count
                        z.first_pick_worker = wid
                    z.picked = True
                    z.picked_by.add(wid)
                    z.pick_count += 1

                    if self.split_zones and worker_side is not None:
                        if worker_side == "L":
                            z.picked_L = True
                            z.picked_by_L.add(wid)
                        else:
                            z.picked_R = True
                            z.picked_by_R.add(wid)

                    frame_info[wrist_key] = {
                        "status": "IN:B%d" % z.batch_id,
                        "batch_id": z.batch_id,
                        "px": px, "py": py,
                    }
                else:
                    frame_info[wrist_key] = {"status": "OUT", "batch_id": None, "px": px, "py": py}

        return frame_info

    def get_report(self):
        all_zones = self.completed_zones + self.active_zones
        total = len(all_zones)
        completed = len(self.completed_zones)
        picked = sum(1 for z in self.completed_zones if z.picked)
        missed = completed - picked

        per_worker_picks = {}
        for z in self.completed_zones:
            for wid in z.picked_by:
                key = str(wid)
                per_worker_picks[key] = per_worker_picks.get(key, 0) + 1

        report = {
            "total_batches_spawned": total,
            "completed_batches": completed,
            "active_batches": len(self.active_zones),
            "picked": picked,
            "missed": missed,
            "pickup_rate_percent": round(picked / completed * 100, 1) if completed > 0 else 0,
            "per_worker_picks": per_worker_picks,
            "belt_speed": self.belt_speed,
            "zone_width": self.zone_width,
        }

        if self.split_zones:
            picked_L = sum(1 for z in self.completed_zones if z.picked_L)
            picked_R = sum(1 for z in self.completed_zones if z.picked_R)
            picked_both = sum(1 for z in self.completed_zones if z.picked_L and z.picked_R)
            report["picked_L"] = picked_L
            report["picked_R"] = picked_R
            report["picked_both"] = picked_both
            report["coverage_L_pct"] = round(picked_L / completed * 100, 1) if completed > 0 else 0
            report["coverage_R_pct"] = round(picked_R / completed * 100, 1) if completed > 0 else 0
            report["full_coverage_pct"] = round(picked_both / completed * 100, 1) if completed > 0 else 0

        return report
