"""ITC Belt Monitoring Dashboard — frame-level analytics from moving zone JSON."""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import json
import glob
import os
import cv2
import numpy as np
import tempfile
import subprocess
import sqlite3
import imageio_ffmpeg
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

st.set_page_config(page_title="ITC Belt Monitor", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
SUMMARY_DIR = "output/moving_zones"

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
    .kpi-card {
        text-align: center;
        padding: 20px 16px;
        border-radius: 12px;
        border: 2px solid rgba(128,128,128,0.25);
        background: transparent;
    }
    .kpi-card .value { font-size: 2.6rem; font-weight: 700; margin: 0; }
    .kpi-card .label { font-size: 0.82rem; opacity: 0.65; margin: 4px 0 0 0; }
    .kpi-card .sub { font-size: 0.75rem; opacity: 0.5; margin: 2px 0 0 0; }
    .worker-card {
        background: transparent;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        border-left: 5px solid;
        border-top: 1px solid rgba(128,128,128,0.2);
        border-right: 1px solid rgba(128,128,128,0.2);
        border-bottom: 1px solid rgba(128,128,128,0.2);
        min-height: 200px;
    }
    .worker-card h3 { margin: 0 0 16px 0; }
    .worker-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid rgba(128,128,128,0.15);
        font-size: 0.92rem;
    }
    .worker-row:last-child { border-bottom: none; }
    .worker-row .wlabel { opacity: 0.7; text-align: left; }
    .worker-row .wvalue { font-weight: 600; text-align: right; }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 8px 0 4px 0;
    }
    .section-desc {
        font-size: 0.78rem;
        opacity: 0.65;
        margin: 0 0 12px 0;
    }
    /* Compact sidebar selectboxes */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] {
        min-width: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        padding-left: 6px !important;
        padding-right: 20px !important;
        font-size: 0.85rem !important;
        min-height: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_summaries():
    files = set()
    for base in [BASE_DIR, DATA_DIR]:
        files.update(glob.glob(os.path.join(base, SUMMARY_DIR, "*_moving_summary.json")))
    data = {}
    for f in sorted(files):
        with open(f) as fh:
            d = json.load(fh)
            label = d.get("video", os.path.basename(f)).replace(".mp4", "").replace("_", " ").title()
            data[label] = d
    return data


def load_live_sessions(db_path="output/live.db"):
    if not os.path.exists(db_path):
        return {}
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        sessions = conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC").fetchall()
        result = {}
        for sess in sessions:
            sid = sess["id"]
            status_tag = "LIVE" if sess["status"] == "running" else "Done"
            label = "[%s] Session %d (%s)" % (status_tag, sid, sess["start_time"])

            rows = conn.execute(
                "SELECT data FROM per_second WHERE session_id = ? ORDER BY t",
                (sid,)).fetchall()
            per_second = [json.loads(r["data"]) for r in rows]

            ze_rows = conn.execute(
                "SELECT * FROM zone_events WHERE session_id = ? ORDER BY id",
                (sid,)).fetchall()
            zone_events = []
            for z in ze_rows:
                zone_events.append({
                    "batch_id": z["batch_id"],
                    "spawn_time": z["spawn_time"],
                    "retire_time": z["retire_time"],
                    "picked": bool(z["picked"]),
                    "picked_L": bool(z["picked_L"]),
                    "picked_R": bool(z["picked_R"]),
                    "picked_by": json.loads(z["picked_by"]) if z["picked_by"] else [],
                    "picked_by_L": json.loads(z["picked_by_L"]) if z["picked_by_L"] else [],
                    "picked_by_R": json.loads(z["picked_by_R"]) if z["picked_by_R"] else [],
                    "first_pick_time": z["first_pick_time"],
                    "first_pick_worker": z["first_pick_worker"],
                })

            duration = per_second[-1]["t"] if per_second else 0

            rec_path = None
            ann_path = None
            sess_keys = sess.keys()
            try:
                rec_path = sess["recording_path"] if "recording_path" in sess_keys else None
                ann_path = sess["annotated_path"] if "annotated_path" in sess_keys else None
            except Exception:
                pass

            result[label] = {
                "video": "Live Session %d" % sid,
                "angle": "top",
                "duration_seconds": duration,
                "fps": sess["fps"] or 25.0,
                "effective_fps": sess["fps"] or 25.0,
                "belt_speed": sess["belt_speed"],
                "zone_width": sess["zone_width"],
                "belt_height": sess["belt_height"],
                "split_zones": True,
                "n_workers": sess["n_workers"],
                "zone_events": zone_events,
                "per_second": per_second,
                "_live": sess["status"] == "running",
                "_session_id": sid,
                "_recording_path": rec_path,
                "_annotated_path": ann_path,
            }
        conn.close()
        return result
    except Exception:
        return {}


def fmt_time(sec):
    sec = int(sec)
    if sec >= 3600:
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return "%d:%02d:%02d" % (h, m, s)
    m, s = divmod(sec, 60)
    return "%d:%02d" % (m, s)


def detect_n_workers(data):
    n = data.get("n_workers")
    if n:
        return n
    per_sec = data.get("per_second", [])
    if per_sec:
        sample = per_sec[0]
        n = 0
        while ("w%d_present" % n) in sample:
            n += 1
        return max(n, 2)
    return 2


def filter_data(data, t_start, t_end):
    zones = data.get("zone_events", [])
    per_sec = data.get("per_second", [])
    n_workers = detect_n_workers(data)

    zones_in_range = [z for z in zones if z["retire_time"] >= t_start and z["retire_time"] <= t_end]
    total_zones = len(zones_in_range)
    picked_zones = [z for z in zones_in_range if z["picked"]]
    missed_zones = total_zones - len(picked_zones)
    coverage_pct = round(len(picked_zones) / total_zones * 100, 1) if total_zones > 0 else 0

    worker_zones = {}
    for wid in range(n_workers):
        worker_zones[wid] = len([z for z in picked_zones if wid in z.get("picked_by", [])])
    multi_zones = len([z for z in picked_zones if len(z.get("picked_by", [])) >= 2])
    worker_exclusive = {}
    for wid in range(n_workers):
        worker_exclusive[wid] = len([z for z in picked_zones
                                     if len(z.get("picked_by", [])) == 1 and wid in z["picked_by"]])

    secs_in_range = [s for s in per_sec if s["t"] >= t_start and s["t"] <= t_end]
    n_secs = len(secs_in_range)

    if n_secs > 0:
        picking_secs = sum(1 for s in secs_in_range if s.get("any_picking"))
        gap_secs = n_secs - picking_secs
    else:
        picking_secs = gap_secs = 0

    worker_picks = {}
    worker_ppm = {}
    if secs_in_range:
        baselines = {}
        for wid in range(n_workers):
            baselines[wid] = 0
        for s in per_sec:
            if s["t"] < t_start:
                for wid in range(n_workers):
                    baselines[wid] = s.get("w%d_picks" % wid, 0)
            else:
                break
        end_entry = secs_in_range[-1]
        for wid in range(n_workers):
            worker_picks[wid] = end_entry.get("w%d_picks" % wid, 0) - baselines[wid]
    else:
        for wid in range(n_workers):
            worker_picks[wid] = 0

    duration_mins = (t_end - t_start) / 60
    for wid in range(n_workers):
        worker_ppm[wid] = round(worker_picks[wid] / duration_mins, 1) if duration_mins > 0 else 0

    # Build sweep-line for unchecked-zone intervals — O(n log n), not O(n * m)
    # unchecked window: [spawn_time, first_pick_time) if picked, [spawn_time, retire_time] if not
    sweep_opens = []
    sweep_closes = []
    for z in zones:
        if not z["picked"]:
            sweep_opens.append(z["spawn_time"])
            sweep_closes.append(z["retire_time"] + 1)  # inclusive retire_time
        elif z.get("first_pick_time") is not None:
            sweep_opens.append(z["spawn_time"])
            sweep_closes.append(z["first_pick_time"])   # exclusive: zone picked at this t
    sweep_opens.sort()
    sweep_closes.sort()

    open_i = close_i = 0
    active_unchecked = 0
    coverage_gaps = []
    gap_start = None
    for s in secs_in_range:
        t = s["t"]
        while open_i < len(sweep_opens) and sweep_opens[open_i] <= t:
            active_unchecked += 1
            open_i += 1
        while close_i < len(sweep_closes) and sweep_closes[close_i] <= t:
            active_unchecked -= 1
            close_i += 1
        active_unchecked = max(0, active_unchecked)
        is_gap = active_unchecked > 0 and not s.get("any_picking")
        if is_gap:
            if gap_start is None:
                gap_start = t
        else:
            if gap_start is not None:
                gap_dur = t - gap_start
                if gap_dur >= 3:
                    coverage_gaps.append({"start": gap_start, "duration": gap_dur})
                gap_start = None
    if gap_start is not None:
        gap_dur = (secs_in_range[-1]["t"] + 1) - gap_start
        if gap_dur >= 3:
            coverage_gaps.append({"start": gap_start, "duration": gap_dur})

    coverage_gaps.sort(key=lambda g: -g["duration"])

    worker_present_secs = {}
    worker_active_secs = {}
    worker_idle_secs = {}
    worker_util = {}
    for wid in range(n_workers):
        present = sum(1 for s in secs_in_range if s.get("w%d_present" % wid, s.get("w%d_active" % wid)))
        active = sum(1 for s in secs_in_range if s.get("w%d_active" % wid))
        worker_present_secs[wid] = present
        worker_active_secs[wid] = active
        worker_idle_secs[wid] = present - active
        worker_util[wid] = round(active / present * 100, 1) if present > 0 else 0

    worker_responses = {wid: [] for wid in range(n_workers)}
    for z in zones_in_range:
        if z.get("first_pick_time") is not None and z.get("first_pick_worker") is not None:
            response = z["first_pick_time"] - z["spawn_time"]
            pw = z["first_pick_worker"]
            if response >= 0 and pw in worker_responses:
                worker_responses[pw].append(response)
    worker_avg_response = {}
    for wid in range(n_workers):
        r = worker_responses[wid]
        worker_avg_response[wid] = round(sum(r) / len(r), 1) if r else None

    # --- Miss accountability: present-miss vs absent-miss by side ---
    near_workers = list(range(n_workers // 2))
    far_workers = list(range(n_workers // 2, n_workers))
    split = data.get("split_zones", False) and n_workers >= 4
    sec_by_t = {s["t"]: s for s in secs_in_range}

    present_misses = 0
    absent_misses = 0
    l_picked = l_present_miss = l_absent_miss = 0
    r_picked = r_present_miss = r_absent_miss = 0
    l_total = r_total = 0

    for z in zones_in_range:
        zone_t0 = int(z["spawn_time"])
        zone_t1 = int(z["retire_time"])

        if not z["picked"]:
            any_present = False
            for t in range(zone_t0, zone_t1 + 1):
                s = sec_by_t.get(t)
                if s and any(s.get("w%d_present" % w) for w in range(n_workers)):
                    any_present = True
                    break
            if any_present:
                present_misses += 1
            else:
                absent_misses += 1

        if split:
            l_total += 1
            r_total += 1
            if z.get("picked_L", z["picked"]):
                l_picked += 1
            else:
                near_present = False
                for t in range(zone_t0, zone_t1 + 1):
                    s = sec_by_t.get(t)
                    if s and any(s.get("w%d_present" % w) for w in near_workers):
                        near_present = True
                        break
                if near_present:
                    l_present_miss += 1
                else:
                    l_absent_miss += 1

            if z.get("picked_R", z["picked"]):
                r_picked += 1
            else:
                far_present = False
                for t in range(zone_t0, zone_t1 + 1):
                    s = sec_by_t.get(t)
                    if s and any(s.get("w%d_present" % w) for w in far_workers):
                        far_present = True
                        break
                if far_present:
                    r_present_miss += 1
                else:
                    r_absent_miss += 1

    result = {
        "n_workers": n_workers,
        "total_zones": total_zones,
        "picked_zones": len(picked_zones),
        "missed_zones": missed_zones,
        "coverage_pct": coverage_pct,
        "multi_zones": multi_zones,
        "gap_secs": gap_secs,
        "picking_secs": picking_secs,
        "n_secs": n_secs,
        "secs_in_range": secs_in_range,
        "zones_in_range": zones_in_range,
        "coverage_gaps": coverage_gaps[:5],
        "present_misses": present_misses,
        "absent_misses": absent_misses,
        "split_zones": split,
        "l_total": l_total,
        "l_picked": l_picked,
        "l_present_miss": l_present_miss,
        "l_absent_miss": l_absent_miss,
        "r_total": r_total,
        "r_picked": r_picked,
        "r_present_miss": r_present_miss,
        "r_absent_miss": r_absent_miss,
    }
    for wid in range(n_workers):
        result["w%d_zones" % wid] = worker_zones[wid]
        result["w%d_exclusive" % wid] = worker_exclusive[wid]
        result["w%d_picks" % wid] = worker_picks[wid]
        result["w%d_ppm" % wid] = worker_ppm[wid]
        result["w%d_present_secs" % wid] = worker_present_secs[wid]
        result["w%d_active_secs" % wid] = worker_active_secs[wid]
        result["w%d_idle_secs" % wid] = worker_idle_secs[wid]
        result["w%d_util" % wid] = worker_util[wid]
        result["w%d_avg_response" % wid] = worker_avg_response[wid]
    result["both_zones"] = multi_zones
    return result


def render_kpis(m):
    split = m.get("split_zones", False)

    if split:
        col1, col2, col3, col4 = st.columns(4)
        l_total = m["l_total"]
        r_total = m["r_total"]
        l_picked = m["l_picked"]
        r_picked = m["r_picked"]
        l_missed = l_total - l_picked
        r_missed = r_total - r_picked
        l_pct = round(l_picked / l_total * 100, 1) if l_total > 0 else 0
        r_pct = round(r_picked / r_total * 100, 1) if r_total > 0 else 0

        l_color = "#2ecc71" if l_pct >= 80 else ("#f39c12" if l_pct >= 60 else "#e74c3c")
        with col1:
            st.markdown(
                '<div class="kpi-card" style="border-color:%s">'
                '<p class="value" style="color:%s">%.1f%%</p>'
                '<p class="label">Near Side (L)</p>'
                '<p class="sub">%d of %d — %d missed</p>'
                '</div>' % (l_color, l_color, l_pct, l_picked, l_total, l_missed),
                unsafe_allow_html=True,
            )
        r_color = "#2ecc71" if r_pct >= 80 else ("#f39c12" if r_pct >= 60 else "#e74c3c")
        with col2:
            st.markdown(
                '<div class="kpi-card" style="border-color:%s">'
                '<p class="value" style="color:%s">%.1f%%</p>'
                '<p class="label">Far Side (R)</p>'
                '<p class="sub">%d of %d — %d missed</p>'
                '</div>' % (r_color, r_color, r_pct, r_picked, r_total, r_missed),
                unsafe_allow_html=True,
            )
    else:
        col1, col2, col3, col4 = st.columns(4)
        cov_color = "#2ecc71" if m["coverage_pct"] >= 80 else (
            "#f39c12" if m["coverage_pct"] >= 60 else "#e74c3c")
        with col1:
            st.markdown(
                '<div class="kpi-card" style="border-color:%s">'
                '<p class="value" style="color:%s">%.1f%%</p>'
                '<p class="label">Material Inspected</p>'
                '<p class="sub">%d of %d zones checked</p>'
                '</div>' % (cov_color, cov_color, m["coverage_pct"],
                            m["picked_zones"], m["total_zones"]),
                unsafe_allow_html=True,
            )
        miss_color = "#e74c3c" if m["missed_zones"] > 0 else "#2ecc71"
        with col2:
            st.markdown(
                '<div class="kpi-card" style="border-color:%s">'
                '<p class="value" style="color:%s">%d</p>'
                '<p class="label">Zones Missed</p>'
                '<p class="sub">Material passed without inspection</p>'
                '</div>' % (miss_color, miss_color, m["missed_zones"]),
                unsafe_allow_html=True,
            )

    with col3:
        n_w = m["n_workers"]
        total_picks = sum(m.get("w%d_picks" % i, 0) for i in range(n_w))
        st.markdown(
            '<div class="kpi-card">'
            '<p class="value" style="color:#3498db">%d</p>'
            '<p class="label">Total Pick Actions</p>'
            '<p class="sub">Hand reached into belt zone</p>'
            '</div>' % total_picks,
            unsafe_allow_html=True,
        )
    with col4:
        gap_color = "#e74c3c" if m["gap_secs"] > 30 else "#2ecc71"
        st.markdown(
            '<div class="kpi-card" style="border-color:%s">'
            '<p class="value" style="color:%s">%s</p>'
            '<p class="label">Hands Off Belt</p>'
            '<p class="sub">No worker hand was on the belt</p>'
            '</div>' % (gap_color, gap_color, fmt_time(m["gap_secs"])),
            unsafe_allow_html=True,
        )


WORKER_COLORS = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c", "#2ecc71"]

WORKER_SIDE_LABELS = {
    2: {0: "Near side", 1: "Far side"},
    4: {0: "Near-left", 1: "Near-right", 2: "Far-left", 3: "Far-right"},
}


def render_workers(m):
    st.markdown('<p class="section-title">Worker Performance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Individual productivity metrics tracked by computer vision. '
        'All values update dynamically with the selected time range.</p>',
        unsafe_allow_html=True,
    )

    n_w = m["n_workers"]
    side_map = WORKER_SIDE_LABELS.get(n_w, {})

    for row_start in range(0, n_w, 2):
        cols = st.columns(2)
        for col_idx in range(2):
            wid = row_start + col_idx
            if wid >= n_w:
                break
            color = WORKER_COLORS[wid % len(WORKER_COLORS)]
            side = side_map.get(wid, "Pos %d" % wid)
            title = "Worker %d (%s)" % (wid, side)

            zones = m["w%d_zones" % wid]
            exclusive = m["w%d_exclusive" % wid]
            picks = m["w%d_picks" % wid]
            ppm = m["w%d_ppm" % wid]
            shared = zones - exclusive
            present = m["w%d_present_secs" % wid]
            active = m["w%d_active_secs" % wid]
            idle = m["w%d_idle_secs" % wid]
            absent = m["n_secs"] - present
            util = m["w%d_util" % wid]
            avg_resp = m["w%d_avg_response" % wid]

            util_color = "#2ecc71" if util >= 70 else ("#f39c12" if util >= 50 else "#e74c3c")
            resp_str = "%.1f sec" % avg_resp if avg_resp is not None else "—"
            presence_pct = round(present / m["n_secs"] * 100, 1) if m["n_secs"] > 0 else 0

            with cols[col_idx]:
                st.markdown(
                    '<div class="worker-card" style="border-left-color:%s">'
                    '<h3 style="color:%s">%s</h3>'

                    '<div class="worker-row" style="opacity:0.5; font-size:0.78rem; border-bottom:none; padding-bottom:0;">'
                    '<span>PRESENCE &amp; PRODUCTIVITY</span></div>'

                    '<div class="worker-row"><span class="wlabel">Present at Belt</span>'
                    '<span class="wvalue">%s <span style="opacity:0.6; font-size:0.85rem">(%.1f%%)</span></span></div>'

                    '<div class="worker-row"><span class="wlabel">Away from Belt</span>'
                    '<span class="wvalue">%s</span></div>'

                    '<div class="worker-row"><span class="wlabel">Active Picking Time</span>'
                    '<span class="wvalue">%s</span></div>'

                    '<div class="worker-row"><span class="wlabel">Idle at Belt</span>'
                    '<span class="wvalue">%s</span></div>'

                    '<div class="worker-row"><span class="wlabel">Utilization</span>'
                    '<span class="wvalue" style="color:%s">%.1f%%</span></div>'

                    '<div class="worker-row" style="opacity:0.5; font-size:0.78rem; border-bottom:none; padding-bottom:0; padding-top:12px;">'
                    '<span>INSPECTION</span></div>'

                    '<div class="worker-row"><span class="wlabel">Zones Checked</span>'
                    '<span class="wvalue">%d</span></div>'

                    '<div class="worker-row"><span class="wlabel">— Only by this worker</span>'
                    '<span class="wvalue">%d</span></div>'

                    '<div class="worker-row"><span class="wlabel">— Shared with others</span>'
                    '<span class="wvalue">%d</span></div>'

                    '<div class="worker-row"><span class="wlabel">Pick Actions</span>'
                    '<span class="wvalue">%d</span></div>'

                    '<div class="worker-row"><span class="wlabel">Pick Rate</span>'
                    '<span class="wvalue">%s / min</span></div>'

                    '<div class="worker-row"><span class="wlabel">Avg Response Time</span>'
                    '<span class="wvalue">%s</span></div>'

                    '</div>' % (
                        color, color, title,
                        fmt_time(present), presence_pct,
                        fmt_time(absent),
                        fmt_time(active), fmt_time(idle),
                        util_color, util,
                        zones, exclusive, shared, picks, ppm,
                        resp_str,
                    ),
                    unsafe_allow_html=True,
                )


def render_activity_strip(m):
    st.markdown('<p class="section-title">Inspection Timeline (per minute)</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Each minute shows: what % of that minute had a hand on the belt (inspection), '
        'and what % each worker\'s hands were moving. '
        'Red shaded regions mark gaps where unchecked zones passed without inspection.</p>',
        unsafe_allow_html=True,
    )

    secs = m["secs_in_range"]
    if not secs:
        st.info("No data in selected range.")
        return

    n_w = m["n_workers"]

    minutes = {}
    for s in secs:
        minute = s["t"] // 60
        if minute not in minutes:
            bucket = {"total": 0, "inspecting": 0}
            for wid in range(n_w):
                bucket["w%d" % wid] = 0
            minutes[minute] = bucket
        minutes[minute]["total"] += 1
        if s.get("any_picking"):
            minutes[minute]["inspecting"] += 1
        for wid in range(n_w):
            if s.get("w%d_active" % wid):
                minutes[minute]["w%d" % wid] += 1

    sorted_mins = sorted(minutes.keys())
    labels = [fmt_time(mn * 60) for mn in sorted_mins]
    inspect_pcts = [round(minutes[mn]["inspecting"] / minutes[mn]["total"] * 100, 1) for mn in sorted_mins]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Belt inspection %",
        x=labels, y=inspect_pcts,
        marker_color="#2a78d6",
        hovertemplate="Time %{x}<br>Inspection: %{y:.1f}%<extra></extra>",
    ))
    for wid in range(n_w):
        color = WORKER_COLORS[wid % len(WORKER_COLORS)]
        w_pcts = [round(minutes[mn]["w%d" % wid] / minutes[mn]["total"] * 100, 1) for mn in sorted_mins]
        fig.add_trace(go.Bar(
            name="W%d hands moving %%" % wid,
            x=labels, y=w_pcts,
            marker_color=color,
            hovertemplate="Time %%{x}<br>W%d active: %%{y:.1f}%%<extra></extra>" % wid,
        ))

    # Add red shaded regions for coverage gaps
    gaps = m.get("coverage_gaps", [])
    for gap in gaps:
        g_start_min = gap["start"] / 60.0
        g_end_min = (gap["start"] + gap["duration"]) / 60.0
        x0_label = "%d:00" % int(g_start_min)
        x1_label = "%d:00" % int(g_end_min)
        x0_idx = sorted_mins.index(int(g_start_min)) if int(g_start_min) in sorted_mins else None
        x1_idx = sorted_mins.index(int(g_end_min)) if int(g_end_min) in sorted_mins else x0_idx
        if x0_idx is not None:
            fig.add_vrect(
                x0=x0_idx - 0.5, x1=(x1_idx if x1_idx is not None else x0_idx) + 0.5,
                fillcolor="rgba(231,76,60,0.12)",
                line=dict(color="rgba(231,76,60,0.4)", width=1),
                layer="below",
                annotation_text="%ds gap" % gap["duration"],
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color="#e74c3c",
            )

    fig.update_layout(
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        yaxis=dict(
            title="% of minute",
            range=[0, 105],
            gridcolor="#e1e0d9",
            ticksuffix="%",
        ),
        xaxis=dict(title="Time (minutes)"),
        height=350,
        margin=dict(t=30, b=50, l=60, r=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
    )
    st.plotly_chart(fig, width="stretch")


@st.cache_data
def extract_frame(video_path, time_sec):
    """Extract a single frame from video at given timestamp.
    Uses the video's own FPS to seek correctly (works for both
    raw source and annotated output at different frame rates)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = min(int(time_sec * fps), total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


@st.cache_data
def extract_clip(video_path, start_sec, end_sec):
    """Extract a video clip using ffmpeg for browser-compatible H.264."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    duration = end_sec - start_sec

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = [
        ffmpeg, "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        tmp_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        pass

    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
        with open(tmp_path, "rb") as f:
            clip_bytes = f.read()
        os.unlink(tmp_path)
        return clip_bytes

    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
    return None


def resolve_annotated_video(video_name):
    """Find the annotated output video path — checks data mount then repo dir."""
    # Strip any extension (.mp4 or .avi) to get base name
    base_name = video_name
    for ext in (".mp4", ".avi", ".MP4", ".AVI"):
        if base_name.endswith(ext):
            base_name = base_name[: -len(ext)]
            break
    name_variants = [base_name, base_name.replace(" ", "_").lower()]
    suffixes = ["_moving", "_full_finetuned", ""]
    for base in [DATA_DIR, BASE_DIR]:
        for name in name_variants:
            for suffix in suffixes:
                path = os.path.join(base, SUMMARY_DIR, name + suffix + ".mp4")
                if os.path.exists(path):
                    return path
    for base in [DATA_DIR, BASE_DIR]:
        for name in name_variants:
            path = os.path.join(base, name + ".mp4")
            if os.path.exists(path):
                return path
    return os.path.join(DATA_DIR, SUMMARY_DIR, base_name + "_moving.mp4")


def render_gap_highlights(m, video_name, is_live=False, recording_path=None,
                          annotated_path=None):
    gaps = m.get("coverage_gaps", [])
    if not gaps:
        st.success("No significant coverage gaps found in this range.")
        return

    st.markdown('<p class="section-title">Coverage Gaps</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Periods where unchecked material was passing through '
        'but no worker was inspecting it. Select a gap to review.</p>',
        unsafe_allow_html=True,
    )

    gap_labels = [
        "Gap %d — %s to %s (%d sec)" % (i + 1, fmt_time(g["start"]),
         fmt_time(g["start"] + g["duration"]), g["duration"])
        for i, g in enumerate(gaps)
    ]

    selected_gap = st.selectbox("Select a gap to review", gap_labels, key="gap_select")
    gap_idx = gap_labels.index(selected_gap)
    gap = gaps[gap_idx]

    # Resolve video file — prefer annotated video (has zones/boxes),
    # fall back to raw recording, then offline annotated video
    if is_live:
        if annotated_path and os.path.exists(annotated_path):
            video_file = annotated_path
        elif recording_path and os.path.exists(recording_path):
            video_file = recording_path
        else:
            st.info("Recording not available for this session. "
                    "Use `--record --stream` flags to enable gap video replay.")
            return
    else:
        video_file = resolve_annotated_video(video_name)
        if not os.path.exists(video_file):
            st.info("Annotated video not found — clip playback unavailable.")
            return

    pad = 2
    MAX_CLIP_SEC = 90
    clip_start = max(0, gap["start"] - pad)
    raw_end = gap["start"] + gap["duration"] + pad
    clip_end = min(raw_end, clip_start + MAX_CLIP_SEC)
    truncated = clip_end < raw_end

    label = "Clip: %s → %s" % (fmt_time(clip_start), fmt_time(clip_end))
    if truncated:
        label += " (first %ds of %ds gap shown)" % (MAX_CLIP_SEC - pad, gap["duration"])
    else:
        label += " (includes %ds padding before & after)" % pad
    st.markdown(
        '<div style="padding:8px 0 4px 0; font-size:0.85rem; opacity:0.65;">%s</div>' % label,
        unsafe_allow_html=True,
    )

    clip_bytes = extract_clip(video_file, clip_start, clip_end)
    if clip_bytes:
        st.video(clip_bytes, format="video/mp4")
    else:
        st.warning("Could not extract clip from video.")


def render_coverage_trend(m):
    st.markdown('<p class="section-title">How Coverage Changed Over Time</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Cumulative inspection rate as material passes through. '
        'A falling line means more zones are being missed in that period.</p>',
        unsafe_allow_html=True,
    )

    secs = m["secs_in_range"]
    zones = m["zones_in_range"]
    if not secs or not zones:
        st.info("No zones completed in selected range.")
        return

    # Sort zones by retire_time once, then walk incrementally — O(n log n) not O(n²)
    sorted_zones = sorted(zones, key=lambda z: z["retire_time"])
    zone_idx = 0
    total_retired = 0
    total_picked = 0

    # Sample every 60s for long videos, every 1s for short ones
    duration_range = secs[-1]["t"] - secs[0]["t"] if secs else 0
    step = 60 if duration_range > 3600 else (10 if duration_range > 600 else 1)
    sampled_secs = [s for i, s in enumerate(secs) if i % step == 0]
    if secs[-1] not in sampled_secs:
        sampled_secs.append(secs[-1])

    cov_times = []
    cov_vals = []
    zone_counts = []
    for s in sampled_secs:
        t = s["t"]
        while zone_idx < len(sorted_zones) and sorted_zones[zone_idx]["retire_time"] <= t:
            total_retired += 1
            if sorted_zones[zone_idx]["picked"]:
                total_picked += 1
            zone_idx += 1
        pct = round(total_picked / total_retired * 100, 1) if total_retired > 0 else 0
        cov_times.append(t)
        cov_vals.append(pct)
        zone_counts.append(total_retired)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cov_times, y=cov_vals,
        mode="lines",
        line=dict(color="#2ecc71", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(46,204,113,0.08)",
        hovertemplate="%{text}<extra></extra>",
        text=["%s — %.1f%% coverage (%d zones)" % (fmt_time(t), v, c)
              for t, v, c in zip(cov_times, cov_vals, zone_counts)],
    ))

    fig.add_hline(y=80, line_dash="dot", line_color="#999",
                  annotation_text="80% target", annotation_position="top left")

    max_t = max(cov_times) if cov_times else 0
    tick_interval = 3600 if max_t > 7200 else (1800 if max_t > 3600 else (600 if max_t > 1200 else 60))
    tick_vals = list(range(0, max_t + 1, tick_interval))

    fig.update_layout(
        yaxis=dict(title="Coverage %", range=[0, 105]),
        xaxis=dict(
            title="Time",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=[fmt_time(t) for t in tick_vals],
        ),
        height=300,
        margin=dict(t=10, b=40, l=60, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def render_miss_accountability(m):
    st.markdown('<p class="section-title">Miss Accountability</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Distinguishes zones missed while workers were present (accountability) '
        'from zones missed due to worker absence. Side-aware: near-side workers own L coverage, '
        'far-side workers own R coverage.</p>',
        unsafe_allow_html=True,
    )

    if m["missed_zones"] == 0:
        st.success("No missed zones in this range.")
        return

    if not m["split_zones"]:
        col1, col2 = st.columns(2)
        pm = m["present_misses"]
        am = m["absent_misses"]
        total_m = pm + am
        with col1:
            pct = round(pm / total_m * 100, 1) if total_m > 0 else 0
            st.markdown(
                '<div class="kpi-card" style="border-color:#e67e22">'
                '<p class="value" style="color:#e67e22">%d</p>'
                '<p class="label">Present but Missed</p>'
                '<p class="sub">%.1f%% of misses — workers were at belt</p>'
                '</div>' % (pm, pct),
                unsafe_allow_html=True,
            )
        with col2:
            pct = round(am / total_m * 100, 1) if total_m > 0 else 0
            st.markdown(
                '<div class="kpi-card" style="border-color:#95a5a6">'
                '<p class="value" style="color:#95a5a6">%d</p>'
                '<p class="label">Absent — No Worker</p>'
                '<p class="sub">%.1f%% of misses — no worker at belt</p>'
                '</div>' % (am, pct),
                unsafe_allow_html=True,
            )
        return

    n_w = m["n_workers"]
    near_ids = ", ".join("W%d" % i for i in range(n_w // 2))
    far_ids = ", ".join("W%d" % i for i in range(n_w // 2, n_w))

    fig = go.Figure()

    for side, label, workers_label, picked, pm, am, color in [
        ("L", "Near Side (L)", near_ids,
         m["l_picked"], m["l_present_miss"], m["l_absent_miss"], "#3498db"),
        ("R", "Far Side (R)", far_ids,
         m["r_picked"], m["r_present_miss"], m["r_absent_miss"], "#9b59b6"),
    ]:
        total = picked + pm + am
        if total == 0:
            continue
        fig.add_trace(go.Bar(
            name="Picked", x=[label], y=[picked],
            marker_color="#2ecc71", showlegend=(side == "L"),
            hovertemplate="%s: %d picked (%.1f%%)<extra></extra>" % (
                label, picked, picked / total * 100),
        ))
        fig.add_trace(go.Bar(
            name="Missed (workers present)", x=[label], y=[pm],
            marker_color="#e67e22", showlegend=(side == "L"),
            hovertemplate="%s: %d missed while %s present<extra></extra>" % (
                label, pm, workers_label),
        ))
        fig.add_trace(go.Bar(
            name="Missed (no worker)", x=[label], y=[am],
            marker_color="#e74c3c", showlegend=(side == "L"),
            hovertemplate="%s: %d missed — no %s worker present<extra></extra>" % (
                label, am, workers_label),
        ))

    fig.update_layout(
        barmode="stack",
        yaxis=dict(title="Zones"),
        height=350,
        margin=dict(t=30, b=40, l=60, r=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=12),
        ),
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    for col, side_label, workers_label, picked, pm, am in [
        (col1, "Near Side (L)", near_ids,
         m["l_picked"], m["l_present_miss"], m["l_absent_miss"]),
        (col2, "Far Side (R)", far_ids,
         m["r_picked"], m["r_present_miss"], m["r_absent_miss"]),
    ]:
        total = picked + pm + am
        cov_pct = round(picked / total * 100, 1) if total > 0 else 0
        total_missed = pm + am
        with col:
            st.markdown(
                '<div style="padding:8px 12px; border-radius:8px; '
                'border:1px solid rgba(128,128,128,0.25);">'
                '<div style="font-weight:600; margin-bottom:8px;">%s '
                '<span style="opacity:0.5; font-size:0.8rem">(%s)</span></div>'
                '<div class="worker-row"><span class="wlabel">Coverage</span>'
                '<span class="wvalue">%.1f%%</span></div>'
                '<div class="worker-row"><span class="wlabel">Zones picked</span>'
                '<span class="wvalue" style="color:#2ecc71">%d</span></div>'
                '<div class="worker-row"><span class="wlabel">Missed total</span>'
                '<span class="wvalue" style="color:#e74c3c">%d</span></div>'
                '<div class="worker-row"><span class="wlabel">&nbsp;&nbsp;Workers present</span>'
                '<span class="wvalue" style="color:#e67e22">%d</span></div>'
                '<div class="worker-row"><span class="wlabel">&nbsp;&nbsp;No worker present</span>'
                '<span class="wvalue" style="color:#95a5a6">%d</span></div>'
                '</div>' % (side_label, workers_label, cov_pct, picked,
                            total_missed, pm, am),
                unsafe_allow_html=True,
            )


def render_zone_breakdown(m):
    st.markdown('<p class="section-title">Zone Inspection Breakdown</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Who checked what — shows how the inspection workload '
        'was split between workers.</p>',
        unsafe_allow_html=True,
    )

    n_w = m["n_workers"]
    labels = []
    values = []
    colors = []
    for wid in range(n_w):
        labels.append("W%d only" % wid)
        values.append(m["w%d_exclusive" % wid])
        colors.append(WORKER_COLORS[wid % len(WORKER_COLORS)])
    labels.append("Multiple" if n_w > 2 else "Both workers")
    values.append(m["multi_zones"])
    colors.append("#2ecc71")
    labels.append("Not inspected")
    values.append(m["missed_zones"])
    colors.append("#e74c3c")

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="label+value",
        textfont_size=11,
        textposition="outside",
        hovertemplate="%{label}: %{value} zones (%{percent})<extra></extra>",
    )])
    fig.update_layout(
        height=380,
        margin=dict(t=40, b=40, l=40, r=40),
        annotations=[dict(
            text="%d<br>zones" % m["total_zones"],
            x=0.5, y=0.5, font_size=18, showarrow=False,
        )],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, width="stretch")


def main():
    summaries = load_summaries()
    live_sessions = load_live_sessions()
    all_sources = {}
    all_sources.update(summaries)
    all_sources.update(live_sessions)

    if not all_sources:
        st.error("No data found. Run the pipeline or start a live session first.")
        return

    # --- Sidebar ---
    st.sidebar.markdown("## ITC Belt Monitor")
    st.sidebar.markdown("---")
    source_names = list(all_sources.keys())
    selected = st.sidebar.selectbox("Select Source", source_names)

    data = all_sources[selected]
    is_live = data.get("_live", False)

    if "_last_refresh" not in st.session_state:
        st.session_state["_last_refresh"] = datetime.now(IST)

    if is_live:
        st.sidebar.markdown(":red_circle: **LIVE**")
        refresh_options = {"5s": 5, "10s": 10, "30s": 30, "Off": 0}
        refresh_label = st.sidebar.selectbox(
            "Auto Refresh", list(refresh_options.keys()), index=1,
            key="auto_refresh_interval")
        refresh_sec = refresh_options[refresh_label]
        if st.sidebar.button("Refresh Data", key="live_refresh", use_container_width=True):
            st.session_state["_last_refresh"] = datetime.now(IST)
            st.rerun()
        last = st.session_state["_last_refresh"]
        now = datetime.now(IST)
        ago_sec = int((now - last).total_seconds())
        if ago_sec < 60:
            ago_text = "%d sec ago" % ago_sec
        else:
            ago_text = "%d min ago" % (ago_sec // 60)
        st.sidebar.caption("Last refreshed at %s IST (%s)" % (
            last.strftime("%I:%M:%S %p"), ago_text))
        auto_js = ""
        if refresh_sec > 0:
            auto_js = """
            setTimeout(function() {
                var buttons = window.parent.document.querySelectorAll('button');
                for (var i = 0; i < buttons.length; i++) {
                    if (buttons[i].textContent.trim() === 'Refresh Data') {
                        buttons[i].click();
                        break;
                    }
                }
            }, %d);
            """ % (refresh_sec * 1000)
        components.html("""
        <script>
        document.addEventListener('visibilitychange', function() {
            if (!document.hidden) {
                var buttons = window.parent.document.querySelectorAll('button');
                for (var i = 0; i < buttons.length; i++) {
                    if (buttons[i].textContent.trim() === 'Refresh Data') {
                        buttons[i].click();
                        break;
                    }
                }
            }
        });
        %s
        </script>
        """ % auto_js, height=0)
    else:
        if st.sidebar.button("Refresh Data"):
            st.cache_data.clear()
            st.session_state["_last_refresh"] = datetime.now(IST)
            st.rerun()
        last = st.session_state["_last_refresh"]
        now = datetime.now(IST)
        ago_sec = int((now - last).total_seconds())
        if ago_sec < 60:
            ago_text = "%d sec ago" % ago_sec
        else:
            ago_text = "%d min ago" % (ago_sec // 60)
        st.sidebar.caption("Last refreshed at %s IST (%s)" % (
            last.strftime("%I:%M:%S %p"), ago_text))

    duration = data.get("duration_seconds", 0)

    if st.session_state.get("_last_video") != selected:
        for k in ["from_h", "from_m", "from_s", "to_h", "to_m", "to_s"]:
            st.session_state.pop(k, None)
        st.session_state["_last_video"] = selected

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Time Range Filter**")

    use_hours = duration >= 3600
    dur_h, dur_rem = divmod(duration, 3600)
    dur_m, dur_s = divmod(dur_rem if use_hours else duration, 60)
    from_max = duration - 1

    if use_hours:
        from_max_h, from_rem = divmod(from_max, 3600)
        from_max_m, from_max_s = divmod(from_rem, 60)

        st.sidebar.caption("**From**")
        fc1, fc2, fc3 = st.sidebar.columns(3)
        with fc1:
            from_h = st.selectbox("hr", list(range(0, int(dur_h) + 1)), index=0, key="from_h",
                                  format_func=lambda x: "%dh" % x, label_visibility="collapsed")
        from_min_max = int(from_max_m) if from_h == int(from_max_h) else 59
        with fc2:
            from_m = st.selectbox("min", list(range(0, from_min_max + 1)), index=0, key="from_m",
                                  format_func=lambda x: "%02dm" % x, label_visibility="collapsed")
        from_sec_max = int(from_max_s) if (from_h == int(from_max_h) and from_m == int(from_max_m)) else 59
        with fc3:
            from_s = st.selectbox("sec", list(range(0, from_sec_max + 1)), index=0, key="from_s",
                                  format_func=lambda x: "%02ds" % x, label_visibility="collapsed")

        st.sidebar.caption("**To**")
        tc1, tc2, tc3 = st.sidebar.columns(3)
        with tc1:
            to_h = st.selectbox("hr", list(range(0, int(dur_h) + 1)), index=int(dur_h), key="to_h",
                                format_func=lambda x: "%dh" % x, label_visibility="collapsed")
        to_min_max = int(dur_m) if to_h == int(dur_h) else 59
        with tc2:
            to_m = st.selectbox("min", list(range(0, to_min_max + 1)),
                                index=min(int(dur_m), to_min_max), key="to_m",
                                format_func=lambda x: "%02dm" % x, label_visibility="collapsed")
        to_sec_max = int(dur_s) if (to_h == int(dur_h) and to_m == int(dur_m)) else 59
        to_s_default = min(int(dur_s), to_sec_max) if (to_h == int(dur_h) and to_m == int(dur_m)) else 0
        with tc3:
            to_s = st.selectbox("sec", list(range(0, to_sec_max + 1)), index=to_s_default, key="to_s",
                                format_func=lambda x: "%02ds" % x, label_visibility="collapsed")

        t_start = max(0, min(from_h * 3600 + from_m * 60 + from_s, duration))
        t_end = max(0, min(to_h * 3600 + to_m * 60 + to_s, duration))
    else:
        from_max_m, from_max_s = divmod(from_max, 60)
        min_options = list(range(0, int(dur_m) + 1))
        from_min_options = list(range(0, int(from_max_m) + 1))

        st.sidebar.caption("**From**")
        fc1, fc2 = st.sidebar.columns(2)
        with fc1:
            from_m = st.selectbox("min", from_min_options, index=0, key="from_m",
                                  format_func=lambda x: "%d min" % x, label_visibility="collapsed")
        from_sec_max = int(from_max_s) if from_m == int(from_max_m) else 59
        with fc2:
            from_s = st.selectbox("sec", list(range(0, from_sec_max + 1)), index=0, key="from_s",
                                  format_func=lambda x: "%02d sec" % x, label_visibility="collapsed")

        st.sidebar.caption("**To**")
        tc1, tc2 = st.sidebar.columns(2)
        with tc1:
            to_m = st.selectbox("min", min_options, index=len(min_options) - 1, key="to_m",
                                format_func=lambda x: "%d min" % x, label_visibility="collapsed")
        to_sec_max = int(dur_s) if to_m == int(dur_m) else 59
        to_s_default = min(int(dur_s), to_sec_max) if to_m == int(dur_m) else 0
        with tc2:
            to_s = st.selectbox("sec", list(range(0, to_sec_max + 1)), index=to_s_default, key="to_s",
                                format_func=lambda x: "%02d sec" % x, label_visibility="collapsed")

        t_start = max(0, min(from_m * 60 + from_s, duration))
        t_end = max(0, min(to_m * 60 + to_s, duration))

    if t_start > t_end:
        t_start, t_end = t_end, t_start

    st.sidebar.caption("Showing **%s** to **%s** (%s)" % (
        fmt_time(t_start), fmt_time(t_end), fmt_time(t_end - t_start)))
    st.sidebar.markdown("---")

    m = filter_data(data, t_start, t_end)

    # --- Header ---
    st.markdown("## Belt Inspection Report")
    st.caption("%s  |  %s to %s  (%s of %s)" % (
        selected, fmt_time(t_start), fmt_time(t_end),
        fmt_time(t_end - t_start), fmt_time(duration)))

    # Live annotated video feed
    if is_live:
        stream_port = int(os.environ.get("STREAM_PORT", "8503"))
        stream_url = "http://localhost:%d/live" % stream_port
        st.markdown(
            '<p class="section-title">Live Processing Feed</p>',
            unsafe_allow_html=True)
        fullscreen_url = "http://localhost:%d/" % stream_port
        st.markdown(
            '<div style="border-radius:8px;overflow:hidden;'
            'border:2px solid #333;">'
            '<img src="%s" style="width:100%%;height:auto;display:block;" '
            'alt="Live annotated feed from pipeline" />'
            '</div>'
            '<p style="font-size:0.75rem;opacity:0.5;margin:4px 0 0 0;">'
            'Direct link: <a href="%s" target="_blank">%s</a> · '
            '<a href="%s" target="_blank">Fullscreen</a></p>'
            % (stream_url, stream_url, stream_url, fullscreen_url),
            unsafe_allow_html=True)

    st.markdown("---")

    # Row 1: KPI cards
    render_kpis(m)
    st.markdown("")

    # Row 2: Worker cards + Zone donut side by side
    col_left, col_right = st.columns([3, 2])
    with col_left:
        render_workers(m)
    with col_right:
        render_zone_breakdown(m)

    st.markdown("---")

    # Row 3: Miss accountability
    render_miss_accountability(m)

    st.markdown("---")

    # Row 4: Inspection timeline
    render_activity_strip(m)

    st.markdown("---")

    # Row 4: Coverage gaps with video clip player
    video_name = data.get("video", "")
    for _ext in (".mp4", ".avi", ".MP4", ".AVI"):
        video_name = video_name.replace(_ext, "")
    render_gap_highlights(m, video_name, is_live=is_live,
                          recording_path=data.get("_recording_path"),
                          annotated_path=data.get("_annotated_path"))

    st.markdown("---")

    # Row 5: Coverage trend
    render_coverage_trend(m)


if __name__ == "__main__":
    main()
