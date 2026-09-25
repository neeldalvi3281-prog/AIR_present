"""Presenter Controller developer preview (DEBUG only, never benchmark numbers).

Reform interaction model: whole-hand palm mouse (always) + palm-rotation
page navigation (clockwise = PREVIOUS, anti-clockwise = NEXT) + pinch
hold-click (thumb+index). Hold the YO sign (index+pinky out, middle+ring
folded, thumb folded) and rotate: clockwise twists zoom in,
anti-clockwise twists zoom out (Ctrl+wheel ticks). Hold the peace sign
(index+middle out, rest folded, thumb folded) pointed right to scroll
up, left to scroll down (plain wheel ticks). Anything else resets zoom
and scroll; while YO/PEACE holds, nav and clicks are silent.

Usage:
    python app.py --preview                # run until Q/ESC
    python app.py --preview --preview-seconds 10   # auto-exit after 10 s

Controls: Q or ESC = close, P = toggle overlay.
Benchmark stays headless: python tools/benchmark.py (drawing excluded).

Safety: real input is sent only when config action.enabled=true AND
action.dry_run=false (LIVE mode). Defaults are safe: nothing is sent.
"""
import argparse
import os
import sys
import time

import cv2
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera.capture import CameraManager
from config import load_config
from performance.profiler import Profiler
from ui.developer import (build_action_lines, build_lines,
                          build_palm_lines, build_pinch_lines,
                          build_rotation_lines, build_scroll_lines,
                          build_zoom_lines, draw)
from vision.features import FrameFeatures, derive
from vision.filters import Filter2D
from vision.hand_tracker import HandTracker
from vision.palm import palm_angle_deg, palm_centroid
from vision.pinch import PinchTracker
from vision.poses import PEACE, YO, classify_pose, peace_direction
from vision.peace_scroll import PeaceScrollTracker
from vision.rotation import RotationTracker, rotation_frame
from vision.yo_zoom import YoZoomTracker
from actions.dispatcher import ActionDispatcher
from actions.keyboard import KeyboardAdapter
from actions.mouse import MouseAdapter, apply_control_box

WINDOW = "Presenter Controller - Developer Preview (Q/ESC quit, P overlay)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true",
                    help="launch developer preview window")
    ap.add_argument("--preview-seconds", type=float, default=0,
                    help="auto-exit after N seconds (0 = until Q/ESC)")
    args = ap.parse_args()
    cfg = load_config()
    cam = CameraManager(cfg["camera"]["device"], cfg["camera"]["width"],
                        cfg["camera"]["height"], cfg["camera"]["target_fps"])
    try:
        cam.start()
    except RuntimeError as e:
        print(f"CAMERA: UNAVAILABLE ({e})")
        return
    tracker = HandTracker(cfg["model"]["path"])
    rotation = RotationTracker(cfg["rotation"])
    pinch = PinchTracker(cfg["pinch"])
    yo_zoom = YoZoomTracker(cfg["zoom"])
    peace_scroll = PeaceScrollTracker(cfg["scroll"])
    palm_filter = Filter2D(min_cutoff=cfg["mouse"]["min_cutoff"],
                           beta=cfg["mouse"]["beta"],
                           d_cutoff=cfg["mouse"]["d_cutoff"])
    keyboard = KeyboardAdapter()
    mouse = MouseAdapter()
    dispatcher = ActionDispatcher(cfg["action"], keyboard, mouse=mouse)
    print(f"action control: enabled={cfg['action']['enabled']} "
          f"dry_run={cfg['action']['dry_run']} -> mode {dispatcher.mode()}")
    print(f"subsystems: mouse={cfg['mouse']['enabled']} "
          f"rotation={cfg['rotation']['enabled']} pinch={cfg['pinch']['enabled']} "
          f"zoom={cfg['zoom']['enabled']}")
    print("Focus requirement: the presentation app must be the active window.")
    print("Zoom: hold the YO sign and twist clockwise (in) / "
          "anti-clockwise (out).")
    print("Scroll: hold the peace sign pointed right (up) / left (down).")
    profiler = Profiler()
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime process-CPU baseline
    cpu, ram = 0.0, 0.0
    last_sysmon = 0.0
    last_seq = -1
    show_overlay = True
    ts_counter = [int(time.perf_counter() * 1000)]
    t_start = time.perf_counter()

    def next_ts():
        now = int(time.perf_counter() * 1000)
        if now <= ts_counter[0]:
            now = ts_counter[0] + 1
        ts_counter[0] = now
        return now

    try:
        while True:
            frame, cap_ts, seq = cam.read_latest()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            if key in (ord("p"), ord("P")):
                show_overlay = not show_overlay
            if frame is None or seq == last_seq:
                if args.preview_seconds and time.perf_counter() - t_start > args.preview_seconds:
                    break
                time.sleep(0.002)
                continue
            last_seq = seq
            # Mirror (selfie view) BEFORE detection so landmarks match display.
            # Mirror compensation for rotation lives in exactly one place:
            # vision.palm.MIRROR_SIGN (physical CW -> NEXT).
            frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            profiler.note_capture(cap_ts)
            t0 = time.perf_counter()
            det = tracker.detect(frame, next_ts())
            infer_s = time.perf_counter() - t0
            profiler.note_inference(now, det is not None, infer_s,
                                    time.perf_counter() - cap_ts)
            # System diagnostics at ~4 Hz (not per frame): process CPU +
            # RAM stay current on screen without per-frame syscall cost.
            if now - last_sysmon >= 0.25:
                cpu = proc.cpu_percent(interval=None)
                ram = proc.memory_info().rss / (1024 * 1024)
                profiler.note_system(cpu, ram)
                last_sysmon = now
            derived, conf, hand = None, 0.0, ""
            palm_norm, angle, screen_xy = None, None, (0.0, 0.0)
            pose, mouse_on = "UNKNOWN", False
            t0 = time.perf_counter()
            if det is not None:
                h, w = frame.shape[:2]
                lm = det["landmarks_norm"]
                # ONE validated bundle per frame: every gesture path below
                # shares its coercion + norms (opti §16), identical results.
                feats = FrameFeatures.from_landmarks(lm)
                derived = derive(feats, w, h) if show_overlay else None
                conf, hand = det["hand_confidence"], det["handedness"]
                # Independent paths from the same landmarks: filtered palm
                # centroid drives the mouse; RAW orientation drives rotation.
                palm_norm = palm_centroid(feats)
                angle = palm_angle_deg(feats)
                pose = classify_pose(feats, cfg["fist"])
                events = []
                if cfg["pinch"]["enabled"]:
                    events += pinch.update(feats, now)
                # Cursor first: mouse follows the palm in every pose.
                # Filter stays fed whenever the hand is present (even in
                # dry-run) so going live never starts from a stale position.
                if cfg["mouse"]["smoothing_enabled"]:
                    fx, fy = palm_filter.filter(palm_norm[0], palm_norm[1], now)
                else:
                    fx, fy = palm_norm
                screen_xy = apply_control_box(fx, fy, cfg["mouse"]["mapping"])
                if (cfg["mouse"]["enabled"]
                        and dispatcher.mode() == "LIVE"):
                    mouse.move(screen_xy[0], screen_xy[1])
                    mouse_on = True
                # Exclusive gesture sensing: while YO/PEACE holds, pinch
                # never dispatches (no click mid-zoom/scroll) and page nav
                # resets (rotation_frame only measures OPEN). Anything but
                # YO resets the zoom anchor; anything but PEACE resets the
                # scroll repeater — no stale comparisons.
                if pose == YO:
                    yo_zoom_active = True
                else:
                    yo_zoom_active = False
                    yo_zoom.reset()
                if pose == PEACE:
                    peace_active = True
                else:
                    peace_active = False
                    peace_scroll.reset()
                routed = []
                for ev in events:
                    if pose == YO or pose == PEACE:
                        continue
                    routed.append(ev)
                events = routed
                # Lifecycle in vision.rotation.rotation_frame: ONLY OPEN
                # measures; FIST/UNKNOWN/pinch-held reset; next OPEN = fresh
                # reference. Hand loss below also resets.
                events += rotation_frame(
                    rotation, pose, pinch.held, angle, now,
                    enabled=cfg["rotation"]["enabled"],
                    open_required=cfg["rotation"].get("open_required", True))
                # ZOOM rotation: feed the YO angle only while YO holds.
                if yo_zoom_active and cfg["zoom"]["enabled"]:
                    events += yo_zoom.update(angle, now)
                # SCROLL repeat: feed the peace direction only while PEACE
                # holds (right = up, left = down).
                if peace_active and cfg["scroll"]["enabled"]:
                    events += peace_scroll.update(peace_direction(feats), now)
                for event in events:
                    rec = dispatcher.dispatch(event, now)
                    if rec is not None:
                        print(f"[INFO] Gesture {rec.gesture}")
                        print(f"[INFO] Action {rec.action}")
                        print(f"[INFO] Input {rec.key}")
                        print(f"[INFO] Mode {rec.mode}")
            else:
                # Hand gone: no move, no click, no nav, no zoom, no scroll.
                # Reset temporal state so reappearance resumes clean (no
                # jump, no stale). A mid-hold disappearance yields
                # PINCH_RELEASED so a held button always releases.
                rotation.update(None, now)
                yo_zoom.update(None, now)
                peace_scroll.update(None, now)
                for event in pinch.update(None, now):
                    rec = dispatcher.dispatch(event, now)
                    if rec is not None:
                        print(f"[INFO] Gesture {rec.gesture}")
                        print(f"[INFO] Action {rec.action}")
                        print(f"[INFO] Input {rec.key}")
                        print(f"[INFO] Mode {rec.mode}")
                palm_filter.reset()
            ges_ms = (time.perf_counter() - t0) * 1000.0
            # Cached stats (median/p95 at most ~2 Hz); overlay text is
            # built only when visible. draw() skips rendering when
            # derived is None (overlay off).
            s = profiler.summary_cached()
            fh, fw = frame.shape[:2]
            lines = []
            control_box, palm_dot = None, None
            if show_overlay:
                # Fixed virtual control area (never follows the hand):
                # mapping fractions -> frame pixels, drawn every frame.
                mp = cfg["mouse"]["mapping"]
                control_box = (mp["left"] * fw, mp["top"] * fh,
                               mp["right"] * fw, mp["bottom"] * fh)
                if det is not None:
                    palm_dot = (fx * fw, fy * fh)
                # Hand gone -> TRACKING: NO HAND line, window stays open.
                lines = build_lines(det is not None, conf, hand, fw, fh,
                                    s["camera_fps"], s["inference_fps"],
                                    s["infer_latency_ms"]["mean"],
                                    s["infer_latency_ms"]["p95"],
                                    cpu, ram, camera_status="OK")
                lines += build_palm_lines(palm_norm, screen_xy,
                                          angle if angle is not None else 0.0,
                                          mouse_on, ges_ms=ges_ms)
                lines += build_rotation_lines(rotation)
                lines += build_pinch_lines(pinch, pose)
                lines += build_zoom_lines(yo_zoom)
                lines += build_scroll_lines(peace_scroll)
                lines += build_action_lines(dispatcher)
            draw(frame, derived, lines, angle_deg=angle,
                 control_box=control_box, palm_px=palm_dot)
            cv2.imshow(WINDOW, frame)
            if args.preview_seconds and time.perf_counter() - t_start > args.preview_seconds:
                break
    finally:
        cam.stop()
        tracker.close()
        mouse.release()  # never quit with the button stuck down
        keyboard.close()  # release any held key; shutdown must not stick keys
        cv2.destroyAllWindows()
    print("Preview profiler:", profiler.summary())


if __name__ == "__main__":
    main()
