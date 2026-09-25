"""Developer preview overlay (OpenCV only — most resource-efficient).

Draws: 21 landmarks, hand connections, bbox, palm center, palm orientation
vector, fixed mouse-control box, metrics, status.
"""
import cv2

TEXT_COLOR = (0, 255, 0)
HAND_COLOR = (0, 255, 0)
CONN_COLOR = (0, 200, 0)
BOX_COLOR = (255, 0, 0)
PALM_COLOR = (0, 255, 255)
ORIENT_COLOR = (255, 0, 255)
CONTROL_COLOR = (0, 165, 255)

# MediaPipe hand topology (21 pts): fingers + palm links.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def draw(frame, derived, lines, angle_deg=None, control_box=None,
         palm_px=None):
    """Draw hand overlay + metric lines onto frame (in place, returns it).

    angle_deg: palm orientation (vision.palm); draws the wrist->middle-MCP
    direction arrow so rotation direction is debuggable on screen.
    control_box: (x0, y0, x1, y1) frame-pixel rect of the fixed virtual
    mouse-control area (drawn only when given; never follows the hand).
    palm_px: (x, y) pixel position of the filtered palm anchor dot.
    """
    if derived is not None:
        pts = derived["landmarks_px"].astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, (int(pts[a][0]), int(pts[a][1])),
                     (int(pts[b][0]), int(pts[b][1])), CONN_COLOR, 1)
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 3, HAND_COLOR, -1)
        x0, y0, x1, y1 = derived["bbox"]
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), BOX_COLOR, 2)
        px, py = derived["palm_center"]
        cv2.circle(frame, (int(px), int(py)), 5, PALM_COLOR, -1)
        wx, wy = derived["wrist"]
        cv2.circle(frame, (int(wx), int(wy)), 5, (0, 0, 255), 2)
        if angle_deg is not None:
            import math
            rad = math.radians(angle_deg)
            leng = max(derived["hand_scale"], 1.0) * 1.2
            ex = int(px + math.sin(rad) * leng)
            ey = int(py - math.cos(rad) * leng)
            cv2.arrowedLine(frame, (int(px), int(py)), (ex, ey),
                            ORIENT_COLOR, 2, tipLength=0.25)
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, TEXT_COLOR, 1, cv2.LINE_AA)
        y += 20
    if control_box is not None:
        x0, y0, x1, y1 = (int(v) for v in control_box)
        cv2.rectangle(frame, (x0, y0), (x1, y1), CONTROL_COLOR, 2)
        cv2.putText(frame, "MOUSE CONTROL AREA", (x0 + 6, max(y0 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CONTROL_COLOR, 1,
                    cv2.LINE_AA)
    if palm_px is not None:
        cv2.circle(frame, (int(palm_px[0]), int(palm_px[1])), 6,
                   CONTROL_COLOR, 2)
    return frame


def build_lines(detected, confidence, handedness, width, height,
                cam_fps, infer_fps, infer_mean_ms, infer_p95_ms,
                cpu, ram, camera_status="OK"):
    """All Phase-1 preview lines. detected=False -> TRACKING: NO HAND."""
    track = "DETECTED" if detected else "NO HAND"
    return [
        f"CAMERA: {camera_status}",
        f"TRACKING: {track}",
        f"hand {'detected' if detected else 'not detected'} "
        f"{confidence:.2f} {handedness}".strip(),
        f"res {width}x{height}",
        f"camera FPS {cam_fps:.1f}  inference FPS {infer_fps:.1f}",
        f"infer {infer_mean_ms:.1f} ms  p95 {infer_p95_ms:.1f} ms",
        f"CPU {cpu:.1f}%  RAM {ram:.0f} MB",
    ]


def metric_lines(cam_fps, infer_fps, infer_ms, cpu, ram, confidence, handedness=""):
    """Back-compat wrapper (mean-only)."""
    return build_lines(bool(confidence > 0), confidence, handedness,
                       0, 0, cam_fps, infer_fps, infer_ms, infer_ms, cpu, ram)


def build_palm_lines(palm_norm, screen_xy, angle_deg, mouse_on, ges_ms=0.0):
    """Whole-hand mouse panel: palm anchor, mapped + absolute screen target."""
    if palm_norm is None:
        return ["Palm: --  Screen: --"]
    sx, sy = screen_xy
    ax, ay = int(sx * 65535), int(sy * 65535)
    return [
        f"Palm X = {palm_norm[0]:.3f}  Y = {palm_norm[1]:.3f}  "
        f"angle {angle_deg:+.1f}deg",
        f"Screen X = {ax}  Y = {ay}  Mouse: {'ON' if mouse_on else 'OFF'}  "
        f"ges {ges_ms:.2f}ms",
    ]


def build_rotation_lines(rotation):
    """Step-model panel: committed, delta, next threshold, remaining."""
    if rotation.committed is None:
        return ["Rotation: IDLE  committed --", "Last rotation: NONE"]
    nxt = (rotation.next_clockwise if rotation.delta >= 0
           else rotation.next_anticlockwise)
    return [
        f"Committed {rotation.committed:+.1f}deg  "
        f"Delta {rotation.delta:+.1f}deg",
        f"Next threshold: {nxt:+.1f}deg  Remaining: "
        f"{rotation.remaining:.1f}deg",
        f"Rotation direction: {rotation.direction}  "
        f"stability {rotation.confirm_progress}",
        f"Last rotation: {rotation.last_event}",
    ]


def build_pinch_lines(pinch, pose="UNKNOWN"):
    """Pose + pinch state machine + mouse button (left-click path)."""
    mouse = "LEFT DOWN" if pinch.held else "UP"
    return [
        f"Pose: {pose}",
        f"Pinch: {pinch.distance:.3f}  smooth {pinch.distance_smooth:.3f}  "
        f"state: {pinch.state}  ({pinch.confirm_progress})",
        f"Mouse: {mouse}  last: {pinch.last_event}",
    ]


def build_zoom_lines(zoom):
    """Zoom panel: committed YO angle anchor, relative twist, last step."""
    if zoom.anchor is None:
        return ["Zoom: IDLE  anchor --", "Last zoom: NONE"]
    return [
        f"Zoom anchor {zoom.anchor:+.1f}deg  twist {zoom.delta:+.1f}deg  "
        f"({zoom.confirm_progress})",
        f"Last zoom: {zoom.last_event}",
    ]


def build_scroll_lines(scroll):
    """Scroll panel: confirmed peace direction, last step."""
    if scroll.direction == 0:
        return ["Scroll: IDLE  dir --", "Last scroll: NONE"]
    arrow = "right/up" if scroll.direction > 0 else "left/down"
    return [
        f"Scroll dir {arrow}  ({scroll.confirm_progress})",
        f"Last scroll: {scroll.last_event}",
    ]


def build_action_lines(dispatcher):
    """Action section: gesture/action/input/mode + last action."""
    rec = dispatcher.last_record
    if rec is None:
        return [
            "GESTURE: NONE",
            f"MODE: {dispatcher.mode()}",
            "LAST ACTION: NONE",
        ]
    return [
        f"GESTURE: {rec.gesture}",
        f"ACTION: {rec.action}",
        f"INPUT: {rec.key}",
        f"MODE: {rec.mode}",
        f"LAST ACTION: {dispatcher.last_action}  {dispatcher.last_action_time}",
    ]
