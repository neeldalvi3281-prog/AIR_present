"""Cheap landmark-derived features (Phase 1 only, no gesture logic)."""
import math

import numpy as np

# Indices into the 21 MediaPipe hand landmarks.
WRIST = 0
PALM_IDS = (0, 5, 9, 13, 17)
MIDDLE_MCP = 9

# Finger landmark ids (mirrors vision.poses maps; kept local so this
# module imports nothing and every consumer can import it).
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_TIPS = (4, 8, 12, 16, 20)
_PIPS = (3, 6, 10, 14, 18)
_MCP_IDS = (5, 9, 13, 17)


class FrameFeatures:
    """One validated 21x2 array + every per-frame scalar the gesture
    modules need, computed exactly once.

    All consumer functions (poses/palm/pinch) accept raw landmarks or a
    FrameFeatures and return identical results: raw inputs are coerced
    through from_landmarks internally, while app.py builds ONE bundle
    per frame and passes it everywhere (single coercion, single norm
    set, zero-copy passthrough when the input is already a compliant
    float64 array).
    """

    __slots__ = ("pts", "wrist", "hand_scale", "mcp_centroid",
                 "palm_centroid", "angle", "pinch_raw", "peace_dx",
                 "thumb_out", "_tip_d", "_pip_d")

    def __init__(self, pts, wrist, hand_scale, mcp_centroid, palm_centroid,
                 angle, pinch_raw, peace_dx, thumb_out, tip_d, pip_d):
        self.pts = pts
        self.wrist = wrist
        self.hand_scale = hand_scale
        self.mcp_centroid = mcp_centroid
        self.palm_centroid = palm_centroid
        self.angle = angle
        self.pinch_raw = pinch_raw
        self.peace_dx = peace_dx
        self.thumb_out = thumb_out
        self._tip_d = tip_d
        self._pip_d = pip_d

    @classmethod
    def from_landmarks(cls, lm):
        a = np.asarray(lm, dtype=np.float64)
        # Zero-copy passthrough when already a compliant array.
        pts = a if a.shape == (21, 2) else a.reshape(21, 2)
        wrist = pts[WRIST]
        scale = max(float(np.linalg.norm(pts[MIDDLE_MCP] - wrist)), 1e-6)
        mcp = pts[list(_MCP_IDS)].mean(axis=0)
        palm = pts[list(PALM_IDS)].mean(axis=0)
        d = mcp - wrist
        ang = math.degrees(math.atan2(d[0], -d[1]))
        if ang <= -180.0:
            ang += 360.0
        elif ang > 180.0:
            ang -= 360.0
        tips = pts[list(_TIPS)] - wrist
        pips = pts[list(_PIPS)] - wrist
        tip_d = np.linalg.norm(tips, axis=1)
        pip_d = np.linalg.norm(pips, axis=1)
        thumb_out = (float(np.linalg.norm(pts[4] - pts[MIDDLE_MCP]))
                     / scale > 0.8)
        pinch_raw = (float(np.linalg.norm(pts[8] - pts[4])) / scale)
        peace_dx = ((pts[8][0] - pts[5][0]) / scale,
                    (pts[12][0] - pts[9][0]) / scale)
        return cls(pts, wrist, scale, mcp, palm, ang, pinch_raw, peace_dx,
                   thumb_out, tip_d, pip_d)

    def ext(self, finger, fold_ratio):
        """tip farther from wrist than PIP x ratio (same test as ever)."""
        i = _FINGERS.index(finger)
        return bool(self._tip_d[i] > self._pip_d[i] * fold_ratio)


def derive(normalized_landmarks, image_width, image_height):
    """Compute wrist, palm center, bbox, hand scale in pixel space.

    Args:
        normalized_landmarks: (21, 2) array of x/y in [0, 1], or a
            FrameFeatures (reuses its validated array, no re-coercion).
        image_width, image_height: frame size.

    Returns dict with wrist, palm_center, bbox, hand_scale, landmarks_px.
    hand_scale = wrist->middle-MCP distance (px), used later to normalize
    movement across distances/webcams (PRD hand-scale normalization).
    """
    if isinstance(normalized_landmarks, FrameFeatures):
        lm = normalized_landmarks.pts
    else:
        lm = np.asarray(normalized_landmarks, dtype=np.float64).reshape(21, 2)
    px = np.empty((21, 2), dtype=np.float64)
    px[:, 0] = lm[:, 0] * image_width
    px[:, 1] = lm[:, 1] * image_height

    wrist = px[WRIST].copy()
    # Palm center from 5 stable palm points (wrist + 4 MCPs): robust against
    # single-point jitter, unlike wrist-alone.
    palm_center = px[list(PALM_IDS)].mean(axis=0)
    palm_norm = tuple(float(v) for v in lm[list(PALM_IDS)].mean(axis=0))
    x0, y0 = px.min(axis=0)
    x1, y1 = px.max(axis=0)
    # Hand scale = wrist->middle-MCP distance (stable palm dimension, not a
    # noisy fingertip). Normalizes movement across camera distances/webcams.
    hand_scale = float(np.linalg.norm(px[MIDDLE_MCP] - px[WRIST]))
    hand_scale = max(hand_scale, 1e-6)  # avoid div-by-zero downstream

    return {
        "wrist": (float(wrist[0]), float(wrist[1])),
        "palm_center": (float(palm_center[0]), float(palm_center[1])),
        "palm_norm": palm_norm,
        "bbox": (float(x0), float(y0), float(x1), float(y1)),
        "hand_scale": hand_scale,
        "scale_norm": hand_scale / float(image_width),
        "landmarks_px": px,
    }
