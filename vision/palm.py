"""Whole-hand palm geometry from 21 normalized landmarks (reform §5-§11).

Anchor (mouse): centroid of wrist + 4 MCPs — the whole hand moves the
pointer, never the index fingertip alone.

Orientation (rotation): angle of the wrist -> MCP-centroid axis (mean of
index/middle/ring/pinky MCPs), 0 = fingers image-up, positive = clockwise
in image coordinates. Centroid damps single-landmark noise that caused
direction flips. 0 is only a readout label — any absolute hand angle is
valid; the rotation session commits whatever angle it starts at.
Uses landmark *differences* only, so it is translation- and scale-invariant
and identical for left/right hands sharing the same up axis.

Mirror: the preview frame is mirrored (selfie view) BEFORE detection, which
reverses apparent rotation — physical clockwise shows up as negative
(counter-clockwise) in landmark space. MIRROR_SIGN compensates exactly once,
here; rotation code multiplies measured deltas by it and must not invert
signs anywhere else.
"""
import math

import numpy as np

from vision.features import FrameFeatures

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
PALM_IDS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
MCP_IDS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

MIRROR_SIGN = -1


def _pts(lm):
    """Raw landmarks -> validated array; bundles share theirs (no copy)."""
    if isinstance(lm, FrameFeatures):
        return lm.pts
    return np.asarray(lm, dtype=np.float64).reshape(21, 2)


def palm_centroid(lm):
    """Mean of wrist + 4 MCPs, normalized [0,1] coords. Mouse anchor."""
    c = _pts(lm)[list(PALM_IDS)].mean(axis=0)
    return (float(c[0]), float(c[1]))


def palm_scale(lm):
    """Wrist->middle-MCP distance in normalized units (never ~0)."""
    if isinstance(lm, FrameFeatures):
        return lm.hand_scale
    pts = _pts(lm)
    return max(float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST])), 1e-6)


def palm_angle_deg(lm):
    """Palm orientation in degrees, (-180, 180]: 0 = fingers image-up,
    positive = clockwise (image coords, y down).

    Axis = wrist -> mean(4 MCPs). Raw geometry; temporal smoothing lives
    in RotationTracker (filter the angle, not the event).
    """
    if isinstance(lm, FrameFeatures):
        return lm.angle
    pts = _pts(lm)
    mcp_center = pts[list(MCP_IDS)].mean(axis=0)
    d = mcp_center - pts[WRIST]
    ang = math.degrees(math.atan2(d[0], -d[1]))
    if ang <= -180.0:
        ang += 360.0
    elif ang > 180.0:
        ang -= 360.0
    return ang


def wrap_delta_deg(from_deg, to_deg):
    """Wrapped signed difference (to - from) in (-180, 180].

    wrap(179, -179) = +2, not -358.
    """
    d = (float(to_deg) - float(from_deg)) % 360.0
    if d > 180.0:
        d -= 360.0
    return d
