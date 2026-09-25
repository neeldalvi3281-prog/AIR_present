"""Pinch left-click: thumb-tip ↔ index-tip distance / hand scale.

States: OPEN -> CANDIDATE (start threshold) -> HELD (confirm_frames)
        CANDIDATE/HELD -> OPEN only when distance >= release threshold.

normalized = ||lm[4] - lm[8]|| / hand_scale_norm(lm)  — resolution and
distance-to-camera invariant; no palm coords, no pointer velocity.

Reliability:
  light EMA on distance (single landmark spikes do not reset confirm)
  + start < release hysteresis on BOTH enter and abort (0.30..0.37 noise
    never cancels a candidate — only crossing release does)
  + multi-frame confirm, post-release cooldown
  + hand-loss always emits one PINCH_RELEASED if held (never a stuck
    Windows button). No ML, no threads — pure landmark math per frame.
"""
import numpy as np

from vision.features import FrameFeatures
from vision.poses import TIPS, hand_scale_norm

OPEN = "OPEN"
CANDIDATE = "CANDIDATE"
HELD = "HELD"

PINCH_STARTED = "PINCH_STARTED"
PINCH_RELEASED = "PINCH_RELEASED"

# ponytail: light EMA only — enough to kill tip jitter, not enough to lag
# a deliberate pinch (~100 ms). Snap when the jump is large (real pinch).
_SMOOTH_ALPHA = 0.5
_SNAP_DELTA = 0.25


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def pinch_distance_norm(lm):
    """||index tip - thumb tip|| / hand scale (wrist -> middle MCP)."""
    if isinstance(lm, FrameFeatures):
        return lm.pinch_raw
    lm = np.asarray(lm, dtype=np.float64).reshape(21, 2)
    return _dist(lm[TIPS["index"]], lm[TIPS["thumb"]]) / hand_scale_norm(lm)


class PinchTracker:
    """update(lm_or_None, now_s) -> list of events (0 or 1).

    .state is OPEN/CANDIDATE/HELD; .held mirrors logical button-down;
    .distance is the last raw normalized pinch distance (overlay);
    .distance_smooth is the filtered value the state machine uses.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = OPEN
        self.held = False
        self.distance = 1.0
        self.distance_smooth = 1.0
        self.last_event = "NONE"
        self._cand_n = 0
        self._filt = None
        self._cooldown_until = 0.0

    @property
    def confirm_progress(self):
        need = self.cfg["confirm_frames"]
        return f"{min(self._cand_n, need)}/{need}"

    def _smooth(self, raw):
        if self._filt is None:
            self._filt = raw
            return raw
        d = raw - self._filt
        if abs(d) >= _SNAP_DELTA:
            self._filt = raw
        else:
            self._filt = self._filt + _SMOOTH_ALPHA * d
        return self._filt

    def update(self, lm_or_none, now_s):
        events = []
        if lm_or_none is None:
            if self.held:
                self.reset()
                self._cooldown_until = (
                    now_s + self.cfg["click_cooldown_ms"] / 1000.0)
                self.last_event = PINCH_RELEASED
                events.append(PINCH_RELEASED)
            else:
                self.reset()
            return events

        raw = pinch_distance_norm(lm_or_none)
        self.distance = raw
        d = self._smooth(raw)
        self.distance_smooth = d
        start = self.cfg["start_threshold"]
        release = self.cfg["release_threshold"]
        need = self.cfg["confirm_frames"]

        # Release on RAW (crisp UP; EMA would lag past 0.38).
        # Enter/confirm on smoothed (kills tip jitter).
        # Exit only past release: jitter inside [start, release) must not
        # cancel or re-open. Cooldown only after a real HELD release —
        # never while merely open (open-hand d is always >= release and
        # would otherwise arm cooldown every frame, blocking the next pinch).
        if raw >= release:
            if self.held:
                self.reset()
                self._filt = raw  # keep filter continuous across release
                self._cooldown_until = (
                    now_s + self.cfg["click_cooldown_ms"] / 1000.0)
                self.last_event = PINCH_RELEASED
                events.append(PINCH_RELEASED)
            else:
                self._cand_n = 0
                self.state = OPEN
                self._filt = raw
            return events

        if self.held:
            # Still pinched (band or deep): stay HELD, no state change.
            return events

        if d <= start:
            self._cand_n += 1
            self.state = CANDIDATE
            if self._cand_n >= need and now_s >= self._cooldown_until:
                self.state = HELD
                self.held = True
                self.last_event = PINCH_STARTED
                events.append(PINCH_STARTED)
        elif self.state == CANDIDATE:
            # Inside hysteresis band: hold the count (no cancel).
            pass
        return events
