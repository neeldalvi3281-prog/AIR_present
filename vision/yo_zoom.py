"""YO-gated rotation zoom: committed-angle steps while the rock-on holds.

Inspired by the page-nav rotation pattern (committed angle, confirm,
hysteresis, spacing) but a dedicated lean tracker: it measures rotation
RELATIVE to the angle committed on YO entry, never against a fixed pose.
Every `step_degrees` of physical rotation emits exactly one
ZOOM_IN_STEP (clockwise) / ZOOM_OUT_STEP (anticlockwise) and advances
the anchor, so continuous twisting zooms continuously with no
return-to-neutral. Anything but YO (or hand loss) resets — no stale
comparisons, no events from other poses.

Emits the existing ZOOM_IN_STEP / ZOOM_OUT_STEP strings so the
dispatcher Ctrl+wheel path is untouched.
"""
from vision.palm import MIRROR_SIGN, wrap_delta_deg

ZOOM_IN_STEP = "ZOOM_IN_STEP"
ZOOM_OUT_STEP = "ZOOM_OUT_STEP"


def _wrap180(deg):
    d = float(deg) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


class YoZoomTracker:
    """update(angle_or_None, now_s) -> list of events (0 or 1 per call).

    reset() clears the anchor (non-YO pose, hand loss).
    .anchor is the committed landmark angle; .delta is last physical
    spread in degrees; .last_event is the last emitted step (overlay).
    """

    def __init__(self, cfg, mirror_sign=MIRROR_SIGN):
        self.cfg = cfg
        self.mirror_sign = mirror_sign
        self.reset()

    def reset(self):
        self.anchor = None
        self.delta = 0.0
        self.last_event = "NONE"
        self._cand_dir = 0
        self._cand_count = 0
        self._spacing_until = 0.0
        self._filt = None

    @property
    def confirm_progress(self):
        need = self.cfg["confirm_frames"]
        return f"{min(self._cand_count, need)}/{need}"

    def _smooth(self, raw):
        if self._filt is None:
            self._filt = raw
            return raw
        d = wrap_delta_deg(self._filt, raw)
        if abs(d) >= 10.0:
            self._filt = raw
        else:
            self._filt = _wrap180(self._filt + 0.35 * d)
        return self._filt

    def update(self, angle_or_none, now_s):
        events = []
        if angle_or_none is None:
            self.reset()
            return events
        angle = self._smooth(float(angle_or_none))
        step = self.cfg["step_degrees"]
        need = self.cfg["confirm_frames"]
        hyst = self.cfg["hysteresis_degrees"]

        if self.anchor is None:
            self.anchor = angle
            self.delta = 0.0
            return events

        delta = self.mirror_sign * wrap_delta_deg(self.anchor, angle)
        self.delta = delta
        ad = abs(delta)
        d = 1 if delta > 0 else (-1 if delta < 0 else 0)

        if ad >= step:
            if d == self._cand_dir:
                self._cand_count += 1
            else:
                self._cand_dir = d
                self._cand_count = 1
        elif ad < step - hyst:
            self._cand_dir = 0
            self._cand_count = 0
        # Between (step-hyst, step): hold the count, no chatter.

        if (self._cand_count >= need and self._cand_dir != 0
                and now_s >= self._spacing_until):
            ev = ZOOM_IN_STEP if self._cand_dir > 0 else ZOOM_OUT_STEP
            self.last_event = ev
            events.append(ev)
            self.anchor = _wrap180(
                self.anchor + self.mirror_sign * self._cand_dir * step)
            self._spacing_until = (
                now_s + self.cfg["tick_spacing_ms"] / 1000.0)
            self._cand_dir = 0
            self._cand_count = 0
        return events
