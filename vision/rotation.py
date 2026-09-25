"""Incremental palm-rotation navigation: committed-angle 30-degree steps.

Measures rotation RELATIVE to a committed angle (the orientation of the
most recently consumed step), never against a fixed neutral pose. Every
consumed step_degrees of physical rotation emits exactly one event and
advances the committed angle by that step, so continuous rotation pages
continuously with NO return-to-neutral requirement. Direction reversal
works naturally: rotating back consumes steps the other way.

Reliability stack (false negatives preferred over false positives):
  wide-open-palm gating (OPEN only; FIST/UNKNOWN reset the session)
  + light angle EMA (snaps on real motion, smooths jitter only)
  + 4-frame confirmation (threshold must hold, same direction)
  + hysteresis band (dips just under the step hold, not cancel)
  + committed advance in LANDMARK space via mirror_sign * dir * step
  + step spacing (multi-step jumps drain one event per frame max)
  + hand-loss / fist reset (no stale orientation comparisons)

Translation cannot fire it: the input is a palm orientation angle
(landmark differences), not a screen position. Absolute hand angle is
irrelevant: committed starts wherever the open palm already points, so
an upright, horizontal, diagonal, or upside-down hand all navigate the
same once it turns 30 degrees relative to that start.

Sign convention: positive relative = physical clockwise = NEXT slide.
The tracker multiplies measured (mirrored-landmark) deltas by mirror_sign
(default vision.palm.MIRROR_SIGN); no other sign flips exist. Committed
is stored in LANDMARK angle space and advanced by
mirror_sign * physical_dir * step so the reference tracks the real
landmark motion (undoing the mirror exactly once).
"""
from vision.palm import MIRROR_SIGN, wrap_delta_deg

CLOCKWISE = "CLOCKWISE_ROTATION_CONFIRMED"
ANTICLOCKWISE = "ANTICLOCKWISE_ROTATION_CONFIRMED"

OPEN_POSE = "OPEN"

IDLE = "IDLE"
SESSION = "SESSION"

# ponytail: snap on real motion (>= SNAP_DEG), EMA only for jitter —
# a pure low-pass would lag forever below the step threshold.
_SNAP_DEG = 10.0
_JITTER_ALPHA = 0.35


def pose_allows_navigation(pose):
    """Only a wide-open palm may navigate. FIST clicks, UNKNOWN idles."""
    return pose == OPEN_POSE


def rotation_frame(rotation, pose, held, angle, now_s,
                   enabled=True, open_required=True):
    """One-frame lifecycle: ONLY a wide-open palm may measure or page.

    FIST, UNKNOWN, or held-button → full reset (candidate, committed,
    confirm counter). Next OPEN commits a fresh reference — a stationary
    fist at any angle can never generate navigation, and a prior fist's
    orientation never becomes a fake rotation.
    """
    if not enabled:
        return []
    if held or pose != OPEN_POSE:
        rotation.reset()
        return []
    return rotation.update(angle, now_s)


def _wrap180(deg):
    d = float(deg) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


class RotationTracker:
    """update(angle_deg_or_None, now_s) -> list of events (0 or 1 per call).

    cancel() drops an in-progress candidate but keeps the session (used
    for UNKNOWN poses); reset() clears everything (hand loss, fist).
    """

    def __init__(self, cfg, mirror_sign=MIRROR_SIGN):
        self.cfg = cfg
        self.mirror_sign = mirror_sign
        self.reset()

    def reset(self):
        self.state = IDLE
        self.committed = None
        self.delta = 0.0  # last physical delta vs committed, degrees
        self.direction = "—"
        self.last_event = "NONE"
        self._cand_dir = 0
        self._cand_count = 0
        self._spacing_until = 0.0
        self._filt = None  # smoothed landmark angle

    def cancel(self):
        """Drop the in-progress candidate, keep the committed session."""
        self._cand_dir = 0
        self._cand_count = 0
        self.direction = "—"

    @property
    def confirm_progress(self):
        """'n/need' stability string for the overlay."""
        return f"{min(self._cand_count, self.cfg['confirm_frames'])}/" \
            f"{self.cfg['confirm_frames']}"

    @property
    def next_clockwise(self):
        """Landmark angle of the next physical-CW threshold (overlay)."""
        if self.committed is None:
            return 0.0
        return _wrap180(self.committed
                        + self.mirror_sign * self.cfg["step_degrees"])

    @property
    def next_anticlockwise(self):
        if self.committed is None:
            return 0.0
        return _wrap180(self.committed
                        - self.mirror_sign * self.cfg["step_degrees"])

    @property
    def remaining(self):
        """Degrees still needed for the next step (overlay use)."""
        return max(self.cfg["step_degrees"] - abs(self.delta), 0.0)

    def _physical_delta(self, angle):
        return self.mirror_sign * wrap_delta_deg(self.committed, angle)

    def _smooth(self, raw):
        """Filter the angle (not the event): EMA on short-path delta,
        snap when the jump is large enough to be real rotation."""
        if self._filt is None:
            self._filt = raw
            return raw
        d = wrap_delta_deg(self._filt, raw)
        if abs(d) >= _SNAP_DEG:
            self._filt = raw
        else:
            self._filt = _wrap180(self._filt + _JITTER_ALPHA * d)
        return self._filt

    def update(self, angle_or_none, now_s):
        events = []
        if angle_or_none is None:
            self.reset()  # disappearance: no events, fresh session later
            return events
        angle = self._smooth(float(angle_or_none))
        step = self.cfg["step_degrees"]
        need = self.cfg["confirm_frames"]
        hyst = self.cfg["hysteresis_degrees"]

        if self.state == IDLE:
            self.committed = angle
            self.delta = 0.0
            self.state = SESSION
            return events

        delta = self._physical_delta(angle)
        self.delta = delta
        ad = abs(delta)
        d = 1 if delta > 0 else (-1 if delta < 0 else 0)
        self.direction = ("CLOCKWISE" if d > 0
                          else ("ANTICLOCKWISE" if d < 0 else "—"))

        if ad >= step:
            if d == self._cand_dir:
                self._cand_count += 1
            else:
                self._cand_dir = d
                self._cand_count = 1
        elif ad < step - hyst:
            # Clearly back inside: cancel. Between (step-hyst, step):
            # hold the count (hysteresis band, no chatter).
            self._cand_dir = 0
            self._cand_count = 0

        if (self._cand_count >= need and self._cand_dir != 0
                and now_s >= self._spacing_until):
            ev = CLOCKWISE if self._cand_dir > 0 else ANTICLOCKWISE
            self.last_event = ev
            events.append(ev)
            # Consume exactly one step in LANDMARK space: physical dir
            # was mirror-corrected into _cand_dir, so undo the mirror
            # when advancing committed or the reference runs away and
            # holding still keeps re-firing (and wrap flips direction).
            self.committed = _wrap180(
                self.committed + self.mirror_sign * self._cand_dir * step)
            self._spacing_until = (
                now_s + self.cfg["step_action_spacing_ms"] / 1000.0)
            self._cand_dir = 0
            self._cand_count = 0
        return events
