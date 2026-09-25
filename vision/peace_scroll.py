"""Peace-hold scroll: repeated wheel ticks while the direction holds.

Hold-to-scroll (like holding an arrow key): the first tick needs
`confirm_frames` consecutive frames of the same direction (a passing
peace never scrolls); then one SCROLL_UP_STEP (+1, pointing right) /
SCROLL_DOWN_STEP (-1, pointing left) every `repeat_ms`. A direction
flip re-confirms from zero. Any None feed (other pose, hand loss)
resets silently — no events, no stuck repeat.

Pure counter math per frame; no ML, no threads.
"""

SCROLL_UP_STEP = "SCROLL_UP_STEP"
SCROLL_DOWN_STEP = "SCROLL_DOWN_STEP"


class PeaceScrollTracker:
    """update(direction_or_None, now_s) -> list of events (0 or 1 per call).

    direction is +1 (right/up), -1 (left/down), or None (reset).
    .direction is the confirmed direction (0 until confirmed);
    .last_event is the last emitted step (overlay).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.direction = 0
        self.last_event = "NONE"
        self._cand_dir = 0
        self._cand_count = 0
        self._next_at = 0.0

    @property
    def confirm_progress(self):
        need = self.cfg["confirm_frames"]
        return f"{min(self._cand_count, need)}/{need}"

    def update(self, direction_or_none, now_s):
        events = []
        if direction_or_none is None or direction_or_none == 0:
            self.reset()
            return events
        d = 1 if direction_or_none > 0 else -1
        need = self.cfg["confirm_frames"]
        repeat = self.cfg["repeat_ms"] / 1000.0

        if d == self._cand_dir:
            self._cand_count += 1
        else:
            # Fresh direction (or first sighting): re-confirm from one.
            self._cand_dir = d
            self._cand_count = 1
        if self._cand_count >= need and now_s >= self._next_at:
            ev = SCROLL_UP_STEP if d > 0 else SCROLL_DOWN_STEP
            self.last_event = ev
            events.append(ev)
            self.direction = d
            self._next_at = now_s + repeat
        return events
