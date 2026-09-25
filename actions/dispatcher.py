"""ActionDispatcher: confirmed gesture event -> logical action -> adapter.

Knows nothing about MediaPipe, geometry, or frames. Consumes only the
confirmed-event strings produced by vision.rotation / vision.pinch /
vision.yo_zoom.

Safety matrix (from config["action"]):
  enabled=false            -> observe only, never send
  enabled=true dry_run=true -> display/log only, never send
  enabled=true dry_run=false -> real SendInput permitted

Duplicate protection: an identical event arriving inside
duplicate_window_ms is acknowledged but not re-sent (belt-and-braces
over the gesture-layer cooldowns; no queues).

No modes, no lock state: whole-hand mouse + rotation navigation +
pinch hold-click + Ctrl+wheel zoom steps. LEFT_DOWN / LEFT_UP drive
the held button through the mouse adapter (drag works: the palm keeps
moving the cursor while held); navigation sends through the keyboard
adapter; ZOOM_IN / ZOOM_OUT send one atomic Ctrl+wheel tick
(Ctrl is always released via try/finally + KeyboardAdapter.close()).
"""
from dataclasses import dataclass
import datetime

from actions import events as E

_ARROW_KEYS = ("LEFT", "RIGHT", "UP", "DOWN")


def _display_key(key):
    return f"{key}_ARROW" if key in _ARROW_KEYS else key


@dataclass
class ActionRecord:
    gesture: str
    action: str
    key: str
    mode: str  # OBSERVED / DRY_RUN / LIVE
    sent: bool = False


class ActionDispatcher:
    def __init__(self, cfg, keyboard, mouse=None):
        self.cfg = cfg
        self.keyboard = keyboard
        self.mouse = mouse
        self.mapping = dict(cfg.get("mapping", E.DEFAULT_MAPPING))
        self.keymap = dict(cfg.get("keys", {}))
        self.last_action = "NONE"
        self.last_action_time = ""
        self.last_record = None
        self._last_seen = {}  # gesture -> last dispatch timestamp (s)

    def mode(self):
        if not self.cfg.get("enabled", False):
            return "OBSERVED"
        return "DRY_RUN" if self.cfg.get("dry_run", True) else "LIVE"

    def dispatch(self, gesture_event, now_s):
        """Route one confirmed gesture event. Returns ActionRecord or None
        (None = unmapped/unknown event, or duplicate inside window)."""
        action = self.mapping.get(gesture_event)
        if action is None:
            return None
        if action == E.LEFT_DOWN:
            return self._record(gesture_event, action, "LEFT_DOWN",
                                self.mode(), self._maybe_hold(True))
        if action == E.LEFT_UP:
            return self._record(gesture_event, action, "LEFT_UP",
                                self.mode(), self._maybe_hold(False))
        if action == E.ZOOM_IN:
            return self._record(gesture_event, action, "CTRL+WHEEL+",
                                self.mode(), self._maybe_zoom(+1))
        if action == E.ZOOM_OUT:
            return self._record(gesture_event, action, "CTRL+WHEEL-",
                                self.mode(), self._maybe_zoom(-1))
        if action == E.SCROLL_UP:
            return self._record(gesture_event, action, "WHEEL+",
                                self.mode(), self._maybe_scroll(+1))
        if action == E.SCROLL_DOWN:
            return self._record(gesture_event, action, "WHEEL-",
                                self.mode(), self._maybe_scroll(-1))
        window = self.cfg.get("duplicate_window_ms", 150) / 1000.0
        prev = self._last_seen.get(gesture_event)
        if prev is not None and (now_s - prev) < window:
            return None
        self._last_seen[gesture_event] = now_s
        key = self.keymap.get(action)
        if key is None:
            return None
        mode = self.mode()
        sent = False
        if mode == "LIVE":
            self.keyboard.press_key(key)
            sent = True
        return self._record(gesture_event, action, _display_key(key), mode, sent)

    def _maybe_hold(self, down):
        if self.mode() == "LIVE" and self.mouse is not None:
            return self.mouse.left_down() if down else self.mouse.left_up()
        return False

    def _maybe_zoom(self, direction):
        """One atomic Ctrl+wheel tick. Ctrl always released (finally)."""
        if (self.mode() != "LIVE" or self.mouse is None
                or self.keyboard is None):
            return False
        try:
            self.keyboard.key_down("CTRL")
            try:
                self.mouse.wheel(direction)
            finally:
                try:
                    self.keyboard.key_up("CTRL")
                except Exception:
                    pass
        except Exception:
            return False
        return True

    def _maybe_scroll(self, direction):
        """One plain wheel tick (no Ctrl). LIVE + mouse only."""
        if self.mode() != "LIVE" or self.mouse is None:
            return False
        try:
            self.mouse.wheel(direction)
        except Exception:
            return False
        return True

    def _record(self, gesture, action, key, mode, sent):
        rec = ActionRecord(gesture=gesture, action=action,
                           key=key, mode=mode, sent=sent)
        self.last_action = action
        self.last_action_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.last_record = rec
        return rec
