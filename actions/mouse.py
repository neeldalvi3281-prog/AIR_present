"""Windows mouse adapter via native SendInput (ctypes, no extra dependency).

Absolute movement uses the 0..65535 virtual-desktop space mapped from
normalized [0,1] pointer coordinates; the primary-display size is detected
at runtime (no hardcoded resolution). click() is an atomic down+up pair;
left_down()/left_up() hold the button across frames (pinch drag).
wheel() sends one WHEEL_DELTA notch (Ctrl+wheel = app zoom).
release() is shutdown safety: it ups a held button so Q/ESC or hand loss
can never leave a stuck click.
"""
import ctypes
from ctypes import wintypes

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
WHEEL_DELTA = 120
INPUT_MOUSE = 0


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_void_p)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("mi", _MOUSEINPUT)]


def apply_control_box(x_norm, y_norm, mapping):
    """Fixed virtual control box -> [0,1] screen (normalized camera coords).

    mapped_x = clamp((x - left) / (right - left), 0, 1), same for Y, so
    the box edges hit physical screen corners (outside clamps to the
    edge — the box is the complete usable range). A degenerate span
    (right <= left or bottom <= top) falls back to plain clamping.
    Pure; unit-tested.
    """
    def _one(v, lo, hi):
        v = min(max(float(v), 0.0), 1.0)
        span = hi - lo
        if span <= 1e-9:
            return v
        return min(max((v - lo) / span, 0.0), 1.0)

    return (_one(x_norm, mapping.get("left", 0.0), mapping.get("right", 1.0)),
            _one(y_norm, mapping.get("top", 0.0), mapping.get("bottom", 1.0)))


def _send(dx, dy, flags):
    inp = _INPUT(type=INPUT_MOUSE,
                 mi=_MOUSEINPUT(dx, dy, 0, flags, 0, None))
    n = ctypes.windll.user32.SendInput(1, ctypes.byref(inp),
                                       ctypes.sizeof(_INPUT))
    if n != 1:
        raise RuntimeError("SendInput mouse failed")


def _send_wheel(delta):
    inp = _INPUT(type=INPUT_MOUSE,
                 mi=_MOUSEINPUT(0, 0, delta, MOUSEEVENTF_WHEEL, 0, None))
    n = ctypes.windll.user32.SendInput(1, ctypes.byref(inp),
                                       ctypes.sizeof(_INPUT))
    if n != 1:
        raise RuntimeError("SendInput mouse wheel failed")


class MouseAdapter:
    """Moves the cursor and clicks/holds. Injectable senders for dry tests."""

    def __init__(self, sender=None, wheel_sender=None):
        self._sender = sender or _send
        self._wheel_sender = wheel_sender or _send_wheel
        self.moves = 0
        self.clicks = 0
        self.wheels = 0
        self.button_held = False
        self._last_abs = None  # skip truly identical MOVE repeats

    @staticmethod
    def to_absolute(x_norm, y_norm):
        """Normalized [0,1] -> SendInput absolute 0..65535 (clamped)."""
        x = min(max(float(x_norm), 0.0), 1.0)
        y = min(max(float(y_norm), 0.0), 1.0)
        return (int(x * 65535), int(y * 65535))

    def move(self, x_norm, y_norm):
        ax, ay = self.to_absolute(x_norm, y_norm)
        if (ax, ay) == self._last_abs:
            return False  # stationary: no redundant SendInput
        self._sender(ax, ay, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        self._last_abs = (ax, ay)
        self.moves += 1
        return True

    def click(self):
        self.left_down()
        self.left_up()
        self.clicks += 1

    def left_down(self):
        """Press and hold. Idempotent: returns True only if actually sent."""
        if self.button_held:
            return False
        self._sender(0, 0, MOUSEEVENTF_LEFTDOWN)
        self.button_held = True
        return True

    def left_up(self):
        """Release a held button. No-op (False) when nothing is held."""
        if not self.button_held:
            return False
        self._sender(0, 0, MOUSEEVENTF_LEFTUP)
        self.button_held = False
        return True

    def release(self):
        """Shutdown safety: never leave the button stuck down."""
        self.left_up()

    def wheel(self, direction):
        """One wheel notch: direction > 0 forward (zoom in with Ctrl held).

        Returns True (exactly one SendInput per call; callers rate-limit).
        """
        delta = WHEEL_DELTA if direction > 0 else -WHEEL_DELTA
        self._wheel_sender(delta)
        self.wheels += 1
        return True
