"""Windows keyboard adapter via native SendInput (ctypes, no extra dependency).

Only atomic press_key() is public for actions: key-down is always paired
with key-up in a try/finally, so a crash path can't leave a key held.
close() releases anything still tracked as held (shutdown safety).
"""
import ctypes
from ctypes import wintypes

VK = {"LEFT": 0x25, "RIGHT": 0x27, "UP": 0x26, "DOWN": 0x28,
      "ESCAPE": 0x1B, "CTRL": 0x11}
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1

ULONG_PTR = ctypes.c_void_p  # pointer-sized unsigned (8 bytes on x64)


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT),
                ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    # Mirrors Win32 INPUT: DWORD type + 4-byte pad + 32-byte union = 40
    # bytes on x64. cbSize MUST be 40 or SendInput returns 0.
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT),
                              ctypes.c_int]
_user32.SendInput.restype = wintypes.UINT


def _send(vk, flags):
    inp = _INPUT(type=INPUT_KEYBOARD,
                 ki=_KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags,
                                time=0, dwExtraInfo=None))
    n = _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
    if n != 1:
        err = ctypes.get_last_error()
        raise RuntimeError(
            f"SendInput failed for VK={vk:#x} "
            f"(sent {n}/1, cbSize={ctypes.sizeof(_INPUT)}, "
            f"win32 err {err}: {ctypes.FormatError(err)})")


class KeyboardAdapter:
    """Sends key presses. Injectable sender enables dry unit tests."""

    def __init__(self, sender=None):
        self._sender = sender or _send
        self._held = set()

    def key_down(self, name):
        vk = VK[name]
        self._sender(vk, 0)
        self._held.add(name)

    def key_up(self, name):
        vk = VK[name]
        try:
            self._sender(vk, KEYEVENTF_KEYUP)
        finally:
            self._held.discard(name)

    def press_key(self, name):
        """Atomic down+up. key_up() discards from _held in a finally,
        so no exception path can leave a key tracked as held."""
        if name not in VK:
            raise ValueError(f"unknown key {name!r}")
        self.key_down(name)
        self.key_up(name)

    @property
    def held(self):
        return frozenset(self._held)

    def close(self):
        """Release any tracked held keys. Called on shutdown."""
        for name in sorted(self._held):
            try:
                self._sender(VK[name], KEYEVENTF_KEYUP)
            except Exception:
                pass
        self._held.clear()
