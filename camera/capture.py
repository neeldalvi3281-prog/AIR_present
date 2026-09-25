"""Latest-frame camera manager (OpenCV backend).

One background thread owns VideoCapture and keeps only the newest frame.
Old frames are discarded, never queued (PRD latest-frame policy).
"""
import threading
import time

import cv2

# Backend selector names for benchmarks (opti §6). None = legacy default:
# CAP_DSHOW, then bare VideoCapture fallback.
BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}

# Codec selector names (opti §7). None = device default.
CODECS = {"default": None, "mjpg": "MJPG"}


def _fourcc_name(code):
    try:
        name = bytes(((int(code) >> s) & 0xFF) for s in (0, 8, 16, 24)
                     ).decode("ascii", errors="ignore").strip("\x00 ")
        # Guard against driver garbage: real FOURCCs are alphanumeric.
        if name and all(c.isalnum() for c in name):
            return name
        return "?"
    except Exception:
        return "?"


def _backend_name(value):
    for name, const in BACKENDS.items():
        if const == value:
            return name
    return str(value)


class CameraManager:
    def __init__(self, device=0, width=640, height=480, target_fps=30,
                 backend=None, codec=None):
        self.device = device
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.backend = backend  # None, "dshow", "msmf", "any", ...
        self.codec = codec  # None, "mjpg", "default"
        self._cap = None
        self._backend_used = None
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self.actual_codec = "?"
        self.device = device
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._cap = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._seq = 0
        self._has_unread = False
        self._captured = 0
        self._overwritten = 0  # frames discarded before consumer read them
        self._consumed = 0
        self.status = "closed"

    def start(self):
        if self.backend is None:
            self._cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
            self._backend_used = cv2.CAP_DSHOW
            if not self._cap.isOpened():
                # Fallback without backend flag (some webcams/displays).
                self._cap = cv2.VideoCapture(self.device)
                self._backend_used = cv2.CAP_ANY
        else:
            const = BACKENDS.get(self.backend, cv2.CAP_ANY)
            self._cap = cv2.VideoCapture(self.device, const)
            self._backend_used = const
        if not self._cap.isOpened():
            self.status = "unavailable"
            raise RuntimeError(f"Camera unavailable (device={self.device})")
        fourcc = CODECS.get(self.codec or "default")
        if fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC,
                           cv2.VideoWriter_fourcc(*fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.target_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        # Read back what the device ACTUALLY accepted (opti §8): never
        # report requested values as achieved.
        self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.actual_codec = _fourcc_name(
            self._cap.get(cv2.CAP_PROP_FOURCC))
        self._running = True
        self.status = "open"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def set_resolution(self, width, height):
        self.width, self.height = width, height
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read() if self._cap is not None else (False, None)
            if not ok or frame is None:
                self.status = "read-failed"
                time.sleep(0.01)
                continue
            self.status = "open"
            with self._lock:
                if self._has_unread:
                    self._overwritten += 1
                self._frame = frame
                self._timestamp = time.perf_counter()
                self._seq += 1
                self._has_unread = True
                self._captured += 1

    def read_latest(self):
        """Return (frame, capture_timestamp, seq) or (None, 0.0, last_seq)."""
        with self._lock:
            if self._frame is None:
                return None, 0.0, self._seq
            self._has_unread = False
            self._consumed += 1
            return self._frame, self._timestamp, self._seq

    def stats(self):
        with self._lock:
            return {
                "captured": self._captured,
                "consumed": self._consumed,
                "overwritten_unread": self._overwritten,
                "width": self.width,
                "height": self.height,
                "status": self.status,
                "requested_width": self.width,
                "requested_height": self.height,
                "requested_fps": self.target_fps,
                "actual_width": self.actual_width,
                "actual_height": self.actual_height,
                "actual_fps": self.actual_fps,
                "backend": _backend_name(self._backend_used),
                "codec": self.actual_codec,
            }

    @property
    def resolution(self):
        return (self.width, self.height)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.status = "closed"
