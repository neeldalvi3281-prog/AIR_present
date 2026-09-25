"""Rolling performance profiler: mean/median/p95/min/max + pipeline counters."""
import time
from collections import deque

import numpy as np


class RollingStat:
    def __init__(self, maxlen=600):
        self._values = deque(maxlen=maxlen)

    def push(self, value):
        self._values.append(float(value))

    def summary(self):
        if not self._values:
            return {"count": 0, "mean": 0.0, "median": 0.0,
                    "p95": 0.0, "min": 0.0, "max": 0.0}
        a = np.asarray(self._values, dtype=np.float64)
        return {
            "count": int(a.size),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "min": float(a.min()),
            "max": float(a.max()),
        }


class Profiler:
    """Collects camera/inference timings and system stats for Phase 1."""

    def __init__(self):
        self.infer_latency = RollingStat()
        self.e2e_latency = RollingStat()
        self.cpu = RollingStat()
        self.ram_mb = RollingStat()
        self._cap_times = deque(maxlen=600)
        self._infer_times = deque(maxlen=600)
        self.frames_seen = 0
        self.frames_with_hand = 0
        self._cached_summary = None
        self._cached_at = 0.0

    def note_capture(self, timestamp):
        self._cap_times.append(float(timestamp))

    def note_inference(self, timestamp, found_hand, infer_lat_s, e2e_lat_s):
        self._infer_times.append(float(timestamp))
        self.infer_latency.push(infer_lat_s * 1000.0)  # ms
        self.e2e_latency.push(e2e_lat_s * 1000.0)  # ms
        self.frames_seen += 1
        if found_hand:
            self.frames_with_hand += 1

    def note_system(self, cpu_pct, ram_mb):
        self.cpu.push(cpu_pct)
        self.ram_mb.push(ram_mb)

    @staticmethod
    def _fps(times):
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        if span <= 0:
            return 0.0
        return (len(times) - 1) / span

    def summary_cached(self, min_interval_s=0.5, captured=0,
                       overwritten_unread=0):
        """Cached summary for live overlays: full median/p95 stats at most
        every min_interval_s (default 2 Hz). Push-side notes stay per-frame.
        """
        now = time.perf_counter()
        if (self._cached_summary is None
                or now - self._cached_at >= min_interval_s):
            self._cached_summary = self.summary(
                captured=captured, overwritten_unread=overwritten_unread)
            self._cached_at = now
        return self._cached_summary

    def summary(self, captured=0, overwritten_unread=0):
        cam_fps = self._fps(self._cap_times)
        infer_fps = self._fps(self._infer_times)
        detection_rate = (self.frames_with_hand / self.frames_seen
                          if self.frames_seen else 0.0)
        drop_rate = (overwritten_unread / captured if captured else 0.0)
        return {
            "camera_fps": round(cam_fps, 2),
            "inference_fps": round(infer_fps, 2),
            "infer_latency_ms": self.infer_latency.summary(),
            "e2e_latency_ms": self.e2e_latency.summary(),
            "frames_seen": self.frames_seen,
            "frames_with_hand": self.frames_with_hand,
            "hand_detection_rate": round(detection_rate, 4),
            "captured": captured,
            "frame_drops": overwritten_unread,
            "frame_drop_rate": round(drop_rate, 4),
            "cpu_pct": self.cpu.summary(),
            "ram_mb": self.ram_mb.summary(),
        }
