"""Low-latency One-Euro adaptive filter (Casiez et al.) for palm/fingertip tracks.

High cutoff when moving fast (low lag), low cutoff when still (jitter-free).
dt is clamped to [1ms, 200ms] so timestamp anomalies can't destabilize it.
"""
import math
from dataclasses import dataclass


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class OneEuroFilter:
    x0: float = 0.0
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0
    _x: float = 0.0
    _dx: float = 0.0
    _t: float | None = None
    _init: bool = False

    def __init__(self, x0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.x0 = x0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = x0
        self._dx = 0.0
        self._t = None
        self._init = False

    def reset(self, x=0.0):
        self._x = x
        self._dx = 0.0
        self._t = None
        self._init = False

    def filter(self, x, t):
        if not self._init:
            self._x = x
            self._t = t
            self._init = True
            return x
        dt = min(max(t - self._t, 1e-3), 0.2)
        self._t = t
        # Filtered derivative -> adaptive cutoff -> filtered signal.
        a_d = _alpha(self.d_cutoff, dt)
        d = (x - self._x) / dt
        self._dx = a_d * d + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


class Filter2D:
    """Two independent One-Euro channels for (x, y) positions."""

    def __init__(self, x0=0.0, y0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self._fx = OneEuroFilter(x0, min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(y0, min_cutoff, beta, d_cutoff)

    def reset(self, x=0.0, y=0.0):
        self._fx.reset(x)
        self._fy.reset(y)

    def filter(self, x, y, t):
        return (self._fx.filter(x, t), self._fy.filter(y, t))
