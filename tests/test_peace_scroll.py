"""PeaceScrollTracker tests: entry confirm, hold repeat, flip, reset.

Directions fed directly as +1 (point right / scroll up), -1 (point
left / scroll down), None (other pose / hand lost). No webcam.
"""
import unittest


def _cfg(**over):
    cfg = {"confirm_frames": 4, "repeat_ms": 200}
    cfg.update(over)
    return cfg


def _tracker(**over):
    from vision.peace_scroll import PeaceScrollTracker
    return PeaceScrollTracker(_cfg(**over))


def _feed(tr, ds, t0=0.0, dt=1 / 30.0):
    out = []
    t = t0
    for d in ds:
        out += tr.update(d, t)
        t += dt
    return out


class TestEntry(unittest.TestCase):
    def test_needs_four_frames_then_ticks(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [1] * 3), [])
        self.assertEqual(_feed(tr, [1], t0=3 / 30.0), ["SCROLL_UP_STEP"])

    def test_flicker_below_confirm_no_tick(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [1, 1, 1, None, 1, 1]), [])
        self.assertEqual(tr.direction, 0)

    def test_left_ticks_down(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [-1] * 4), ["SCROLL_DOWN_STEP"])


class TestRepeat(unittest.TestCase):
    def test_hold_repeats_every_repeat_ms(self):
        tr = _tracker()
        out = _feed(tr, [1] * 4, t0=0.0)  # first tick at ~0.1
        out += _feed(tr, [1] * 6, t0=0.2)  # next at ~0.3
        out += _feed(tr, [1] * 6, t0=0.4)  # next at ~0.5
        self.assertEqual(out, ["SCROLL_UP_STEP"] * 3)

    def test_no_burst_faster_than_repeat(self):
        tr = _tracker()
        # 30 frames ~= 1 s: ticks only at ~0.1, 0.3, 0.5, 0.7, 0.9.
        self.assertEqual(_feed(tr, [1] * 30, t0=0.0),
                         ["SCROLL_UP_STEP"] * 5)


class TestFlip(unittest.TestCase):
    def test_direction_flip_reconfirms(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [1] * 4, t0=0.0), ["SCROLL_UP_STEP"])
        # Flip: single frames of -1 do nothing until 4 accumulate.
        self.assertEqual(_feed(tr, [-1] * 3, t0=1.0), [])
        self.assertEqual(_feed(tr, [-1], t0=1.1), ["SCROLL_DOWN_STEP"])
        self.assertEqual(tr.direction, -1)

    def test_zero_resets_silently(self):
        tr = _tracker()
        _feed(tr, [1] * 4, t0=0.0)
        self.assertEqual(tr.update(0, 1.0), [])
        self.assertEqual(tr.direction, 0)

    def test_none_resets_silently(self):
        tr = _tracker()
        _feed(tr, [-1] * 4, t0=0.0)
        self.assertEqual(tr.update(None, 1.0), [])
        self.assertEqual(tr.direction, 0)
        self.assertEqual(tr.last_event, "NONE")
        # Fresh hold re-confirms from zero.
        self.assertEqual(_feed(tr, [-1] * 3, t0=2.0), [])
        self.assertEqual(_feed(tr, [-1], t0=2.1), ["SCROLL_DOWN_STEP"])


if __name__ == "__main__":
    unittest.main()
