"""Pinch left-click tests: hysteresis, confirm, hold/drag, hand-loss safety.

Synthetic 21-landmark hands. No webcam, no real SendInput.
"""
import unittest

import numpy as np


def _cfg(**over):
    cfg = {"start_threshold": 0.30, "release_threshold": 0.38,
           "confirm_frames": 3, "click_cooldown_ms": 600}
    cfg.update(over)
    return cfg


def _open_hand():
    """Thumb and index far apart (normalized distance >> start)."""
    lm = np.zeros((21, 2))
    lm[0] = [0.50, 0.80]
    xs = {5: 0.44, 9: 0.50, 13: 0.56, 17: 0.62}
    for mcp, x in xs.items():
        lm[mcp] = [x, 0.60]
        lm[mcp + 1] = [x, 0.48]
        lm[mcp + 2] = [x, 0.38]
        lm[mcp + 3] = [x, 0.28]
    lm[1] = [0.40, 0.68]
    lm[2] = [0.36, 0.62]
    lm[3] = [0.32, 0.56]
    lm[4] = [0.28, 0.50]  # thumb tip out
    return lm


def _pinch_at(norm_dist, base=None):
    """Place thumb tip so ||index-thumb||/scale == norm_dist."""
    lm = (base if base is not None else _open_hand()).copy()
    scale = float(np.linalg.norm(lm[0] - lm[9]))
    index = lm[8].astype(np.float64)
    lm[4] = index + np.array([norm_dist * scale, 0.0])
    return lm


def _travel_at(norm_dist):
    """Pinch shape with palm shifted (mouse drag: position changes)."""
    lm = _pinch_at(norm_dist)
    lm += np.array([0.10, -0.05])
    return lm


def _tracker(**over):
    from vision.pinch import PinchTracker
    return PinchTracker(_cfg(**over))


def _feed(tr, lms, t0=0.0, dt=1 / 30.0):
    out = []
    t = t0
    for lm in lms:
        out += tr.update(lm, t)
        t += dt
    return out


class TestDistance(unittest.TestCase):
    def test_normalized_formula(self):
        from vision.pinch import pinch_distance_norm
        from vision.poses import hand_scale_norm
        lm = _pinch_at(0.25)
        d = pinch_distance_norm(lm)
        self.assertAlmostEqual(d, 0.25, places=6)
        self.assertAlmostEqual(
            float(np.linalg.norm(lm[4] - lm[8])) / hand_scale_norm(lm),
            d, places=9)

    def test_invariant_to_uniform_scale_about_wrist(self):
        from vision.pinch import pinch_distance_norm
        lm = _pinch_at(0.20)
        wrist = lm[0].copy()
        scaled = wrist + (lm - wrist) * 1.7
        self.assertAlmostEqual(pinch_distance_norm(scaled),
                               pinch_distance_norm(lm), places=6)

    def test_invariant_to_translation(self):
        from vision.pinch import pinch_distance_norm
        lm = _pinch_at(0.20)
        self.assertAlmostEqual(pinch_distance_norm(lm + np.array([0.3, 0.2])),
                               pinch_distance_norm(lm), places=6)


class TestNormalClick(unittest.TestCase):
    def test_confirm_then_release_one_down_one_up(self):
        tr = _tracker()
        # Warm filter with open hand (does not arm cooldown).
        _feed(tr, [_open_hand()], t0=0.0)
        # 2 frames below start: still candidate
        self.assertEqual(_feed(tr, [_pinch_at(0.15)] * 2, t0=0.1), [])
        self.assertEqual(tr.state, "CANDIDATE")
        # 3rd frame: HELD + STARTED
        self.assertEqual(_feed(tr, [_pinch_at(0.15)], t0=0.2),
                         ["PINCH_STARTED"])
        self.assertTrue(tr.held)
        # Release past threshold: exactly one UP
        self.assertEqual(_feed(tr, [_pinch_at(0.50)], t0=0.5),
                         ["PINCH_RELEASED"])
        self.assertFalse(tr.held)
        self.assertEqual(tr.state, "OPEN")

    def test_needs_three_consecutive_start_frames(self):
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        self.assertEqual(_feed(tr, [_pinch_at(0.15)] * 2, t0=0.1), [])
        self.assertEqual(_feed(tr, [_pinch_at(0.15)], t0=0.2),
                         ["PINCH_STARTED"])

    def test_open_hand_frames_do_not_block_next_pinch(self):
        """Regression: open d >= release must not re-arm cooldown every frame."""
        tr = _tracker()
        for i in range(30):
            self.assertEqual(tr.update(_open_hand(), i / 30.0), [])
        self.assertEqual(_feed(tr, [_pinch_at(0.15)] * 3, t0=1.0),
                         ["PINCH_STARTED"])

    def test_band_noise_does_not_cancel_candidate(self):
        """0.31..0.37 is inside hysteresis: count holds, does not reset."""
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        t = 0.1
        # dip below start, then jitter in band, then dip again
        for d in (0.20, 0.35, 0.36, 0.20, 0.37, 0.20):
            tr.update(_pinch_at(d), t)
            t += 1 / 30.0
        self.assertTrue(tr.held)
        self.assertEqual(tr.last_event, "PINCH_STARTED")


class TestHoldNoRepeat(unittest.TestCase):
    def test_long_hold_single_down(self):
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        self.assertEqual(_feed(tr, [_pinch_at(0.15)] * 120, t0=0.5),
                         ["PINCH_STARTED"])

    def test_hold_in_hysteresis_band_stays_held(self):
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        self.assertEqual(_feed(tr, [_pinch_at(0.15)] * 3, t0=0.1),
                         ["PINCH_STARTED"])
        # 0.30..0.36: between start and release — still HELD, no events
        for d in (0.30, 0.32, 0.34, 0.36):
            self.assertEqual(_feed(tr, [_pinch_at(d)], t0=1.0), [])
            self.assertEqual(tr.state, "HELD")
        # Cross release: one UP
        self.assertEqual(_feed(tr, [_pinch_at(0.40)], t0=1.5),
                         ["PINCH_RELEASED"])


class TestThresholdJitter(unittest.TestCase):
    def test_jitter_around_start_at_most_one_click(self):
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        total = []
        t = 0.1
        for d in (0.29, 0.31, 0.30, 0.29, 0.31) * 6:
            total += tr.update(_pinch_at(d), t)
            t += 1 / 30.0
        starts = total.count("PINCH_STARTED")
        releases = total.count("PINCH_RELEASED")
        # Band holds count: may confirm once, never thrash.
        self.assertLessEqual(starts, 1)
        self.assertLessEqual(releases, starts)
        if starts == 1:
            self.assertTrue(tr.held)  # never released (never hit 0.38)

    def test_start_release_hysteresis_gap(self):
        cfg = _cfg()
        self.assertLess(cfg["start_threshold"], cfg["release_threshold"])


class TestDrag(unittest.TestCase):
    def test_pinch_move_release_one_down_one_up(self):
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        self.assertEqual(_feed(tr, [_pinch_at(0.15)] * 3, t0=0.1),
                         ["PINCH_STARTED"])
        # Palm moves while still pinched: still HELD, no extra events
        self.assertEqual(_feed(tr, [_travel_at(0.18)] * 30, t0=1.0), [])
        self.assertTrue(tr.held)
        self.assertEqual(_feed(tr, [_pinch_at(0.50)], t0=3.0),
                         ["PINCH_RELEASED"])


class TestRepeatedPinch(unittest.TestCase):
    def test_two_lifecycles(self):
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        out = []
        out += _feed(tr, [_pinch_at(0.15)] * 3, t0=0.1)
        out += _feed(tr, [_pinch_at(0.50)] * 2, t0=2.0)
        out += _feed(tr, [_pinch_at(0.15)] * 3, t0=3.0)
        out += _feed(tr, [_pinch_at(0.50)] * 2, t0=5.0)
        self.assertEqual(out, ["PINCH_STARTED", "PINCH_RELEASED",
                               "PINCH_STARTED", "PINCH_RELEASED"])


class TestHandLossSafety(unittest.TestCase):
    def test_hand_lost_while_held_emits_one_release(self):
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        self.assertEqual(_feed(tr, [_pinch_at(0.15)] * 3, t0=0.1),
                         ["PINCH_STARTED"])
        self.assertEqual(tr.update(None, 1.0), ["PINCH_RELEASED"])
        self.assertFalse(tr.held)
        self.assertEqual(_feed(tr, [_pinch_at(0.15)] * 3, t0=2.0),
                         ["PINCH_STARTED"])

    def test_hand_lost_before_confirm_no_release(self):
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        _feed(tr, [_pinch_at(0.15)] * 2, t0=0.1)
        self.assertEqual(tr.update(None, 1.0), [])
        self.assertFalse(tr.held)

    def test_hand_lost_never_stuck_held(self):
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        _feed(tr, [_pinch_at(0.15)] * 3, t0=0.1)
        tr.update(None, 1.0)
        self.assertEqual(tr.state, "OPEN")
        self.assertFalse(tr.held)


class TestCooldownAndNoOpenPoseRequirement(unittest.TestCase):
    def test_repinch_inside_cooldown_suppressed(self):
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        _feed(tr, [_pinch_at(0.15)] * 3, t0=0.1)
        _feed(tr, [_pinch_at(0.50)] * 2, t0=0.5)
        # ~100 ms after release: still inside 600 ms cooldown
        self.assertEqual(_feed(tr, [_pinch_at(0.15)] * 3, t0=0.6), [])

    def test_pinch_works_when_other_fingers_folded(self):
        base = _open_hand()
        for m in (9, 13, 17):
            base[m + 1] = [base[m][0], 0.58]
            base[m + 2] = [base[m][0], 0.64]
            base[m + 3] = [base[m][0], 0.70]
        tr = _tracker()
        _feed(tr, [_open_hand()], t0=0.0)
        self.assertEqual(_feed(tr, [_pinch_at(0.15, base)] * 3, t0=0.1),
                         ["PINCH_STARTED"])


class TestNoFistClick(unittest.TestCase):
    def test_poses_module_has_no_fist_click(self):
        from vision import poses as P
        self.assertFalse(hasattr(P, "FistTracker"))
        self.assertFalse(hasattr(P, "FIST_STARTED"))
        self.assertFalse(hasattr(P, "FIST_RELEASED"))


if __name__ == "__main__":
    unittest.main()
