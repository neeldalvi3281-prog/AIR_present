"""YoZoomTracker tests: YO-gated committed-angle zoom steps.

Angles fed directly in degrees. mirror_sign=+1 keeps tests in physical
space (positive = clockwise = zoom in); app uses the default -1.
No webcam, no SendInput.
"""
import unittest


def _cfg(**over):
    cfg = {"step_degrees": 30.0, "confirm_frames": 4,
           "hysteresis_degrees": 3.0, "tick_spacing_ms": 250}
    cfg.update(over)
    return cfg


def _tracker(**over):
    from vision.yo_zoom import YoZoomTracker
    mirror = over.pop("mirror_sign", 1)
    return YoZoomTracker(_cfg(**over), mirror_sign=mirror)


def _feed(tr, angles, t0=0.0, dt=1 / 30.0):
    out = []
    t = t0
    for a in angles:
        out += tr.update(a, t)
        t += dt
    return out


class TestSteps(unittest.TestCase):
    def test_first_frame_anchors_no_event(self):
        tr = _tracker()
        self.assertEqual(tr.update(10.0, 0.0), [])
        self.assertAlmostEqual(tr.anchor, 10.0)

    def test_clockwise_step_fires_zoom_in_once(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [0.0] + [30.0] * 4), ["ZOOM_IN_STEP"])

    def test_anticlockwise_step_fires_zoom_out_once(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [0.0] + [-30.0] * 4), ["ZOOM_OUT_STEP"])

    def test_continuous_twist_pages_continuously(self):
        tr = _tracker()
        out = _feed(tr, [0.0])
        t = 1.0
        for target in (30.0, 60.0, 90.0):
            out += _feed(tr, [target] * 4, t0=t)
            t += 1.0
        self.assertEqual(out, ["ZOOM_IN_STEP"] * 3)

    def test_reversal_consumes_back(self):
        tr = _tracker()
        _feed(tr, [0.0])
        self.assertEqual(_feed(tr, [30.0] * 4, t0=1.0), ["ZOOM_IN_STEP"])
        self.assertEqual(_feed(tr, [0.0] * 4, t0=2.0), ["ZOOM_OUT_STEP"])

    def test_sub_step_no_action(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [0.0, 10.0, 20.0, 25.0]), [])

    def test_hold_at_step_never_repeats(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [0.0] + [30.0] * 60), ["ZOOM_IN_STEP"])

    def test_hysteresis_band_holds_candidate(self):
        tr = _tracker()
        _feed(tr, [0.0])
        # 28.0 is inside (step-hyst, step)=(27, 30): holds, no cancel.
        self.assertEqual(_feed(tr, [28.0], t0=0.1), [])
        # A snap-sized jump past step completes the confirm.
        self.assertEqual(_feed(tr, [40.0] * 4, t0=0.2), ["ZOOM_IN_STEP"])

    def test_back_inside_cancels(self):
        tr = _tracker()
        _feed(tr, [0.0])
        self.assertEqual(_feed(tr, [30.0], t0=0.1), [])
        self.assertEqual(_feed(tr, [5.0, 30.0], t0=0.2), [])

    def test_tick_spacing_rate_limits(self):
        tr = _tracker()
        _feed(tr, [0.0])
        out = _feed(tr, [30.0] * 4, t0=1.0)
        out += _feed(tr, [60.0] * 4, t0=1.06)  # only 60ms later
        self.assertEqual(out, ["ZOOM_IN_STEP"])

    def test_none_resets_without_event(self):
        tr = _tracker()
        _feed(tr, [0.0, 30.0, 30.0])
        self.assertEqual(tr.update(None, 1.0), [])
        self.assertIsNone(tr.anchor)
        self.assertEqual(_feed(tr, [90.0, 90.0], t0=2.0), [])


class TestDefaultMirror(unittest.TestCase):
    def test_image_ccw_is_physical_cw_zoom_in(self):
        from vision.yo_zoom import YoZoomTracker
        tr = YoZoomTracker(_cfg())  # default MIRROR_SIGN
        self.assertEqual(_feed(tr, [0.0] + [-30.0] * 4), ["ZOOM_IN_STEP"])

    def test_long_cw_never_flips(self):
        from vision.yo_zoom import YoZoomTracker
        tr = YoZoomTracker(_cfg())
        out = _feed(tr, [0.0])
        t = 1.0
        for target in (-30.0, -60.0, -90.0):
            out += _feed(tr, [target] * 4, t0=t)
            t += 1.0
        self.assertEqual(out, ["ZOOM_IN_STEP"] * 3)


if __name__ == "__main__":
    unittest.main()
