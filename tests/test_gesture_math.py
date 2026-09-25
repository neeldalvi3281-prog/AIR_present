"""Deterministic webcam-free tests: One-Euro filter + landmark derive."""
import unittest

import numpy as np


class TestOneEuro(unittest.TestCase):
    def test_step_response_settles(self):
        from vision.filters import OneEuroFilter
        f = OneEuroFilter(x0=0.0)
        out = [f.filter(1.0, t=i / 30.0) for i in range(60)]
        self.assertGreater(out[-1], 0.9)
        self.assertLess(abs(out[2] - 1.0), 0.9)

    def test_constant_input_no_drift(self):
        from vision.filters import OneEuroFilter
        f = OneEuroFilter(x0=0.5)
        for i in range(30):
            self.assertAlmostEqual(f.filter(0.5, t=i / 30.0), 0.5, places=9)

    def test_dt_anomaly_safe(self):
        from vision.filters import Filter2D
        f = Filter2D()
        f.filter(0.1, 0.2, t=0.0)
        x, y = f.filter(10.0, 10.0, t=1000.0)  # huge gap must not explode
        self.assertTrue(np.isfinite(x) and np.isfinite(y))


class TestDerive(unittest.TestCase):
    def test_palm_norm_uses_five_palm_points(self):
        from vision.features import derive
        lm = np.zeros((21, 2))
        lm[[0, 5, 9, 13, 17]] = [[.4, .4], [.5, .4], [.5, .5], [.5, .6], [.4, .5]]
        d = derive(lm, 320, 240)
        self.assertAlmostEqual(d["palm_norm"][0], 0.46, places=6)
        self.assertAlmostEqual(d["palm_norm"][1], 0.48, places=6)
        self.assertGreater(d["hand_scale"], 0)

    def test_scale_norm_stable(self):
        from vision.features import derive
        lm = np.zeros((21, 2))
        lm[0] = [0.5, 0.5]
        lm[9] = [0.6, 0.5]
        d = derive(lm, 320, 240)
        self.assertAlmostEqual(d["scale_norm"], d["hand_scale"] / 320.0, places=9)


if __name__ == "__main__":
    unittest.main()
