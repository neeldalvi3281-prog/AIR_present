"""Reform tests: palm centroid, scale, orientation, wrapped deltas, mirror.

No webcam: synthetic 21-landmark hands in normalized [0,1] coordinates.
"""
import math
import unittest

import numpy as np


def _hand(up_deg=0.0, translate=(0.0, 0.0)):
    """Synthetic palm: wrist at (0.5, 0.8), MCP mean on the up axis.

    up_deg: 0 = fingers point image-up; positive = clockwise (image coords).
    Knuckle side offsets sum to 0 so wrist->MCP-centroid equals up_deg
    (orientation uses mean of 4 MCPs, not a single landmark).
    """
    lm = np.zeros((21, 2))
    tx, ty = translate
    wrist = np.array([0.5 + tx, 0.8 + ty])
    rad = math.radians(up_deg)
    up = np.array([math.sin(rad), -math.cos(rad)])  # image coords, y down
    side = np.array([math.cos(rad), math.sin(rad)])  # knuckle direction
    lm[0] = wrist
    lm[5] = wrist + up * 0.16 - side * 0.06   # index MCP
    lm[9] = wrist + up * 0.20 - side * 0.02   # middle MCP
    lm[13] = wrist + up * 0.16 + side * 0.04  # ring MCP
    lm[17] = wrist + up * 0.14 + side * 0.04  # pinky MCP (side sum = 0)
    return lm


class TestCentroid(unittest.TestCase):
    def test_centroid_is_mean_of_five_palm_points(self):
        from vision.palm import PALM_IDS, palm_centroid
        lm = _hand()
        cx, cy = palm_centroid(lm)
        ex = float(lm[list(PALM_IDS)][:, 0].mean())
        ey = float(lm[list(PALM_IDS)][:, 1].mean())
        self.assertAlmostEqual(cx, ex, places=9)
        self.assertAlmostEqual(cy, ey, places=9)

    def test_centroid_uses_wrist_and_four_mcps(self):
        from vision.palm import PALM_IDS
        self.assertEqual(tuple(sorted(PALM_IDS)), (0, 5, 9, 13, 17))

    def test_centroid_follows_translation(self):
        from vision.palm import palm_centroid
        x0, y0 = palm_centroid(_hand(translate=(0.0, 0.0)))
        x1, y1 = palm_centroid(_hand(translate=(0.1, -0.05)))
        self.assertAlmostEqual(x1 - x0, 0.1, places=9)
        self.assertAlmostEqual(y1 - y0, -0.05, places=9)


class TestAngle(unittest.TestCase):
    def test_up_is_zero(self):
        from vision.palm import palm_angle_deg
        self.assertAlmostEqual(palm_angle_deg(_hand(0.0)), 0.0, places=6)

    def test_clockwise_positive(self):
        from vision.palm import palm_angle_deg
        self.assertAlmostEqual(palm_angle_deg(_hand(30.0)), 30.0, places=6)
        self.assertAlmostEqual(palm_angle_deg(_hand(90.0)), 90.0, places=6)

    def test_anticlockwise_negative(self):
        from vision.palm import palm_angle_deg
        self.assertAlmostEqual(palm_angle_deg(_hand(-45.0)), -45.0, places=6)

    def test_translation_invariant(self):
        from vision.palm import palm_angle_deg
        a = palm_angle_deg(_hand(25.0, translate=(0.0, 0.0)))
        b = palm_angle_deg(_hand(25.0, translate=(0.15, -0.1)))
        self.assertAlmostEqual(a, b, places=6)

    def test_scale_invariant(self):
        from vision.palm import palm_angle_deg
        lm = _hand(20.0)
        # Double the hand size around the wrist: angle must not change.
        big = lm.copy()
        big[1:] = lm[0] + (lm[1:] - lm[0]) * 2.0
        self.assertAlmostEqual(palm_angle_deg(lm), palm_angle_deg(big),
                               places=6)

    def test_chirality_independent(self):
        # Orientation uses the MCP centroid, so swapping knuckle order
        # (opposite hand) must not change it.
        from vision.palm import palm_angle_deg
        lm = _hand(15.0)
        other = lm.copy()
        other[[5, 17]] = other[[17, 5]]  # index<->pinky MCP
        self.assertAlmostEqual(palm_angle_deg(other),
                               palm_angle_deg(lm), places=9)

    def test_uses_mcp_centroid_not_single_landmark(self):
        # Jitter only the middle MCP: centroid damps the swing vs a
        # single wrist->middle axis (which would see the full offset).
        from vision.palm import palm_angle_deg
        base = palm_angle_deg(_hand(0.0))
        lm = _hand(0.0)
        lm[9, 0] += 0.05
        got = palm_angle_deg(lm)
        single = math.degrees(math.atan2(0.05, -0.20))  # old axis approx
        self.assertLess(abs(got - base), abs(single))

    def test_any_absolute_orientation_measured_exactly(self):
        # Hand may rest horizontal, sideways, or upside-down: the angle
        # is still exact (fingers-up is 0deg, not a required pose).
        from vision.palm import palm_angle_deg
        for deg in (0.0, 30.0, 90.0, 179.0, -179.0, -90.0, -45.0, 135.0):
            with self.subTest(deg=deg):
                got = palm_angle_deg(_hand(deg))
                if abs(deg) == 179.0:
                    self.assertAlmostEqual(abs(got), 179.0, places=5)
                    self.assertEqual(got > 0, deg > 0)
                else:
                    self.assertAlmostEqual(got, deg, places=5)


class TestWrap(unittest.TestCase):
    def test_wrap_179_to_minus179_is_plus2(self):
        from vision.palm import wrap_delta_deg
        self.assertAlmostEqual(wrap_delta_deg(179.0, -179.0), 2.0, places=9)

    def test_wrap_minus179_to_179_is_minus2(self):
        from vision.palm import wrap_delta_deg
        self.assertAlmostEqual(wrap_delta_deg(-179.0, 179.0), -2.0, places=9)

    def test_wrap_plain(self):
        from vision.palm import wrap_delta_deg
        self.assertAlmostEqual(wrap_delta_deg(10.0, 40.0), 30.0, places=9)
        self.assertAlmostEqual(wrap_delta_deg(40.0, 10.0), -30.0, places=9)
        self.assertAlmostEqual(wrap_delta_deg(0.0, 180.0), 180.0, places=9)

    def test_wrap_never_exceeds_180(self):
        import random
        from vision.palm import wrap_delta_deg
        rnd = random.Random(11)
        for _ in range(500):
            d = wrap_delta_deg(rnd.uniform(-180, 180),
                               rnd.uniform(-180, 180))
            self.assertLessEqual(abs(d), 180.0)


class TestMirror(unittest.TestCase):
    def test_mirror_sign_documented_and_negative(self):
        # The preview frame is mirrored BEFORE detection, which reverses
        # apparent rotation: physical CW looks CCW in landmarks. One
        # constant compensates, in vision.palm only.
        from vision import palm as P
        self.assertEqual(P.MIRROR_SIGN, -1)
        self.assertIn("mirror", P.__doc__.lower())

    def test_physical_clockwise_maps_positive(self):
        from vision.palm import MIRROR_SIGN, wrap_delta_deg
        # Physical CW of +30 shows up as -30 in mirrored landmarks.
        measured = wrap_delta_deg(0.0, -30.0)
        self.assertGreater(MIRROR_SIGN * measured, 0.0)


class TestScale(unittest.TestCase):
    def test_scale_is_wrist_to_middle_mcp(self):
        from vision.palm import palm_scale
        lm = _hand()
        expected = float(np.linalg.norm(lm[9] - lm[0]))
        self.assertAlmostEqual(palm_scale(lm), expected, places=9)
        self.assertGreater(palm_scale(np.zeros((21, 2))), 0.0)


if __name__ == "__main__":
    unittest.main()
