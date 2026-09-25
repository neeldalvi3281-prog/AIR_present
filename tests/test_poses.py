"""Reform pose tests: OPEN / FIST / YO / PEACE / UNKNOWN classifiers.

Fist/UNKNOWN/YO/PEACE gate rotation navigation — they never click.
Left click lives in vision.pinch (see tests/test_pinch.py).
YO + twist drives zoom (see tests/test_yo_zoom.py); PEACE + direction
drives scroll (see tests/test_peace_scroll.py).
No webcam: synthetic 21-landmark hands.
"""
import math
import unittest

import numpy as np


def _base_open():
    lm = np.zeros((21, 2))
    lm[0] = [0.50, 0.80]  # wrist
    xs = {5: 0.44, 9: 0.50, 13: 0.56, 17: 0.62}
    for mcp, x in xs.items():
        lm[mcp] = [x, 0.60]
        lm[mcp + 1] = [x, 0.48]  # PIP
        lm[mcp + 2] = [x, 0.38]  # DIP
        lm[mcp + 3] = [x, 0.28]  # tip
    lm[1] = [0.40, 0.68]
    lm[2] = [0.36, 0.62]
    lm[3] = [0.32, 0.56]
    lm[4] = [0.28, 0.50]  # thumb tip, extended out
    return lm


def _curl(lm, fingers):
    mcp_of = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for f in fingers:
        m = mcp_of[f]
        x = lm[m][0]
        lm[m + 1] = [x, 0.58]  # PIP
        lm[m + 2] = [x, 0.64]  # DIP
        lm[m + 3] = [x, 0.70]  # tip near palm
    return lm


def _fist():
    lm = _curl(_base_open(), ["index", "middle", "ring", "pinky"])
    lm[4] = [0.44, 0.60]  # thumb folded across palm
    return lm


def _peace(index_tip=(0.68, 0.46), middle_tip=(0.74, 0.46)):
    """Index + middle out, ring + pinky folded, thumb folded."""
    lm = _curl(_base_open(), ["ring", "pinky"])
    lm[5] = [0.44, 0.60]  # index MCP
    lm[6] = [0.52, 0.52]  # PIP
    lm[7] = [0.60, 0.48]  # DIP
    lm[8] = list(index_tip)
    lm[9] = [0.50, 0.60]  # middle MCP
    lm[10] = [0.58, 0.52]  # PIP
    lm[11] = [0.66, 0.48]  # DIP
    lm[12] = list(middle_tip)
    lm[4] = [0.44, 0.60]  # thumb folded across palm
    return lm


def _yo():
    """Rock-on: index + pinky extended, middle + ring folded, thumb folded."""
    lm = _curl(_base_open(), ["middle", "ring"])
    lm[4] = [0.44, 0.60]  # thumb folded across palm
    return lm


def _cfg(**over):
    cfg = {"fold_ratio": 1.15, "fist_folded_min": 4}
    cfg.update(over)
    return cfg


def _rotate(lm, deg):
    """Rotate the whole hand about the wrist by deg (image CW positive)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    wrist = np.asarray(lm[0], dtype=np.float64)
    rel = np.asarray(lm, dtype=np.float64) - wrist
    rot = np.array([[c, -s], [s, c]])
    return rel @ rot.T + wrist


_ORIENTATIONS = (0.0, 15.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0)


class TestClassify(unittest.TestCase):
    def test_open(self):
        from vision.poses import classify_pose
        self.assertEqual(classify_pose(_base_open(), _cfg()), "OPEN")

    def test_fist(self):
        from vision.poses import classify_pose
        self.assertEqual(classify_pose(_fist(), _cfg()), "FIST")

    def test_partial_hand_is_unknown(self):
        from vision.poses import classify_pose
        lm = _curl(_base_open(), ["middle", "ring", "pinky"])  # index out
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_single_extended_finger_is_unknown(self):
        from vision.poses import classify_pose
        lm = _curl(_base_open(), ["index", "middle", "ring", "pinky"])
        lm[4] = [0.28, 0.50]  # thumb out alone
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_pinch_shape_is_unknown(self):
        from vision.poses import classify_pose
        lm = _base_open()
        lm[8] = [0.46, 0.52]
        lm[4] = [0.48, 0.54]
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_thumbs_up_shape_is_unknown(self):
        from vision.poses import classify_pose
        lm = _fist()
        lm[2] = [0.54, 0.60]
        lm[3] = [0.57, 0.47]
        lm[4] = [0.60, 0.34]
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_priority_fist_over_thumb_near_index(self):
        from vision.poses import classify_pose
        lm = _fist()
        lm[4] = [0.45, 0.68]  # thumb tip close to folded index tip
        self.assertEqual(classify_pose(lm, _cfg()), "FIST")

    def test_no_old_labels_leak(self):
        from vision import poses as P
        self.assertFalse(hasattr(P, "INDEX_POINT"))
        self.assertFalse(hasattr(P, "PINCH"))
        self.assertFalse(hasattr(P, "THUMBS_UP"))
        self.assertFalse(hasattr(P, "FistTracker"))  # click moved to pinch


class TestYo(unittest.TestCase):
    """Rock-on: index + pinky out, middle + ring folded, thumb folded."""

    def test_strict_shape(self):
        from vision.poses import classify_pose
        self.assertEqual(classify_pose(_yo(), _cfg()), "YO")

    def test_middle_out_is_unknown(self):
        from vision.poses import classify_pose
        lm = _yo()
        lm[12] = [0.50, 0.28]  # middle tip back up (extended)
        lm[10] = [0.50, 0.48]
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_ring_out_is_unknown(self):
        from vision.poses import classify_pose
        lm = _yo()
        lm[16] = [0.56, 0.28]  # ring tip back up (extended)
        lm[14] = [0.56, 0.48]
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_index_folded_is_unknown(self):
        from vision.poses import classify_pose
        lm = _curl(_base_open(), ["index", "middle", "ring"])
        lm[4] = [0.44, 0.60]
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_thumb_out_is_unknown(self):
        from vision.poses import classify_pose
        lm = _yo()
        lm[4] = [0.28, 0.50]  # thumb extended out
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_fist_still_wins(self):
        from vision.poses import classify_pose
        self.assertEqual(classify_pose(_fist(), _cfg()), "FIST")

    def test_open_unaffected(self):
        from vision.poses import classify_pose
        self.assertEqual(classify_pose(_base_open(), _cfg()), "OPEN")

    def test_yo_holds_when_rotated(self):
        """Twisting must not break the shape: ratios are rotation-proof."""
        from vision.poses import classify_pose
        import math
        import numpy as np
        for deg in (30.0, -30.0, 90.0):
            rad = math.radians(deg)
            c, s = math.cos(rad), math.sin(rad)
            wrist = np.asarray(_yo()[0], dtype=np.float64)
            rel = np.asarray(_yo(), dtype=np.float64) - wrist
            rot = np.array([[c, -s], [s, c]])
            with self.subTest(deg=deg):
                self.assertEqual(
                    classify_pose(rel @ rot.T + wrist, _cfg()), "YO")


class TestPeace(unittest.TestCase):
    """Peace: index + middle out, ring + pinky folded, thumb folded."""

    def test_strict_shape(self):
        from vision.poses import classify_pose
        self.assertEqual(classify_pose(_peace(), _cfg()), "PEACE")

    def test_ring_out_is_unknown(self):
        from vision.poses import classify_pose
        lm = _peace()
        lm[16] = [0.56, 0.28]  # ring tip back up (extended)
        lm[14] = [0.56, 0.48]
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_middle_folded_is_unknown(self):
        from vision.poses import classify_pose
        lm = _curl(_base_open(), ["middle", "ring", "pinky"])
        lm[4] = [0.44, 0.60]
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_thumb_out_is_unknown(self):
        from vision.poses import classify_pose
        lm = _peace()
        lm[4] = [0.28, 0.50]  # thumb extended out
        self.assertEqual(classify_pose(lm, _cfg()), "UNKNOWN")

    def test_fist_still_wins(self):
        from vision.poses import classify_pose
        self.assertEqual(classify_pose(_fist(), _cfg()), "FIST")

    def test_yo_unaffected(self):
        from vision.poses import classify_pose
        lm = _curl(_base_open(), ["middle", "ring"])
        lm[4] = [0.44, 0.60]
        self.assertEqual(classify_pose(lm, _cfg()), "YO")

    def test_open_unaffected(self):
        from vision.poses import classify_pose
        self.assertEqual(classify_pose(_base_open(), _cfg()), "OPEN")


class TestPeaceDirection(unittest.TestCase):
    def test_right_is_up(self):
        from vision.poses import peace_direction
        self.assertEqual(peace_direction(_peace()), 1)

    def test_left_is_down(self):
        from vision.poses import peace_direction
        lm = _peace(index_tip=(0.20, 0.46), middle_tip=(0.26, 0.46))
        self.assertEqual(peace_direction(lm), -1)

    def test_split_is_zero(self):
        from vision.poses import peace_direction
        lm = _peace(index_tip=(0.68, 0.46), middle_tip=(0.26, 0.46))
        self.assertEqual(peace_direction(lm), 0)

    def test_short_reach_is_zero(self):
        from vision.poses import peace_direction
        lm = _peace(index_tip=(0.48, 0.46), middle_tip=(0.54, 0.46))
        self.assertEqual(peace_direction(lm), 0)

    def test_vertical_is_zero(self):
        from vision.poses import peace_direction
        lm = _curl(_base_open(), ["ring", "pinky"])
        lm[4] = [0.44, 0.60]
        self.assertEqual(peace_direction(lm), 0)


class TestAnyHandOrientation(unittest.TestCase):
    """Pose must not require an upright (fingers-image-up) hand.

    In-plane rotation about the wrist preserves the tip/pip distance
    ratios the classifier uses, so OPEN/FIST must hold at every angle.
    """

    def test_open_at_every_orientation(self):
        from vision.poses import classify_pose
        for deg in _ORIENTATIONS:
            with self.subTest(deg=deg):
                self.assertEqual(
                    classify_pose(_rotate(_base_open(), deg), _cfg()),
                    "OPEN")

    def test_fist_at_every_orientation(self):
        from vision.poses import classify_pose
        for deg in _ORIENTATIONS:
            with self.subTest(deg=deg):
                self.assertEqual(
                    classify_pose(_rotate(_fist(), deg), _cfg()),
                    "FIST")

    def test_unknown_at_every_orientation(self):
        from vision.poses import classify_pose
        partial = _curl(_base_open(), ["middle", "ring", "pinky"])
        for deg in _ORIENTATIONS:
            with self.subTest(deg=deg):
                self.assertEqual(
                    classify_pose(_rotate(partial, deg), _cfg()),
                    "UNKNOWN")


def _rotate_pixels_norm(lm, deg, width=640, height=480):
    """Rotate in pixel space then re-normalize by W/H (real camera math).

    Normalized coords are anisotropic (x/640, y/480), so a rotation in
    normalized space is NOT the same as a rotation of the physical hand.
    """
    px = np.asarray(lm, dtype=np.float64) * np.array(
        [float(width), float(height)])
    wrist = px[0].copy()
    rel = px - wrist
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = np.array([[c, -s], [s, c]])
    px = rel @ rot.T + wrist
    return px / np.array([float(width), float(height)])


class TestCameraAnisotropyOrientation(unittest.TestCase):
    """Same poses after a REAL 640x480 pixel-space rotation + renormalize."""

    def test_open_horizontal_and_diagonal(self):
        from vision.poses import classify_pose
        for deg in (45.0, 90.0, 135.0, -45.0, -90.0, 180.0):
            with self.subTest(deg=deg):
                self.assertEqual(
                    classify_pose(
                        _rotate_pixels_norm(_base_open(), deg), _cfg()),
                    "OPEN")

    def test_fist_horizontal_and_diagonal(self):
        from vision.poses import classify_pose
        for deg in (45.0, 90.0, 135.0, -45.0, -90.0, 180.0):
            with self.subTest(deg=deg):
                self.assertEqual(
                    classify_pose(
                        _rotate_pixels_norm(_fist(), deg), _cfg()),
                    "FIST")

    def test_partial_stays_unknown_when_rotated_in_pixels(self):
        from vision.poses import classify_pose
        partial = _curl(_base_open(), ["middle", "ring", "pinky"])
        for deg in (45.0, 90.0, -90.0):
            with self.subTest(deg=deg):
                self.assertEqual(
                    classify_pose(
                        _rotate_pixels_norm(partial, deg), _cfg()),
                    "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
