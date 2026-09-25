"""Shared-features equivalence: bundle path == raw path, exactly.

Guards opti §16: every consumer must return identical results whether it
receives raw landmarks or one FrameFeatures built from them.
"""
import unittest

import numpy as np


def _hands():
    from tests.test_poses import _base_open, _fist, _yo, _peace
    from tests.test_pinch import _pinch_at
    return [_base_open(), _fist(), _yo(), _peace(), _pinch_at(0.2),
            _pinch_at(0.5), np.zeros((21, 2)),
            _base_open() + np.array([0.1, -0.05])]


def _noisy(base, seed=7):
    rnd = np.random.RandomState(seed)
    return base + rnd.uniform(-0.004, 0.004, size=base.shape)


class TestEquivalence(unittest.TestCase):
    def test_bundle_matches_raw_everywhere(self):
        from vision.features import FrameFeatures
        from vision.palm import palm_angle_deg, palm_centroid, palm_scale
        from vision.pinch import pinch_distance_norm
        from vision.poses import (classify_pose, finger_states, is_fist,
                                  is_open, is_yo, is_peace, peace_direction,
                                  hand_scale_norm, thumb_extended)
        from tests.test_poses import _cfg
        cfg, ratio = _cfg(), 1.15
        for base in _hands():
            for lm in (base, _noisy(base)):
                f = FrameFeatures.from_landmarks(lm)
                self.assertEqual(palm_centroid(f), palm_centroid(lm))
                self.assertEqual(palm_angle_deg(f), palm_angle_deg(lm))
                self.assertEqual(palm_scale(f), palm_scale(lm))
                self.assertEqual(pinch_distance_norm(f),
                                 pinch_distance_norm(lm))
                self.assertEqual(hand_scale_norm(f), hand_scale_norm(lm))
                self.assertEqual(finger_states(f, ratio),
                                 finger_states(lm, ratio))
                self.assertEqual(thumb_extended(f, cfg),
                                 thumb_extended(lm, cfg))
                self.assertEqual(is_fist(f, cfg), is_fist(lm, cfg))
                self.assertEqual(is_open(f, cfg), is_open(lm, cfg))
                self.assertEqual(is_yo(f, cfg), is_yo(lm, cfg))
                self.assertEqual(is_peace(f, cfg), is_peace(lm, cfg))
                self.assertEqual(peace_direction(f), peace_direction(lm))
                self.assertEqual(classify_pose(f, cfg), classify_pose(lm, cfg))

    def test_zero_copy_passthrough(self):
        from vision.features import FrameFeatures
        arr = np.zeros((21, 2), dtype=np.float64)
        self.assertIs(FrameFeatures.from_landmarks(arr).pts, arr)

    def test_derive_accepts_bundle(self):
        from vision.features import FrameFeatures, derive
        from tests.test_poses import _base_open
        lm = _base_open()
        f = FrameFeatures.from_landmarks(lm)
        got, want = derive(f, 640, 480), derive(lm, 640, 480)
        self.assertEqual(set(got), set(want))
        for key in got:
            if key == "landmarks_px":
                self.assertTrue((got[key] == want[key]).all())
            else:
                self.assertEqual(got[key], want[key])


if __name__ == "__main__":
    unittest.main()
