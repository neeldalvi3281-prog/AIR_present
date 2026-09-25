"""Step-model rotation tests: committed angle, 30 deg steps, no re-arm.

Angles fed directly in degrees. mirror_sign=+1 keeps tests in physical
space (positive = clockwise); dedicated mirror tests use the default -1.
"""
import unittest

CW = "CLOCKWISE_ROTATION_CONFIRMED"
ACW = "ANTICLOCKWISE_ROTATION_CONFIRMED"


def _cfg(**over):
    cfg = {"step_degrees": 30.0, "confirm_frames": 4,
           "hysteresis_degrees": 3.0, "step_action_spacing_ms": 120,
           "open_required": True}
    cfg.update(over)
    return cfg


def _tracker(**over):
    from vision.rotation import RotationTracker
    mirror = over.pop("mirror_sign", 1)
    return RotationTracker(_cfg(**over), mirror_sign=mirror)


def _feed(tr, angles, t0=0.0, dt=1 / 30.0):
    """Feed angles; return flat list of all emitted events."""
    out = []
    t = t0
    for a in angles:
        out += tr.update(a, t)
        t += dt
    return out


def _feed_frames(tr, angles, t0=0.0, dt=1 / 30.0):
    """Feed angles; return per-frame event lists (burst detection)."""
    out = []
    t = t0
    for a in angles:
        out.append(tr.update(a, t))
        t += dt
    return out


class TestStep(unittest.TestCase):
    def test_sub_step_no_action(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [0.0] + [29.0] * 10), [])

    def test_exact_step_fires_once(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [0.0] + [30.0] * 4), [CW])

    def test_cumulative_60_two_events(self):
        tr = _tracker()
        ev = _feed(tr, [0.0] + [30.0] * 4 + [60.0] * 4)
        self.assertEqual(ev, [CW, CW])

    def test_cumulative_90_three_events_no_hold_repeat(self):
        tr = _tracker()
        ev = _feed(tr, [0.0] + [30.0] * 4 + [60.0] * 4 + [90.0] * 30)
        self.assertEqual(ev, [CW, CW, CW])

    def test_anticlockwise_triple(self):
        tr = _tracker()
        ev = _feed(tr, [0.0] + [-30.0] * 4 + [-60.0] * 4 + [-90.0] * 4)
        self.assertEqual(ev, [ACW, ACW, ACW])

    def test_hold_at_step_exactly_one(self):
        tr = _tracker()
        ev = _feed(tr, [0.0] + [30.0] * 90)  # 3 s held past the step
        self.assertEqual(ev, [CW])

    def test_fewer_than_confirm_frames_no_fire(self):
        tr = _tracker()
        self.assertEqual(_feed(tr, [0.0, 30.0, 30.0, 30.0, 0.0, 0.0]), [])


class TestCommitted(unittest.TestCase):
    def test_committed_advances_per_step(self):
        tr = _tracker()
        _feed(tr, [0.0] + [30.0] * 4)
        self.assertAlmostEqual(tr.committed, 30.0, places=9)

    def test_no_return_to_neutral_needed(self):
        # After firing at +30, staying rotated and pushing further fires
        # again without ever returning near the original baseline.
        tr = _tracker()
        ev = _feed(tr, [10.0] + [40.0] * 4 + [70.0] * 4)
        self.assertEqual(ev, [CW, CW])

    def test_reversal_from_committed(self):
        tr = _tracker()
        ev = _feed(tr, [0.0] + [30.0] * 4 + [60.0] * 4 + [30.0] * 4
                      + [0.0] * 4 + [-30.0] * 4)
        self.assertEqual(ev, [CW, CW, ACW, ACW, ACW])

    def test_flip_mid_candidate_no_double_fire(self):
        tr = _tracker()
        ev = _feed(tr, [0.0, 30.0, 30.0, -30.0, -30.0, -30.0, -30.0])
        self.assertEqual(ev, [ACW])


class TestJitter(unittest.TestCase):
    def test_hover_at_boundary_fires_at_most_once(self):
        tr = _tracker()
        seq = [0.0] + [28.0, 29.0, 30.0, 29.0, 30.0, 29.0] * 8
        ev = _feed(tr, seq)
        self.assertLessEqual(len(ev), 1)

    def test_jitter_around_neutral_no_fire(self):
        import random
        tr = _tracker()
        rnd = random.Random(4)
        ev = _feed(tr, [rnd.uniform(-3, 3) for _ in range(120)])
        self.assertEqual(ev, [])

    def test_wraparound_step_across_180(self):
        tr = _tracker()
        ev = _feed(tr, [170.0] + [-160.0] * 4)  # +30 wrapped
        self.assertEqual(ev, [CW])


class TestFastRotation(unittest.TestCase):
    def test_jump_two_steps_no_same_frame_burst(self):
        tr = _tracker()
        frames = _feed_frames(tr, [0.0] + [64.0] * 10)
        flat = [e for f in frames for e in f]
        self.assertEqual(flat, [CW, CW])
        for f in frames:
            self.assertLessEqual(len(f), 1)  # at most one event per frame

    def test_spacing_blocks_immediate_refire(self):
        tr = _tracker(step_action_spacing_ms=1000)
        ev = _feed(tr, [0.0] + [30.0] * 4 + [60.0] * 10)
        self.assertEqual(ev, [CW])  # second step inside 1000 ms: held back

    def test_spacing_elapsed_allows_next_step(self):
        tr = _tracker(step_action_spacing_ms=1000)
        t = 0.0
        dt = 1 / 30.0

        def feed(angles):
            nonlocal t
            out = []
            for a in angles:
                out += tr.update(a, t)
                t += dt
            return out

        self.assertEqual(feed([0.0] + [30.0] * 4), [CW])
        t += 1.2  # past spacing
        self.assertEqual(feed([60.0] * 4), [CW])


class TestSession(unittest.TestCase):
    def test_none_resets_without_event(self):
        tr = _tracker()
        _feed(tr, [0.0, 30.0, 30.0])
        self.assertEqual(tr.update(None, 1.0), [])
        self.assertEqual(tr.state, "IDLE")

    def test_reappearance_gets_fresh_committed_no_stale(self):
        tr = _tracker()
        _feed(tr, [45.0] + [75.0] * 4)  # one step consumed pre-loss
        tr.update(None, 1.0)
        # Reappear at 0: 0 is the NEW committed, not -75 of rotation.
        ev = _feed(tr, [0.0] * 8, t0=2.0)
        self.assertEqual(ev, [])

    def test_cancel_drops_candidate_keeps_session(self):
        tr = _tracker()
        _feed(tr, [0.0, 30.0, 30.0])  # partial candidate
        tr.cancel()
        self.assertEqual(tr.state, "SESSION")
        self.assertAlmostEqual(tr.committed, 0.0, places=9)
        ev = _feed(tr, [30.0] * 4, t0=0.2)
        self.assertEqual(ev, [CW])

    def test_tilted_start_is_committed(self):
        tr = _tracker()
        ev = _feed(tr, [40.0] * 4 + [70.0] * 4)
        self.assertEqual(ev, [CW])
        tr.reset()
        ev = _feed(tr, [40.0] * 10 + [65.0] * 10)  # 25 deg: under 30
        self.assertEqual(ev, [])


class TestAnyStartOrientation(unittest.TestCase):
    """No upright-hand requirement: committed may be any absolute angle."""

    def test_steps_from_horizontal_hand(self):
        for start in (90.0, -90.0, 180.0, -175.0, 45.0, -30.0):
            with self.subTest(start=start):
                tr = _tracker()
                ev = _feed(tr, [start] * 2 + [start + 30.0] * 4)
                self.assertEqual(ev, [CW])

    def test_anticlockwise_from_horizontal_hand(self):
        tr = _tracker()
        ev = _feed(tr, [90.0] * 2 + [60.0] * 4)
        self.assertEqual(ev, [ACW])

    def test_pose_gate_does_not_encode_upright(self):
        # Any orientation is fine as long as the pose label is OPEN;
        # the gate itself is orientation-blind.
        from vision.rotation import pose_allows_navigation
        self.assertTrue(pose_allows_navigation("OPEN"))


class TestMirror(unittest.TestCase):
    def test_default_mirror_image_ccw_is_physical_cw(self):
        from vision.rotation import RotationTracker
        tr = RotationTracker(_cfg())  # default mirror_sign = -1
        ev = _feed(tr, [0.0] + [-30.0] * 4)
        self.assertEqual(ev, [CW])

    def test_default_mirror_image_cw_is_physical_acw(self):
        from vision.rotation import RotationTracker
        tr = RotationTracker(_cfg())
        ev = _feed(tr, [0.0] + [30.0] * 4)
        self.assertEqual(ev, [ACW])

    def test_default_mirror_multi_step_stays_clockwise(self):
        # Regression: committed must advance in LANDMARK space
        # (mirror_sign * dir * step), else reference runs away and
        # wrap flips CW -> ACW on later steps.
        from vision.rotation import RotationTracker
        tr = RotationTracker(_cfg())
        ev = _feed(tr, [0.0] + [-30.0] * 6 + [-60.0] * 6 + [-90.0] * 20)
        self.assertEqual(ev, [CW, CW, CW])

    def test_default_mirror_hold_still_after_step_fires_once(self):
        # Regression: stationary palm at the stepped angle must not
        # keep paging (wrong-signed committed used to grow forever).
        from vision.rotation import RotationTracker
        tr = RotationTracker(_cfg())
        ev = _feed(tr, [0.0] + [-31.0] * 90)  # 3 s hold past one step
        self.assertEqual(ev, [CW])

    def test_default_mirror_hold_negative_fires_once(self):
        from vision.rotation import RotationTracker
        tr = RotationTracker(_cfg())
        ev = _feed(tr, [0.0] + [31.0] * 90)
        self.assertEqual(ev, [ACW])

    def test_default_mirror_long_cw_never_flips_to_acw(self):
        from vision.rotation import RotationTracker
        tr = RotationTracker(_cfg())
        # Sweep well past ±180 in landmark space, one held angle per step.
        angles = [0.0]
        for i in range(1, 10):
            angles += [-30.0 * i] * 6
        angles += [-30.0 * 9] * 40  # hold at last step
        ev = _feed(tr, angles)
        self.assertEqual(ev, [CW] * 9)


class TestPoseGate(unittest.TestCase):
    def test_open_allows_navigation(self):
        from vision.rotation import pose_allows_navigation
        self.assertTrue(pose_allows_navigation("OPEN"))

    def test_fist_and_unknown_block_navigation(self):
        from vision.rotation import pose_allows_navigation
        self.assertFalse(pose_allows_navigation("FIST"))
        self.assertFalse(pose_allows_navigation("UNKNOWN"))
        self.assertFalse(pose_allows_navigation(None))


class TestTelemetry(unittest.TestCase):
    def test_threshold_and_remaining(self):
        tr = _tracker()
        # Jump >= snap threshold so the filter does not lag the readout.
        _feed(tr, [0.0, 20.0])
        self.assertAlmostEqual(tr.next_clockwise, 30.0, places=9)
        self.assertAlmostEqual(tr.next_anticlockwise, -30.0, places=9)
        self.assertAlmostEqual(tr.remaining, 10.0, places=9)

    def test_jitter_angle_is_smoothed_not_raw(self):
        # Small step below snap: EMA damps a single noisy frame.
        tr = _tracker()
        _feed(tr, [0.0, 9.0])  # 9 < SNAP_DEG(10) -> alpha 0.35
        self.assertLess(tr.delta, 9.0)
        self.assertGreater(tr.delta, 0.0)


class TestRotationFrameLifecycle(unittest.TestCase):
    """OPEN-only: FIST/UNKNOWN/held reset; next OPEN is a fresh session."""

    def _frame(self, tr, pose, angle, held=False, t=0.0):
        from vision.rotation import rotation_frame
        return rotation_frame(tr, pose, held, angle, t)

    def test_open_commits_reference_once(self):
        tr = _tracker()
        self.assertEqual(self._frame(tr, "OPEN", 50.0), [])
        self.assertEqual(tr.state, "SESSION")
        self.assertAlmostEqual(tr.committed, 50.0, places=9)
        self._frame(tr, "OPEN", 52.0, t=0.05)
        self.assertAlmostEqual(tr.committed, 50.0, places=9)

    def test_fist_resets_and_next_open_is_fresh_baseline(self):
        tr = _tracker()
        self._frame(tr, "OPEN", 10.0)
        self._frame(tr, "OPEN", 40.0, t=0.05)
        # Any FIST frame (confirmed or not) ends the session.
        self.assertEqual(self._frame(tr, "FIST", 40.0, t=0.1), [])
        self.assertEqual(tr.state, "IDLE")
        self.assertIsNone(tr.committed)
        # Re-open at a different angle: new reference, no stale delta.
        self.assertEqual(self._frame(tr, "OPEN", 75.0, t=0.2), [])
        self.assertAlmostEqual(tr.committed, 75.0, places=9)
        # Holding still at re-open angle: no events.
        ev = []
        for i in range(20):
            ev += self._frame(tr, "OPEN", 75.0, t=0.25 + i / 30.0)
        self.assertEqual(ev, [])

    def test_unknown_resets_and_no_navigation(self):
        tr = _tracker()
        self._frame(tr, "OPEN", 0.0)
        self._frame(tr, "OPEN", 45.0, t=0.05)  # past step, not confirmed
        self.assertEqual(self._frame(tr, "UNKNOWN", 45.0, t=0.1), [])
        self.assertEqual(tr.state, "IDLE")
        self.assertIsNone(tr.committed)
        # UNKNOWN alone never starts a session.
        self.assertEqual(self._frame(tr, "UNKNOWN", 0.0, t=0.2), [])
        self.assertIsNone(tr.committed)

    def test_held_fist_resets_reference(self):
        from vision.rotation import rotation_frame
        tr = _tracker()
        self._frame(tr, "OPEN", 20.0)
        self.assertEqual(rotation_frame(tr, "FIST", True, 20.0, 0.5), [])
        self.assertEqual(tr.state, "IDLE")
        self._frame(tr, "OPEN", 70.0, t=0.6)
        self.assertAlmostEqual(tr.committed, 70.0, places=9)

    def test_fist_pose_before_start_does_not_open_session(self):
        tr = _tracker()
        self.assertEqual(self._frame(tr, "FIST", 0.0), [])
        self.assertEqual(tr.state, "IDLE")
        self.assertIsNone(tr.committed)

    def test_stationary_fist_at_angles_never_navigates(self):
        tr = _tracker()
        for base in (0.0, 30.0, -30.0, 60.0):
            self._frame(tr, "FIST", base, t=0.0)
            ev = []
            for i in range(30):
                ev += self._frame(tr, "FIST", base + 0.5, t=0.1 + i / 30.0)
            self.assertEqual(ev, [])
            self.assertIsNone(tr.committed)


if __name__ == "__main__":
    unittest.main()
