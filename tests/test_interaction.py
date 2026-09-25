"""Reform tests: whole-hand mouse mapping, margins, click path.

No SendInput, no camera: mouse sender is faked.
"""
import unittest


class FakeMouse:
    def __init__(self):
        self.moved_to = []
        self.clicks = 0
        self.downs = 0
        self.ups = 0
        self.button_held = False

    def move(self, x, y):
        self.moved_to.append((x, y))

    def click(self):
        self.clicks += 1

    def left_down(self):
        self.downs += 1
        self.button_held = True

    def left_up(self):
        self.ups += 1
        self.button_held = False


class TestMouse(unittest.TestCase):
    def test_absolute_mapping(self):
        from actions.mouse import MouseAdapter
        self.assertEqual(MouseAdapter.to_absolute(0.0, 0.0), (0, 0))
        self.assertEqual(MouseAdapter.to_absolute(1.0, 1.0), (65535, 65535))
        self.assertEqual(MouseAdapter.to_absolute(0.5, 0.5), (32767, 32767))

    def test_absolute_clamped(self):
        from actions.mouse import MouseAdapter
        self.assertEqual(MouseAdapter.to_absolute(-1.0, 2.0), (0, 65535))

    def test_move_uses_absolute_flag(self):
        from actions.mouse import MouseAdapter, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_MOVE
        calls = []
        m = MouseAdapter(sender=lambda dx, dy, fl: calls.append((dx, dy, fl)))
        m.move(0.25, 0.75)
        (dx, dy, fl), = calls
        self.assertEqual((dx, dy), (16383, 49151))
        self.assertTrue(fl & MOUSEEVENTF_ABSOLUTE and fl & MOUSEEVENTF_MOVE)

    def test_click_is_down_up_pair(self):
        from actions.mouse import MouseAdapter, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
        calls = []
        m = MouseAdapter(sender=lambda dx, dy, fl: calls.append(fl))
        m.click()
        self.assertEqual(calls, [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP])

    def test_left_down_up_hold(self):
        from actions.mouse import MouseAdapter, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
        calls = []
        m = MouseAdapter(sender=lambda dx, dy, fl: calls.append(fl))
        self.assertTrue(m.left_down())
        self.assertTrue(m.button_held)
        self.assertTrue(m.left_up())
        self.assertFalse(m.button_held)
        self.assertEqual(calls, [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP])

    def test_down_idempotent_no_double_send(self):
        from actions.mouse import MouseAdapter
        calls = []
        m = MouseAdapter(sender=lambda dx, dy, fl: calls.append(fl))
        self.assertTrue(m.left_down())
        self.assertFalse(m.left_down())  # already held: no second SendInput
        self.assertEqual(len(calls), 1)

    def test_up_without_down_no_send(self):
        from actions.mouse import MouseAdapter
        calls = []
        m = MouseAdapter(sender=lambda dx, dy, fl: calls.append(fl))
        self.assertFalse(m.left_up())
        self.assertEqual(calls, [])

    def test_release_ups_only_if_held(self):
        from actions.mouse import MouseAdapter
        calls = []
        m = MouseAdapter(sender=lambda dx, dy, fl: calls.append(fl))
        m.release()
        self.assertEqual(calls, [])
        m.left_down()
        m.release()  # shutdown safety: never leave the button stuck
        self.assertFalse(m.button_held)
        self.assertEqual(len(calls), 2)

    def test_wheel_forward_sends_positive_delta(self):
        from actions.mouse import (MouseAdapter, MOUSEEVENTF_WHEEL,
                                   WHEEL_DELTA)
        calls = []
        m = MouseAdapter(wheel_sender=lambda delta: calls.append(delta))
        self.assertTrue(m.wheel(+1))
        self.assertEqual(calls, [WHEEL_DELTA])
        self.assertEqual(m.wheels, 1)

    def test_wheel_back_sends_negative_delta(self):
        from actions.mouse import MouseAdapter, WHEEL_DELTA
        calls = []
        m = MouseAdapter(wheel_sender=lambda delta: calls.append(delta))
        self.assertTrue(m.wheel(-1))
        self.assertEqual(calls, [-WHEEL_DELTA])

    def test_move_sender_untouched_by_wheel_path(self):
        from actions.mouse import MouseAdapter
        moves, wheels = [], []
        m = MouseAdapter(sender=lambda dx, dy, fl: moves.append((dx, dy, fl)),
                         wheel_sender=wheels.append)
        m.move(0.5, 0.5)
        m.wheel(+1)
        self.assertEqual(len(moves), 1)
        self.assertEqual(len(wheels), 1)


class TestMargins(unittest.TestCase):
    """Legacy margin mapping superseded by the fixed control box
    (see tests/test_control_box.py); this class pins the box math
    through the adapter pipeline."""

    def _map(self):
        return {"left": 0.1, "right": 0.9, "top": 0.1, "bottom": 0.9}

    def test_window_edges_map_to_screen_edges(self):
        from actions.mouse import apply_control_box
        m = self._map()
        self.assertEqual(apply_control_box(0.1, 0.1, m), (0.0, 0.0))
        self.assertEqual(apply_control_box(0.9, 0.9, m), (1.0, 1.0))

    def test_center_maps_to_center(self):
        from actions.mouse import apply_control_box
        x, y = apply_control_box(0.5, 0.5, self._map())
        self.assertAlmostEqual(x, 0.5, places=9)
        self.assertAlmostEqual(y, 0.5, places=9)

    def test_outside_window_clamped(self):
        from actions.mouse import apply_control_box
        self.assertEqual(apply_control_box(-0.5, 1.5, self._map()), (0.0, 1.0))
        self.assertEqual(apply_control_box(0.0, 1.0, self._map()), (0.0, 1.0))

    def test_degenerate_window_safe(self):
        from actions.mouse import apply_control_box
        x, y = apply_control_box(0.5, 0.5, {"left": 0.5, "right": 0.5,
                                            "top": 0.2, "bottom": 0.2})
        self.assertTrue(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)

    def test_mapped_point_reaches_absolute_space(self):
        from actions.mouse import MouseAdapter, apply_control_box
        x, y = apply_control_box(0.1, 0.9, self._map())
        self.assertEqual(MouseAdapter.to_absolute(x, y), (0, 65535))


class TestPalmMousePipeline(unittest.TestCase):
    def test_palm_centroid_flows_to_absolute(self):
        """Palm centroid -> control box -> absolute: pure functions compose."""
        import numpy as np
        from actions.mouse import MouseAdapter, apply_control_box
        from vision.palm import palm_centroid
        lm = np.zeros((21, 2))
        for i, (x, y) in zip((0, 5, 9, 13, 17),
                             [(0.4, 0.4), (0.5, 0.4), (0.5, 0.5),
                              (0.5, 0.6), (0.4, 0.5)]):
            lm[i] = [x, y]
        cx, cy = palm_centroid(lm)
        sx, sy = apply_control_box(cx, cy, {"left": 0.0, "right": 1.0,
                                            "top": 0.0, "bottom": 1.0})
        ax, ay = MouseAdapter.to_absolute(sx, sy)
        self.assertEqual((ax, ay), (int(0.46 * 65535), int(0.48 * 65535)))

    def test_no_hand_means_no_mouse_call(self):
        # Architectural: app.py must only call mouse.move when a hand is
        # detected. The adapter itself never invents positions.
        from actions.mouse import MouseAdapter
        calls = []
        m = MouseAdapter(sender=lambda dx, dy, fl: calls.append((dx, dy, fl)))
        self.assertEqual(calls, [])
        self.assertEqual(m.moves, 0)


if __name__ == "__main__":
    unittest.main()
