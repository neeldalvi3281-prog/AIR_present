"""Fixed virtual mouse-control box: corners, center, clamps, monotonicity.

Box defaults: left=0.07, right=0.95, top=0.08, bottom=0.92 (normalized
camera coords). No webcam, no SendInput.
"""
import time
import unittest

BOX = {"left": 0.07, "right": 0.95, "top": 0.08, "bottom": 0.92}


def _map(**over):
    m = dict(BOX)
    m.update(over)
    return m


class TestCorners(unittest.TestCase):
    def test_four_corners(self):
        from actions.mouse import MouseAdapter, apply_control_box
        cases = [
            ((0.07, 0.08), (0, 0)),
            ((0.95, 0.08), (65535, 0)),
            ((0.07, 0.92), (0, 65535)),
            ((0.95, 0.92), (65535, 65535)),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                x, y = apply_control_box(inp[0], inp[1], _map())
                self.assertEqual(MouseAdapter.to_absolute(x, y), expected)

    def test_center_approximately_screen_center(self):
        from actions.mouse import MouseAdapter, apply_control_box
        x, y = apply_control_box(0.51, 0.50, _map())
        ax, ay = MouseAdapter.to_absolute(x, y)
        self.assertLess(abs(ax - 32768) / 65535, 0.02)
        self.assertEqual(ay, 32767)


class TestOutsideClamps(unittest.TestCase):
    def test_outside_left_clamps_zero(self):
        from actions.mouse import apply_control_box
        x, _ = apply_control_box(0.0, 0.5, _map())
        self.assertEqual(x, 0.0)

    def test_outside_right_clamps_one(self):
        from actions.mouse import apply_control_box
        x, _ = apply_control_box(1.0, 0.5, _map())
        self.assertEqual(x, 1.0)

    def test_outside_top_clamps_zero(self):
        from actions.mouse import apply_control_box
        _, y = apply_control_box(0.5, 0.0, _map())
        self.assertEqual(y, 0.0)

    def test_outside_bottom_clamps_one(self):
        from actions.mouse import apply_control_box
        _, y = apply_control_box(0.5, 1.0, _map())
        self.assertEqual(y, 1.0)

    def test_negative_and_overshoot_clamp(self):
        from actions.mouse import apply_control_box
        self.assertEqual(apply_control_box(-0.5, 1.5, _map()), (0.0, 1.0))


class TestMonotonic(unittest.TestCase):
    def test_left_to_right_increases(self):
        from actions.mouse import apply_control_box
        xs = [apply_control_box(0.07 + i * 0.01, 0.5, _map())[0]
              for i in range(89)]
        self.assertTrue(all(b > a for a, b in zip(xs, xs[1:])))
        self.assertEqual(xs[0], 0.0)
        self.assertEqual(xs[-1], 1.0)

    def test_bottom_to_top_decreases_screen_y(self):
        from actions.mouse import MouseAdapter, apply_control_box
        ys = [MouseAdapter.to_absolute(
            *apply_control_box(0.5, 0.92 - i * 0.01, _map()))[1]
            for i in range(85)]
        self.assertTrue(all(b < a for a, b in zip(ys, ys[1:])))
        self.assertEqual(ys[0], 65535)
        self.assertEqual(ys[-1], 0)


class TestDegenerate(unittest.TestCase):
    def test_degenerate_span_falls_back_to_clamp(self):
        from actions.mouse import apply_control_box
        x, y = apply_control_box(0.5, 0.5,
                                 _map(left=0.5, right=0.5, top=0.5, bottom=0.5))
        self.assertTrue(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)


class TestPerf(unittest.TestCase):
    def test_mapping_under_budget(self):
        from actions.mouse import apply_control_box
        m = _map()
        N = 20000
        t0 = time.perf_counter()
        for i in range(N):
            apply_control_box(0.07 + (i % 88) / 100.0, 0.5, m)
        dt = (time.perf_counter() - t0) / N * 1000.0
        self.assertLess(dt, 0.05)  # spec §17 budget


if __name__ == "__main__":
    unittest.main()
