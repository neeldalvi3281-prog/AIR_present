"""Control-box overlay: static rect, label, palm dot, readout lines."""
import unittest

import numpy as np


def _frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestControlBoxDraw(unittest.TestCase):
    def test_box_edges_drawn_in_orange(self):
        from ui.developer import CONTROL_COLOR, draw
        frame = _frame()
        box = (44, 38, 608, 441)  # 0.07/0.08/0.95/0.92 of 640x480
        draw(frame, None, [], control_box=box)
        edge = frame[38, 300].tolist()
        self.assertEqual(edge, list(CONTROL_COLOR))
        self.assertEqual(frame[200, 200].tolist(), [0, 0, 0])  # inside untouched

    def test_no_box_no_drawing(self):
        from ui.developer import CONTROL_COLOR, draw
        frame = _frame()
        draw(frame, None, [])
        self.assertTrue((frame == 0).all())
        self.assertNotIn(tuple(CONTROL_COLOR),
                         {tuple(px) for px in frame.reshape(-1, 3)})

    def test_palm_dot_drawn(self):
        from ui.developer import CONTROL_COLOR, draw
        frame = _frame()
        draw(frame, None, [], palm_px=(100, 100))
        ring = frame[94, 100].tolist()
        self.assertEqual(ring, list(CONTROL_COLOR))

    def test_box_geometry_matches_mapping(self):
        m = {"left": 0.07, "right": 0.95, "top": 0.08, "bottom": 0.92}
        w, h = 640, 480
        box = (m["left"] * w, m["top"] * h, m["right"] * w, m["bottom"] * h)
        for got, want in zip(box, (44.8, 38.4, 608.0, 441.6)):
            self.assertAlmostEqual(got, want, places=9)


class TestPalmReadout(unittest.TestCase):
    def test_absolute_screen_pixels_shown(self):
        from ui.developer import build_palm_lines
        lines = build_palm_lines((0.51, 0.50), (0.51, 0.50), 0.0, True)
        self.assertIn("X = 33422", lines[1])
        self.assertIn("Y = 32767", lines[1])
        self.assertIn("X = 0.510", lines[0])

    def test_no_hand_placeholder(self):
        from ui.developer import build_palm_lines
        self.assertEqual(build_palm_lines(None, (0, 0), 0.0, False),
                         ["Palm: --  Screen: --"])


if __name__ == "__main__":
    unittest.main()
