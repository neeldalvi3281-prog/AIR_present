"""Reform tests: rotation/pinch mapping, dry-run/disabled/live, duplicates.

Keyboard and mouse adapters are always mocked: no test emits real input.
"""
import unittest


class FakeKeyboard:
    def __init__(self):
        self.presses = []
        self.events = []  # (name, "down"/"up") in order
        self._held = set()
        self.closed = False

    def press_key(self, name):
        self.presses.append(name)

    def key_down(self, name):
        self.events.append((name, "down"))
        self._held.add(name)

    def key_up(self, name):
        self.events.append((name, "up"))
        self._held.discard(name)

    @property
    def held(self):
        return frozenset(self._held)

    def close(self):
        self.closed = True


class FakeMouse:
    def __init__(self):
        self.clicks = 0
        self.downs = 0
        self.ups = 0
        self.wheel_ticks = []  # direction per wheel() call
        self.button_held = False

    def click(self):
        self.clicks += 1

    def left_down(self):
        if self.button_held:
            return False
        self.downs += 1
        self.button_held = True
        return True

    def left_up(self):
        if not self.button_held:
            return False
        self.ups += 1
        self.button_held = False
        return True

    def wheel(self, direction):
        self.wheel_ticks.append(1 if direction > 0 else -1)
        return True

    def release(self):
        if self.button_held:
            self.left_up()


def _cfg(enabled=False, dry_run=True):
    return {"enabled": enabled, "dry_run": dry_run,
            "duplicate_window_ms": 50,
            "mapping": {
                "CLOCKWISE_ROTATION_CONFIRMED": "PREVIOUS_SLIDE",
                "ANTICLOCKWISE_ROTATION_CONFIRMED": "NEXT_SLIDE",
                "PINCH_STARTED": "LEFT_DOWN",
                "PINCH_RELEASED": "LEFT_UP",
                "ZOOM_IN_STEP": "ZOOM_IN",
                "ZOOM_OUT_STEP": "ZOOM_OUT",
                "SCROLL_UP_STEP": "SCROLL_UP",
                "SCROLL_DOWN_STEP": "SCROLL_DOWN",
            },
            "keys": {"NEXT_SLIDE": "RIGHT", "PREVIOUS_SLIDE": "LEFT"}}


def _dispatcher(enabled=False, dry_run=True):
    from actions.dispatcher import ActionDispatcher
    kb, ms = FakeKeyboard(), FakeMouse()
    return ActionDispatcher(_cfg(enabled, dry_run), kb, mouse=ms), kb, ms


class TestMapping(unittest.TestCase):
    def test_clockwise_maps_previous_slide(self):
        d, _, _ = _dispatcher()
        rec = d.dispatch("CLOCKWISE_ROTATION_CONFIRMED", 1.0)
        self.assertEqual(rec.action, "PREVIOUS_SLIDE")
        self.assertEqual(rec.key, "LEFT_ARROW")

    def test_anticlockwise_maps_next_slide(self):
        d, _, _ = _dispatcher()
        rec = d.dispatch("ANTICLOCKWISE_ROTATION_CONFIRMED", 1.0)
        self.assertEqual(rec.action, "NEXT_SLIDE")
        self.assertEqual(rec.key, "RIGHT_ARROW")

    def test_pinch_start_maps_left_down(self):
        d, _, _ = _dispatcher()
        rec = d.dispatch("PINCH_STARTED", 1.0)
        self.assertEqual(rec.action, "LEFT_DOWN")
        self.assertEqual(rec.key, "LEFT_DOWN")

    def test_pinch_release_maps_left_up(self):
        d, _, _ = _dispatcher()
        rec = d.dispatch("PINCH_STARTED", 1.0)
        rec = d.dispatch("PINCH_RELEASED", 1.2)
        self.assertEqual(rec.action, "LEFT_UP")
        self.assertEqual(rec.key, "LEFT_UP")

    def test_fist_events_unmapped(self):
        d, kb, ms = _dispatcher(enabled=True, dry_run=False)
        self.assertIsNone(d.dispatch("FIST_STARTED", 1.0))
        self.assertIsNone(d.dispatch("FIST_RELEASED", 1.1))
        self.assertEqual(kb.presses, [])
        self.assertEqual(ms.downs, 0)
        self.assertEqual(ms.ups, 0)

    def test_zoom_in_maps_ctrl_wheel(self):
        d, _, _ = _dispatcher()
        rec = d.dispatch("ZOOM_IN_STEP", 1.0)
        self.assertEqual(rec.action, "ZOOM_IN")
        self.assertEqual(rec.key, "CTRL+WHEEL+")

    def test_zoom_out_maps_ctrl_wheel(self):
        d, _, _ = _dispatcher()
        rec = d.dispatch("ZOOM_OUT_STEP", 1.0)
        self.assertEqual(rec.action, "ZOOM_OUT")
        self.assertEqual(rec.key, "CTRL+WHEEL-")


class TestModes(unittest.TestCase):
    def test_dry_run_records_but_never_sends(self):
        d, kb, ms = _dispatcher(enabled=True, dry_run=True)
        rec = d.dispatch("PINCH_STARTED", 1.0)
        self.assertEqual(rec.mode, "DRY_RUN")
        self.assertFalse(rec.sent)
        self.assertEqual(kb.presses, [])
        self.assertEqual(ms.downs, 0)
        rec = d.dispatch("PINCH_RELEASED", 1.5)
        self.assertFalse(rec.sent)
        self.assertEqual(ms.ups, 0)

    def test_disabled_observes_but_never_sends(self):
        d, kb, ms = _dispatcher(enabled=False, dry_run=False)
        rec = d.dispatch("PINCH_STARTED", 1.0)
        self.assertEqual(rec.mode, "OBSERVED")
        self.assertFalse(rec.sent)
        self.assertEqual(kb.presses, [])
        self.assertEqual(ms.downs, 0)
        rec = d.dispatch("PINCH_RELEASED", 1.5)
        self.assertFalse(rec.sent)
        self.assertEqual(ms.ups, 0)

    def test_live_sends_keys_and_hold(self):
        d, kb, ms = _dispatcher(enabled=True, dry_run=False)
        rec = d.dispatch("CLOCKWISE_ROTATION_CONFIRMED", 1.0)
        self.assertTrue(rec.sent)
        self.assertEqual(kb.presses, ["LEFT"])
        rec = d.dispatch("ANTICLOCKWISE_ROTATION_CONFIRMED", 2.0)
        self.assertEqual(kb.presses, ["LEFT", "RIGHT"])
        rec = d.dispatch("PINCH_STARTED", 3.0)
        self.assertTrue(rec.sent)
        self.assertEqual(ms.downs, 1)
        self.assertTrue(ms.button_held)
        rec = d.dispatch("PINCH_RELEASED", 4.0)
        self.assertTrue(rec.sent)
        self.assertEqual(ms.ups, 1)
        self.assertFalse(ms.button_held)

    def test_up_without_prior_down_safe(self):
        d, _, ms = _dispatcher(enabled=True, dry_run=False)
        rec = d.dispatch("PINCH_RELEASED", 1.0)
        self.assertEqual(rec.action, "LEFT_UP")
        self.assertFalse(rec.sent)  # nothing held: no SendInput
        self.assertEqual(ms.ups, 0)

    def test_hold_without_mouse_adapter_never_sends(self):
        from actions.dispatcher import ActionDispatcher
        d = ActionDispatcher(_cfg(enabled=True, dry_run=False),
                             FakeKeyboard(), mouse=None)
        rec = d.dispatch("PINCH_STARTED", 1.0)
        self.assertEqual(rec.action, "LEFT_DOWN")
        self.assertFalse(rec.sent)
        rec = d.dispatch("PINCH_RELEASED", 2.0)
        self.assertEqual(rec.action, "LEFT_UP")
        self.assertFalse(rec.sent)


class TestZoomDispatch(unittest.TestCase):
    def test_live_zoom_in_ctrl_down_wheel_up_sequence(self):
        d, kb, ms = _dispatcher(enabled=True, dry_run=False)
        rec = d.dispatch("ZOOM_IN_STEP", 1.0)
        self.assertTrue(rec.sent)
        self.assertEqual(kb.events, [("CTRL", "down"), ("CTRL", "up")])
        self.assertEqual(ms.wheel_ticks, [1])
        self.assertEqual(kb.held, frozenset())  # Ctrl never stuck

    def test_live_zoom_out_wheel_down(self):
        d, kb, ms = _dispatcher(enabled=True, dry_run=False)
        rec = d.dispatch("ZOOM_OUT_STEP", 1.0)
        self.assertTrue(rec.sent)
        self.assertEqual(ms.wheel_ticks, [-1])
        self.assertEqual(kb.held, frozenset())

    def test_dry_run_zoom_never_sends(self):
        d, kb, ms = _dispatcher(enabled=True, dry_run=True)
        rec = d.dispatch("ZOOM_IN_STEP", 1.0)
        self.assertEqual(rec.mode, "DRY_RUN")
        self.assertFalse(rec.sent)
        self.assertEqual(kb.events, [])
        self.assertEqual(ms.wheel_ticks, [])

    def test_disabled_zoom_never_sends(self):
        d, kb, ms = _dispatcher(enabled=False, dry_run=False)
        rec = d.dispatch("ZOOM_OUT_STEP", 1.0)
        self.assertEqual(rec.mode, "OBSERVED")
        self.assertFalse(rec.sent)
        self.assertEqual(ms.wheel_ticks, [])

    def test_zoom_without_adapters_never_sends(self):
        from actions.dispatcher import ActionDispatcher
        d = ActionDispatcher(_cfg(enabled=True, dry_run=False),
                             FakeKeyboard(), mouse=None)
        rec = d.dispatch("ZOOM_IN_STEP", 1.0)
        self.assertFalse(rec.sent)


class TestScrollDispatch(unittest.TestCase):
    def test_scroll_up_maps_plain_wheel(self):
        d, _, _ = _dispatcher()
        rec = d.dispatch("SCROLL_UP_STEP", 1.0)
        self.assertEqual(rec.action, "SCROLL_UP")
        self.assertEqual(rec.key, "WHEEL+")

    def test_scroll_down_maps_plain_wheel(self):
        d, _, _ = _dispatcher()
        rec = d.dispatch("SCROLL_DOWN_STEP", 1.0)
        self.assertEqual(rec.action, "SCROLL_DOWN")
        self.assertEqual(rec.key, "WHEEL-")

    def test_live_scroll_sends_wheel_without_ctrl(self):
        d, kb, ms = _dispatcher(enabled=True, dry_run=False)
        rec = d.dispatch("SCROLL_UP_STEP", 1.0)
        self.assertTrue(rec.sent)
        self.assertEqual(ms.wheel_ticks, [1])
        self.assertEqual(kb.events, [])  # no Ctrl touched
        rec = d.dispatch("SCROLL_DOWN_STEP", 2.0)
        self.assertTrue(rec.sent)
        self.assertEqual(ms.wheel_ticks, [1, -1])

    def test_dry_run_scroll_never_sends(self):
        d, kb, ms = _dispatcher(enabled=True, dry_run=True)
        rec = d.dispatch("SCROLL_UP_STEP", 1.0)
        self.assertEqual(rec.mode, "DRY_RUN")
        self.assertFalse(rec.sent)
        self.assertEqual(ms.wheel_ticks, [])

    def test_disabled_scroll_never_sends(self):
        d, _, ms = _dispatcher(enabled=False, dry_run=False)
        rec = d.dispatch("SCROLL_DOWN_STEP", 1.0)
        self.assertEqual(rec.mode, "OBSERVED")
        self.assertFalse(rec.sent)
        self.assertEqual(ms.wheel_ticks, [])

    def test_scroll_without_mouse_never_sends(self):
        from actions.dispatcher import ActionDispatcher
        d = ActionDispatcher(_cfg(enabled=True, dry_run=False),
                             FakeKeyboard(), mouse=None)
        rec = d.dispatch("SCROLL_UP_STEP", 1.0)
        self.assertFalse(rec.sent)

    def test_end_to_end_peace_to_wheel_live(self):
        """PeaceScroll tracker -> dispatcher -> one plain wheel tick."""
        from vision.peace_scroll import PeaceScrollTracker
        sc = PeaceScrollTracker({"confirm_frames": 4, "repeat_ms": 200})
        d, kb, ms = _dispatcher(enabled=True, dry_run=False)
        events = []
        t = 0.0
        for direction in (1, 1, 1, 1):
            for ev in sc.update(direction, t):
                rec = d.dispatch(ev, t)
                events.append((ev, rec.action if rec else None))
            t += 1 / 30.0
        self.assertEqual(events, [("SCROLL_UP_STEP", "SCROLL_UP")])
        self.assertEqual(ms.wheel_ticks, [1])
        self.assertEqual(kb.events, [])

    def test_end_to_end_twist_to_wheel_live(self):
        """YoZoom tracker -> dispatcher -> one Ctrl+wheel tick."""
        from vision.yo_zoom import YoZoomTracker
        zc = YoZoomTracker({"step_degrees": 30.0, "confirm_frames": 4,
                            "hysteresis_degrees": 3.0,
                            "tick_spacing_ms": 250}, mirror_sign=1)
        d, kb, ms = _dispatcher(enabled=True, dry_run=False)
        events = []
        t = 0.0
        for angle in (0.0, 30.0, 30.0, 30.0, 30.0):
            for ev in zc.update(angle, t):
                rec = d.dispatch(ev, t)
                events.append((ev, rec.action if rec else None))
            t += 1 / 30.0
        self.assertEqual(events, [("ZOOM_IN_STEP", "ZOOM_IN")])
        self.assertEqual(ms.wheel_ticks, [1])
        self.assertEqual(kb.held, frozenset())


class TestDuplicateProtection(unittest.TestCase):
    def test_same_event_inside_window_dropped(self):
        d, kb, _ = _dispatcher(enabled=True, dry_run=False)
        self.assertIsNotNone(d.dispatch("CLOCKWISE_ROTATION_CONFIRMED", 1.0))
        self.assertIsNone(d.dispatch("CLOCKWISE_ROTATION_CONFIRMED", 1.02))
        self.assertEqual(kb.presses, ["LEFT"])

    def test_same_event_after_window_accepted(self):
        d, kb, _ = _dispatcher(enabled=True, dry_run=False)
        d.dispatch("CLOCKWISE_ROTATION_CONFIRMED", 1.0)
        rec = d.dispatch("CLOCKWISE_ROTATION_CONFIRMED", 1.5)
        self.assertIsNotNone(rec)
        self.assertEqual(kb.presses, ["LEFT", "LEFT"])

    def test_different_event_not_blocked(self):
        d, _, _ = _dispatcher()
        d.dispatch("CLOCKWISE_ROTATION_CONFIRMED", 1.0)
        rec = d.dispatch("ANTICLOCKWISE_ROTATION_CONFIRMED", 1.05)
        self.assertIsNotNone(rec)


class TestUnknownEvent(unittest.TestCase):
    def test_unknown_event_safe(self):
        d, kb, ms = _dispatcher(enabled=True, dry_run=False)
        self.assertIsNone(d.dispatch("WAVE_HELLO", 1.0))
        self.assertEqual(kb.presses, [])
        self.assertEqual(ms.clicks, 0)
        self.assertEqual(d.last_action, "NONE")

    def test_old_events_unmapped(self):
        d, kb, ms = _dispatcher(enabled=True, dry_run=False)
        for ev in ("SWIPE_LEFT_CONFIRMED", "SWIPE_RIGHT_CONFIRMED",
                   "FIST_STARTED", "FIST_RELEASED", "FIST_LOCKED",
                   "THUMBS_UP_UNLOCK"):
            self.assertIsNone(d.dispatch(ev, 1.0))
        self.assertEqual(kb.presses, [])
        self.assertEqual(ms.clicks, 0)
        self.assertEqual(ms.downs, 0)


class TestKeyboardSafety(unittest.TestCase):
    def test_press_pairs_down_up(self):
        from actions.keyboard import KeyboardAdapter
        calls = []
        kb = KeyboardAdapter(sender=lambda vk, flags: calls.append((vk, flags)))
        kb.press_key("RIGHT")
        self.assertEqual(calls, [(0x27, 0), (0x27, 0x0002)])
        self.assertEqual(kb.held, frozenset())

    def test_close_releases_held(self):
        from actions.keyboard import KeyboardAdapter
        calls = []
        kb = KeyboardAdapter(sender=lambda vk, flags: calls.append((vk, flags)))
        kb.key_down("LEFT")
        kb.close()
        self.assertEqual(kb.held, frozenset())
        ups = [c for c in calls if c[1] == 0x0002]
        self.assertTrue(ups)

    def test_unknown_key_rejected(self):
        from actions.keyboard import KeyboardAdapter
        kb = KeyboardAdapter(sender=lambda vk, flags: None)
        with self.assertRaises(ValueError):
            kb.press_key("F13")

    def test_arrow_vk_codes(self):
        from actions.keyboard import VK
        self.assertEqual(VK["RIGHT"], 0x27)
        self.assertEqual(VK["LEFT"], 0x25)

    def test_ctrl_vk_code_for_zoom_chord(self):
        from actions.keyboard import VK, KeyboardAdapter
        self.assertEqual(VK["CTRL"], 0x11)
        calls = []
        kb = KeyboardAdapter(sender=lambda vk, flags: calls.append((vk, flags)))
        kb.key_down("CTRL")
        self.assertEqual(kb.held, frozenset({"CTRL"}))
        kb.key_up("CTRL")
        self.assertEqual(kb.held, frozenset())
        self.assertEqual(calls, [(0x11, 0), (0x11, 0x0002)])

    def test_input_struct_matches_win32_x64(self):
        import ctypes
        import struct
        from actions import keyboard as K
        if struct.calcsize("P") != 8:
            self.skipTest("x64 layout check needs 64-bit Python")
        self.assertEqual(ctypes.sizeof(K._INPUT), 40)  # cbSize SendInput wants
        self.assertEqual(K._INPUT.ki.offset, 8)  # union starts after DWORD+pad
        self.assertEqual(K._KEYBDINPUT.dwExtraInfo.offset, 16)  # ULONG_PTR slot

    def test_native_failure_raises_informative_error(self):
        from unittest import mock
        from actions import keyboard as K
        with mock.patch.object(K._user32, "SendInput", return_value=0):
            with self.assertRaises(RuntimeError) as cm:
                K._send(0x27, 0)
        msg = str(cm.exception)
        self.assertIn("0x27", msg)
        self.assertIn("cbSize=40", msg)
        self.assertIn("win32 err", msg)

    def test_dry_run_never_touches_native_sendinput(self):
        from unittest import mock
        from actions import keyboard as K
        from actions.dispatcher import ActionDispatcher
        from actions.keyboard import KeyboardAdapter
        with mock.patch.object(K._user32, "SendInput",
                               side_effect=AssertionError("must not send")):
            d = ActionDispatcher(_cfg(enabled=True, dry_run=True),
                                 KeyboardAdapter())
            rec = d.dispatch("CLOCKWISE_ROTATION_CONFIRMED", 1.0)
            self.assertEqual(rec.mode, "DRY_RUN")
            self.assertFalse(rec.sent)

    def test_disabled_never_touches_native_sendinput(self):
        from unittest import mock
        from actions import keyboard as K
        from actions.dispatcher import ActionDispatcher
        from actions.keyboard import KeyboardAdapter
        with mock.patch.object(K._user32, "SendInput",
                               side_effect=AssertionError("must not send")):
            d = ActionDispatcher(_cfg(enabled=False, dry_run=False),
                                 KeyboardAdapter())
            rec = d.dispatch("CLOCKWISE_ROTATION_CONFIRMED", 1.0)
            self.assertEqual(rec.mode, "OBSERVED")
            self.assertFalse(rec.sent)

    def test_end_to_end_rotation_to_action_live(self):
        """Full path: rotation step -> dispatcher -> exactly 1 press.

        Uses the default mirror sign: physical clockwise (+30) appears as
        -30 in mirrored landmarks and must still navigate PREVIOUS.
        """
        from vision.rotation import RotationTracker
        rot = RotationTracker({"step_degrees": 30.0, "confirm_frames": 4,
                               "hysteresis_degrees": 3.0,
                               "step_action_spacing_ms": 120,
                               "open_required": True})
        d, kb, _ = _dispatcher(enabled=True, dry_run=False)
        events = []
        t = 0.0
        for a in (0.0, -30.0, -30.0, -30.0, -30.0):
            for ev in rot.update(a, t):
                rec = d.dispatch(ev, t)
                events.append((ev, rec.action if rec else None))
            t += 1 / 30.0
        self.assertEqual(events,
                         [("CLOCKWISE_ROTATION_CONFIRMED", "PREVIOUS_SLIDE")])
        self.assertEqual(kb.presses, ["LEFT"])

    def test_end_to_end_pinch_to_mouse_live(self):
        """Pinch tracker -> dispatcher -> one DOWN then one UP."""
        from vision.pinch import PinchTracker
        import numpy as np
        cfg = {"start_threshold": 0.30, "release_threshold": 0.38,
               "confirm_frames": 3, "click_cooldown_ms": 600}
        pinch = PinchTracker(cfg)
        d, _, ms = _dispatcher(enabled=True, dry_run=False)
        lm = np.zeros((21, 2))
        lm[0] = [0.5, 0.8]
        lm[9] = [0.5, 0.6]
        scale = float(np.linalg.norm(lm[0] - lm[9]))
        lm[8] = [0.5, 0.4]
        close = lm.copy()
        close[4] = lm[8] + np.array([0.15 * scale, 0.0])
        far = lm.copy()
        far[4] = lm[8] + np.array([0.8 * scale, 0.0])
        t = 0.0
        for _ in range(3):
            for ev in pinch.update(close, t):
                d.dispatch(ev, t)
            t += 1 / 30.0
        self.assertEqual(ms.downs, 1)
        self.assertTrue(ms.button_held)
        for ev in pinch.update(far, t):
            d.dispatch(ev, t)
        self.assertEqual(ms.ups, 1)
        self.assertFalse(ms.button_held)


if __name__ == "__main__":
    unittest.main()
