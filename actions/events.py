"""Logical actions. The dispatcher maps confirmed gesture events to these.

Rotation semantics (physical hand motion, mirror-compensated once in
vision.palm / vision.rotation):
  physical palm rotation CLOCKWISE -> PREVIOUS slide
  physical palm rotation ANTI-CLOCKWISE -> NEXT slide

Left click is pinch-only: PINCH_STARTED / PINCH_RELEASED. Fist and
rotation never map to LEFT_DOWN / LEFT_UP.

Zoom: ZOOM_IN_STEP / ZOOM_OUT_STEP from vision.yo_zoom committed-angle
steps while the YO sign holds -> atomic Ctrl+wheel ticks.

Scroll: SCROLL_UP_STEP / SCROLL_DOWN_STEP from vision.peace_scroll
hold-to-repeat while the peace sign points right/left -> plain wheel
ticks (no Ctrl) on the focused window.
"""
NEXT_SLIDE = "NEXT_SLIDE"
PREVIOUS_SLIDE = "PREVIOUS_SLIDE"
LEFT_DOWN = "LEFT_DOWN"  # pinch confirmed: press and hold via MouseAdapter
LEFT_UP = "LEFT_UP"  # pinch released / hand lost: release the held button
ZOOM_IN = "ZOOM_IN"  # one Ctrl+wheel-forward tick via adapters
ZOOM_OUT = "ZOOM_OUT"  # one Ctrl+wheel-back tick via adapters
SCROLL_UP = "SCROLL_UP"  # one plain wheel-forward tick via MouseAdapter
SCROLL_DOWN = "SCROLL_DOWN"  # one plain wheel-back tick via MouseAdapter

# Gesture event -> logical action. Overridden by config["action"]["mapping"].
DEFAULT_MAPPING = {
    "CLOCKWISE_ROTATION_CONFIRMED": PREVIOUS_SLIDE,
    "ANTICLOCKWISE_ROTATION_CONFIRMED": NEXT_SLIDE,
    "PINCH_STARTED": LEFT_DOWN,
    "PINCH_RELEASED": LEFT_UP,
    "ZOOM_IN_STEP": ZOOM_IN,
    "ZOOM_OUT_STEP": ZOOM_OUT,
    "SCROLL_UP_STEP": SCROLL_UP,
    "SCROLL_DOWN_STEP": SCROLL_DOWN,
}
