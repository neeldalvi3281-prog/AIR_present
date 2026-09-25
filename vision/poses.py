"""Minimal static hand-pose classification (reform §25-§26).

Active classes: OPEN / FIST / YO / PEACE / UNKNOWN. No thumbs-up
or OK classifiers — they cost CPU and add failure modes for
gestures the reform does not use.

Geometry is pure landmark math in normalized [0,1] space (resolution
independent). A finger counts as extended when its tip is clearly farther
from the wrist than its PIP joint::

    |tip - wrist| / |pip - wrist| > fold_ratio

YO ("rock on"): index + pinky extended, middle + ring folded, thumb
folded. Holding YO and rotating the hand drives presentation zoom
(see vision.yo_zoom); anything else resets it.

PEACE: index + middle extended, ring + pinky folded, thumb folded.
Pointing the peace sign right scrolls up, left scrolls down (direction
from peace_direction, not part of the label; see vision.peace_scroll).

Left click is NOT this module: see vision.pinch.PinchTracker.
Fist/UNKNOWN/YO/PEACE gate palm-rotation navigation (rotation_frame).
"""
from vision.features import FrameFeatures

# Landmark ids.
WRIST = 0
TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIPS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCPS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
FOUR_FINGERS = ("index", "middle", "ring", "pinky")

OPEN = "OPEN"
FIST = "FIST"
YO = "YO"
PEACE = "PEACE"
UNKNOWN = "UNKNOWN"

# Both peace tips must sit this far past their MCPs (× hand scale, same
# side) to count as pointing — kills ambiguous/diagonal flicker.
PEACE_REACH = 0.3


def _feats(lm):
    """Raw landmarks -> shared bundle; bundles pass through untouched."""
    return lm if isinstance(lm, FrameFeatures) else FrameFeatures.from_landmarks(lm)


def hand_scale_norm(lm):
    """Wrist->middle-MCP distance in normalized units (never ~0)."""
    return _feats(lm).hand_scale


def finger_extended(lm, finger, fold_ratio):
    """True when tip is clearly farther from wrist than the PIP joint."""
    return _feats(lm).ext(finger, fold_ratio)


def finger_states(lm, fold_ratio):
    """Dict finger -> bool (extended?). Thumb uses tip-vs-IP ratio too."""
    f = _feats(lm)
    return {finger: f.ext(finger, fold_ratio) for finger in
            ("thumb",) + FOUR_FINGERS}


def thumb_extended(lm, cfg):
    """Thumb out (vs folded across palm), scale-normalized."""
    return _feats(lm).thumb_out


def is_fist(lm, cfg):
    states = finger_states(lm, cfg["fold_ratio"])
    folded = sum(1 for f in FOUR_FINGERS if not states[f])
    # A real fist folds the thumb too; an extended thumb is never a fist.
    return folded >= cfg["fist_folded_min"] and not thumb_extended(lm, cfg)


def is_open(lm, cfg):
    states = finger_states(lm, cfg["fold_ratio"])
    return all(states.values())


def is_yo(lm, cfg):
    """Rock-on: index + pinky out, middle + ring folded, thumb folded."""
    states = finger_states(lm, cfg["fold_ratio"])
    if not (states["index"] and states["pinky"]):
        return False
    if states["middle"] or states["ring"]:
        return False
    if thumb_extended(lm, cfg):
        return False
    return True


def is_peace(lm, cfg):
    """Peace: index + middle out, ring + pinky folded, thumb folded."""
    states = finger_states(lm, cfg["fold_ratio"])
    if not (states["index"] and states["middle"]):
        return False
    if states["ring"] or states["pinky"]:
        return False
    if thumb_extended(lm, cfg):
        return False
    return True


def peace_direction(lm):
    """Pointing of a peace hand: +1 right (scroll up), -1 left (down).

    Both index and middle tips must clear PEACE_REACH past their MCPs
    on the same side; anything else (split, short, vertical) is 0.
    """
    dx_i, dx_m = _feats(lm).peace_dx
    if dx_i >= PEACE_REACH and dx_m >= PEACE_REACH:
        return 1
    if dx_i <= -PEACE_REACH and dx_m <= -PEACE_REACH:
        return -1
    return 0


def classify_pose(lm, cfg):
    """Single winning label: FIST > YO > PEACE > OPEN, else UNKNOWN."""
    f = _feats(lm)
    if is_fist(f, cfg):
        return FIST
    if is_yo(f, cfg):
        return YO
    if is_peace(f, cfg):
        return PEACE
    if is_open(f, cfg):
        return OPEN
    return UNKNOWN
