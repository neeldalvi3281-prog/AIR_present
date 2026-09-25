"""Reform config loader. Local JSON only, no network. Tunables live here, not in code.

Safety: [action] enabled/dry_run is the master gate for ALL real input
(mouse moves and clicks included). Subsystem flags ([mouse]/[rotation]/
[pinch] enabled) only toggle detection paths. Safe defaults: everything
detects, nothing sends (enabled=false, dry_run=true).
"""
import json
import os

DEFAULTS = {
    "camera": {"device": 0, "width": 640, "height": 480, "target_fps": 30},
    "tracking": {
        "max_hands": 1,
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.6,
    },
    "model": {
        "path": "models/hand_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    },
    "benchmark": {
        "resolutions": [[320, 240], [480, 270], [640, 360], [640, 480]],
        "seconds_per_resolution": 10,
        "warmup_seconds": 2,
        "output": "logs/benchmark_latest.json",
    },
    "mouse": {
        "enabled": True,
        "smoothing_enabled": True,
        "min_cutoff": 1.2,
        "beta": 0.02,
        "d_cutoff": 1.0,
        "mapping": {
            "left": 0.07,
            "right": 0.95,
            "top": 0.08,
            "bottom": 0.92,
        },
    },
    "rotation": {
        "enabled": True,
        "step_degrees": 30.0,
        "confirm_frames": 4,
        "hysteresis_degrees": 3.0,
        "step_action_spacing_ms": 120,
        "open_required": True,
    },
    "fist": {
        "enabled": True,
        "fold_ratio": 1.15,
        "fist_folded_min": 4,
    },
    "pinch": {
        "enabled": True,
        "start_threshold": 0.30,
        "release_threshold": 0.38,
        "confirm_frames": 3,
        "click_cooldown_ms": 600,
    },
    "zoom": {
        "enabled": True,
        "step_degrees": 30.0,
        "confirm_frames": 4,
        "hysteresis_degrees": 3.0,
        "tick_spacing_ms": 250,
    },
    "scroll": {
        "enabled": True,
        "confirm_frames": 4,
        "repeat_ms": 200,
    },
    "mode": "presentation",
    "action": {
        "enabled": False,
        "dry_run": True,
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
        "keys": {
            "NEXT_SLIDE": "RIGHT",
            "PREVIOUS_SLIDE": "LEFT",
        },
    },
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config(path=CONFIG_PATH):
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy of defaults
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        _deep_merge(cfg, user)
    else:
        raise FileNotFoundError(f"Config not found: {path}")
    _validate(cfg)
    return cfg


def _deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _validate(cfg):
    cam = cfg["camera"]
    if cam["width"] <= 0 or cam["height"] <= 0:
        raise ValueError("camera width/height must be positive")
    tr = cfg["tracking"]
    if tr["max_hands"] != 1:
        raise ValueError("Phase 1 supports exactly 1 hand (PRD one-hand policy)")
    for key in ("min_detection_confidence", "min_tracking_confidence"):
        if not 0.0 < tr[key] <= 1.0:
            raise ValueError(f"tracking.{key} must be in (0, 1]")
    mo = cfg["mouse"]
    if not isinstance(mo["enabled"], bool):
        raise ValueError("mouse.enabled must be a boolean")
    if not isinstance(mo["smoothing_enabled"], bool):
        raise ValueError("mouse.smoothing_enabled must be a boolean")
    for key in ("min_cutoff", "beta", "d_cutoff"):
        if not mo[key] > 0:
            raise ValueError(f"mouse.{key} must be positive")
    mp = mo["mapping"]
    for lo_key, hi_key in (("left", "right"), ("top", "bottom")):
        lo, hi = mp[lo_key], mp[hi_key]
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"mouse.mapping requires 0 <= {lo_key} < "
                             f"{hi_key} <= 1")
    ro = cfg["rotation"]
    if not isinstance(ro["enabled"], bool):
        raise ValueError("rotation.enabled must be a boolean")
    if not 0.0 < ro["step_degrees"] <= 90.0:
        raise ValueError("rotation.step_degrees must be in (0, 90]")
    if not ro["confirm_frames"] >= 1:
        raise ValueError("rotation.confirm_frames must be >= 1")
    if not 0.0 <= ro["hysteresis_degrees"] < ro["step_degrees"]:
        raise ValueError("rotation requires 0 <= hysteresis < step_degrees")
    if not ro["step_action_spacing_ms"] >= 0:
        raise ValueError("rotation.step_action_spacing_ms must be non-negative")
    if not isinstance(ro["open_required"], bool):
        raise ValueError("rotation.open_required must be a boolean")
    fi = cfg["fist"]
    if not isinstance(fi["enabled"], bool):
        raise ValueError("fist.enabled must be a boolean")
    if not fi["fold_ratio"] > 1.0:
        raise ValueError("fist.fold_ratio must exceed 1.0")
    if fi["fist_folded_min"] not in (3, 4):
        raise ValueError("fist_folded_min must be 3 or 4")
    pi = cfg["pinch"]
    if not isinstance(pi["enabled"], bool):
        raise ValueError("pinch.enabled must be a boolean")
    if not 0.0 < pi["start_threshold"] < pi["release_threshold"]:
        raise ValueError("pinch requires 0 < start_threshold < release_threshold")
    if not pi["confirm_frames"] >= 1:
        raise ValueError("pinch.confirm_frames must be >= 1")
    if not pi["click_cooldown_ms"] >= 0:
        raise ValueError("pinch.click_cooldown_ms must be non-negative")
    zo = cfg["zoom"]
    if not isinstance(zo["enabled"], bool):
        raise ValueError("zoom.enabled must be a boolean")
    if not 0.0 < zo["step_degrees"] <= 90.0:
        raise ValueError("zoom.step_degrees must be in (0, 90]")
    if not zo["confirm_frames"] >= 1:
        raise ValueError("zoom.confirm_frames must be >= 1")
    if not 0.0 <= zo["hysteresis_degrees"] < zo["step_degrees"]:
        raise ValueError("zoom requires 0 <= hysteresis < step_degrees")
    if not zo["tick_spacing_ms"] >= 0:
        raise ValueError("zoom.tick_spacing_ms must be non-negative")
    sc = cfg["scroll"]
    if not isinstance(sc["enabled"], bool):
        raise ValueError("scroll.enabled must be a boolean")
    if not sc["confirm_frames"] >= 1:
        raise ValueError("scroll.confirm_frames must be >= 1")
    if not sc["repeat_ms"] >= 0:
        raise ValueError("scroll.repeat_ms must be non-negative")
    a = cfg["action"]
    if not isinstance(a["enabled"], bool) or not isinstance(a["dry_run"], bool):
        raise ValueError("action.enabled/dry_run must be booleans")
    if not a["duplicate_window_ms"] >= 0:
        raise ValueError("action.duplicate_window_ms must be non-negative")
    for act in a["mapping"].values():
        if (act not in a["keys"]
                and act not in ("LEFT_DOWN", "LEFT_UP", "ZOOM_IN", "ZOOM_OUT",
                                "SCROLL_UP", "SCROLL_DOWN")):
            raise ValueError(f"action mapping target {act!r} has no entry in action.keys")
