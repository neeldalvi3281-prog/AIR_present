"""MediaPipe Hand Landmarker wrapper: 1 hand, handedness, confidence, 21 landmarks.

Note (Phase 2 §19 investigation): the runtime warning
"Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square
ROI" originates inside MediaPipe's bundled hand-landmark graph
(landmark_projection_calculator), not from our call parameters.
HandLandmarkerOptions in mediapipe 1.0.1 exposes no image-dimensions/ROI
knob (base_options, running_mode, num_hands, confidences, callback only),
so there is nothing to pass from our side. Detection quality is unaffected
in our benchmarks (tracking works, confidence ~0.97+); the warning is left
visible rather than suppressed.
"""
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core import base_options as mp_base
from mediapipe.tasks.python import vision as mp_vision


class HandTracker:
    def __init__(self, model_path, min_detection_confidence=0.6,
                 min_tracking_confidence=0.6, num_hands=1):
        if num_hands != 1:
            raise ValueError("Phase 1 supports exactly 1 hand")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {model_path}. "
                "Run: python tools/download_model.py"
            )
        base = mp_base.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.model_path = model_path
        self.mediapipe_version = mp.__version__

    def detect(self, bgr_frame, timestamp_ms):
        """Run detection. Returns None (no hand) or dict.

        Dict: landmarks_norm (21,2 float64), handedness (str),
              hand_confidence (float = handedness category score).
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        arr = np.array([[p.x, p.y] for p in lm], dtype=np.float64)
        handedness = "Unknown"
        confidence = 0.0
        if result.handedness and result.handedness[0]:
            cat = result.handedness[0][0]
            handedness = cat.category_name or "Unknown"
            confidence = float(cat.score)
        return {
            "landmarks_norm": arr,
            "handedness": handedness,
            "hand_confidence": confidence,
        }

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
