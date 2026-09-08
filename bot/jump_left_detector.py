import os
import sys
import numpy as np
import cv2

def _resource_path(filename: str) -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, filename)

class JumpLeftDetector:
    """Matches the "Step 1 of 2: JUMP ONCE / Step 2 of 2: MOVE LEFT" instruction
    banner (JUMP-AND-LEFT.png) against an already-captured full-window frame -
    identifies the specific Lie Detector variant that can be auto-solved, without
    needing its own screen capture (the caller reuses GMDetector's last frame)."""

    def __init__(self, template_path: str = "JUMP-AND-LEFT.png", threshold: float = 0.85):
        self.threshold = threshold
        self.template = None

        path = _resource_path(template_path)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self.template = img

    def check_frame(self, frame: np.ndarray) -> bool:
        if self.template is None or frame is None:
            return False

        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        th, tw = self.template.shape[:2]
        if frame_bgr.shape[0] < th or frame_bgr.shape[1] < tw:
            return False

        result = cv2.matchTemplate(frame_bgr, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return bool(max_val >= self.threshold)
