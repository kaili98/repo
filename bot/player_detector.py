import os
import sys
import numpy as np
import cv2
from bot.window_manager import WindowManager

def _resource_path(filename: str) -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "images", filename)

class PlayerDetector:
    """Scans the same minimap region used for character detection for another
    player's marker (red.png) via template matching."""

    def __init__(self, wm: WindowManager, template_path: str = "red.png", threshold: float = 0.97):
        self.wm = wm
        self.threshold = threshold
        self.template = None
        self.region_x = 0
        self.region_y = 0
        self.region_w = 0
        self.region_h = 0

        path = _resource_path(template_path)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self.template = img

    def set_region(self, x: int, y: int, w: int, h: int):
        self.region_x = x
        self.region_y = y
        self.region_w = w
        self.region_h = h

    def check(self) -> bool:
        """Capture the char/minimap region and return True if red.png is found in it."""
        if self.template is None:
            return False

        frame = self.wm.capture_region(self.region_x, self.region_y, self.region_w, self.region_h)
        if frame is None:
            return False

        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        th, tw = self.template.shape[:2]
        if frame_bgr.shape[0] < th or frame_bgr.shape[1] < tw:
            return False

        result = cv2.matchTemplate(frame_bgr, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return bool(max_val >= self.threshold)
