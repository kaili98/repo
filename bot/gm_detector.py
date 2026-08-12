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
    return os.path.join(base, filename)

class GMDetector:
    """Scans the full game window for a nameplate template (e.g. a GM name) via template matching."""

    def __init__(self, wm: WindowManager, template_path: str = "kin.png", threshold: float = 0.85):
        self.wm = wm
        self.threshold = threshold
        self.template = None

        path = _resource_path(template_path)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self.template = img

    def check(self) -> bool:
        """Capture the full game client area and return True if the template is found."""
        if self.template is None:
            return False

        rect = self.wm.get_client_rect_screen()
        if rect is None:
            return False

        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        if w <= 0 or h <= 0:
            return False

        frame = self.wm.capture_region(0, 0, w, h)
        if frame is None:
            return False

        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        th, tw = self.template.shape[:2]
        if frame_bgr.shape[0] < th or frame_bgr.shape[1] < tw:
            return False

        result = cv2.matchTemplate(frame_bgr, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return bool(max_val >= self.threshold)
