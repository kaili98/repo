import os
import sys
from typing import Sequence
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
    """Scans the full game window for any of several nameplate templates (the
    anti-bot check icon varies between kin.png, kuro.png, viptaxi.png,
    humancheck.png, human-check-black.png, anti.png, etc) via template matching."""

    def __init__(self, wm: WindowManager,
                 template_paths: Sequence[str] = (
                     "kin.png", "kuro.png", "viptaxi.png", "humancheck.png",
                     "human-check-black.png", "anti.png",
                 ),
                 threshold: float = 0.85):
        self.wm = wm
        self.threshold = threshold
        self.templates = []
        self.last_frame = None
        self.last_match_screen_pos = None  # (x, y) absolute screen coords of the last match, top-left

        for name in template_paths:
            path = _resource_path(name)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.templates.append(img)

    def check(self) -> bool:
        """Capture the full game client area and return True if any template is found."""
        if not self.templates:
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

        self.last_frame = frame
        frame_bgr = np.ascontiguousarray(frame[:, :, :3])

        best_val = -1.0
        best_loc = None
        for template in self.templates:
            th, tw = template.shape[:2]
            if frame_bgr.shape[0] < th or frame_bgr.shape[1] < tw:
                continue
            result = cv2.matchTemplate(frame_bgr, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc

        found = best_loc is not None and best_val >= self.threshold
        if found:
            self.last_match_screen_pos = (rect[0] + best_loc[0], rect[1] + best_loc[1])
        return found
