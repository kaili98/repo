from typing import Optional
import numpy as np
from bot.window_manager import WindowManager

class CharDetector:
    def __init__(self, wm: WindowManager):
        self.wm = wm
        self.region_x = 0
        self.region_y = 30
        self.region_w = 300
        self.region_h = 120
        self.r_min = 200
        self.g_min = 200
        self.b_max = 80

    def set_region(self, x: int, y: int, w: int, h: int):
        self.region_x = x
        self.region_y = y
        self.region_w = w
        self.region_h = h

    def detect_char_x(self) -> Optional[int]:
        """Return minimap X of character (median of yellow pixels), or None."""
        img = self.wm.capture_region(self.region_x, self.region_y, self.region_w, self.region_h)
        if img is None or img.size == 0:
            return None

        b = img[:, :, 0].astype(np.int32)
        g = img[:, :, 1].astype(np.int32)
        r = img[:, :, 2].astype(np.int32)

        mask = (r > self.r_min) & (g > self.g_min) & (b < self.b_max)
        if not mask.any():
            return None

        _, xs = np.where(mask)
        return int(np.median(xs))

    def capture_preview(self) -> Optional[np.ndarray]:
        """Return the raw minimap region screenshot (BGRA)."""
        return self.wm.capture_region(self.region_x, self.region_y, self.region_w, self.region_h)

    def capture_annotated_preview(self):
        """Return (annotated_bgra, detected_x).
        Yellow pixels are highlighted red; detected X gets a green vertical line."""
        img = self.capture_preview()
        if img is None:
            return None, None

        b = img[:, :, 0].astype(np.int32)
        g = img[:, :, 1].astype(np.int32)
        r = img[:, :, 2].astype(np.int32)
        mask = (r > self.r_min) & (g > self.g_min) & (b < self.b_max)

        out = img.copy()
        out[mask, 0] = 0
        out[mask, 1] = 0
        out[mask, 2] = 255

        detected_x = None
        if mask.any():
            _, xs = np.where(mask)
            detected_x = int(np.median(xs))
            out[:, detected_x, 0] = 0
            out[:, detected_x, 1] = 255
            out[:, detected_x, 2] = 0

        return out, detected_x
