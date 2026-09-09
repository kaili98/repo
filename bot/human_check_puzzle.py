import os
import sys
from typing import List, Optional, Tuple
import numpy as np
import cv2

def _resource_path(filename: str) -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, filename)

# Offsets (dx, dy) from the anchor's matched top-left to each of the 5 icon
# centers, plus the icon crop half-size - all calibrated from a real full-window
# capture of the "HUMAN CHECK - Click the ONE picture that is different" dialog
# (visual-puzzle.jpg). The anchor itself is a crop of the instruction line,
# which is identical across both "1 of 2" and "2 of 2" steps.
ICON_OFFSETS: List[Tuple[int, int]] = [(72, 68), (132, 68), (192, 68), (252, 68), (312, 68)]
ICON_HALF_SIZE = 22


class HumanCheckPuzzleSolver:
    """Locates the Human Check picture-difference dialog via a stable anchor
    crop of its instruction text, then compares the 5 icon slots (fixed offsets
    from that anchor) to find which one doesn't match the other 4."""

    def __init__(self, anchor_path: str = "human-check-puzzle-anchor.png", threshold: float = 0.85):
        self.threshold = threshold
        self.anchor = None

        path = _resource_path(anchor_path)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self.anchor = img

    def locate(self, frame: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
        """Return the anchor's (x, y) top-left position in `frame`, or None if
        the dialog isn't present."""
        if self.anchor is None or frame is None:
            return None

        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        ah, aw = self.anchor.shape[:2]
        if frame_bgr.shape[0] < ah or frame_bgr.shape[1] < aw:
            return None

        result = cv2.matchTemplate(frame_bgr, self.anchor, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < self.threshold:
            return None
        return max_loc

    def find_odd_icon(self, frame: np.ndarray, anchor_pos: Tuple[int, int]) -> Optional[int]:
        """Return the 0-based index (0-4) of the icon that looks different from
        the other 4, or None if the icon slots couldn't be cropped (e.g. dialog
        partially off-screen)."""
        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        ax, ay = anchor_pos
        h, w = frame_bgr.shape[:2]

        icons = []
        for dx, dy in ICON_OFFSETS:
            cx, cy = ax + dx, ay + dy
            x0, y0 = cx - ICON_HALF_SIZE, cy - ICON_HALF_SIZE
            x1, y1 = cx + ICON_HALF_SIZE, cy + ICON_HALF_SIZE
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                return None
            icons.append(frame_bgr[y0:y1, x0:x1])

        n = len(icons)
        avg_similarity = []
        for i in range(n):
            sims = []
            for j in range(n):
                if i == j:
                    continue
                result = cv2.matchTemplate(icons[i], icons[j], cv2.TM_CCOEFF_NORMED)
                sims.append(float(result[0, 0]))
            avg_similarity.append(sum(sims) / len(sims))

        # The odd one out looks different from all 4 others, so it has the
        # lowest average similarity - the other 4 (similar to each other) all
        # score high against one another.
        return int(np.argmin(avg_similarity))
