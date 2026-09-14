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
    return os.path.join(base, "images", filename)

# Offsets (dx, dy) from the anchor's matched top-left, calibrated from a real
# full-window capture of the "Click the picture that matches this one"
# dialog (pick-the-picture.png). The anchor is a crop of the constant
# instruction text "Click the picture that matches this one:" (stopping
# before the reference icon, which varies). REFERENCE_OFFSET locates that
# reference icon; OPTION_OFFSETS locate the 4 candidate icons below.
REFERENCE_OFFSET: Tuple[int, int] = (234, -10)
OPTION_OFFSETS: List[Tuple[int, int]] = [(35, 69), (110, 69), (180, 70), (245, 70)]
TEMPLATE_HALF = 18
SEARCH_HALF = 24

# Where to click first to skip/complete the instruction text's typewriter
# animation before analyzing the icons. Same technique proven necessary for
# the "pick the odd" dialog: real back-to-back "2 of 2" captures
# (pick-the-picture-failed6/7/8.jpg) showed the second line still mid-render
# ("All pi", "All pict"), and this dialog's fast-follow second question gives
# even less settle time than the first. Pushed further right (was x=50) so
# the click/cursor icon lands past more of the instruction text instead of
# sitting directly on top of a word; still well clear of REFERENCE_OFFSET's
# x=234 (a different row anyway - REFERENCE_OFFSET's dy is negative).
TEXT_CLICK_OFFSET: Tuple[int, int] = (90, 7)


class PickPicturePuzzleSolver:
    """Locates the "click the picture that matches this one" dialog via a
    stable anchor crop of its instruction text, then compares the reference
    icon (shown inline in the instructions) against the 4 candidate icons to
    find which one matches."""

    def __init__(self, anchor_path: str = "pick-picture-anchor.png", threshold: float = 0.85):
        self.threshold = threshold
        self.anchor = None

        path = _resource_path(anchor_path)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self.anchor = img

    def locate(self, frame: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
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

    def find_matching_option(self, frame: np.ndarray, anchor_pos: Tuple[int, int]) -> Optional[int]:
        """Return the 0-based index (0-3) of the candidate that matches the
        reference icon, or None if the icons couldn't be cropped."""
        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        ax, ay = anchor_pos
        h, w = frame_bgr.shape[:2]

        rx, ry = ax + REFERENCE_OFFSET[0], ay + REFERENCE_OFFSET[1]
        rx0, ry0 = rx - TEMPLATE_HALF, ry - TEMPLATE_HALF
        rx1, ry1 = rx + TEMPLATE_HALF, ry + TEMPLATE_HALF
        if rx0 < 0 or ry0 < 0 or rx1 > w or ry1 > h:
            return None
        reference = frame_bgr[ry0:ry1, rx0:rx1]

        best_index = None
        best_score = -1.0
        for i, (dx, dy) in enumerate(OPTION_OFFSETS):
            cx, cy = ax + dx, ay + dy
            sx0, sy0 = cx - SEARCH_HALF, cy - SEARCH_HALF
            sx1, sy1 = cx + SEARCH_HALF, cy + SEARCH_HALF
            if sx0 < 0 or sy0 < 0 or sx1 > w or sy1 > h:
                continue
            search = frame_bgr[sy0:sy1, sx0:sx1]
            result = cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_index = i

        return best_index
