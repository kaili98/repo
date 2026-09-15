import os
import sys
from typing import Optional, Tuple
import numpy as np
import cv2

def _resource_path(filename: str) -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "images", filename)

# Offset (dx, dy) of the first apple icon from the anchor's matched
# top-left, calibrated from a real full-window capture of the "Count the
# apples (N/2)" dialog (count-the-apple.png, 3 apples shown). The anchor is
# a crop of the constant instruction line "Count the apples below, then
# click the matching answer." (skipping the "(N/2)" header, which varies).
ICON_START_OFFSET: Tuple[int, int] = (15, 37)
ICON_SPACING = 36
TEMPLATE_HALF = 17
SEARCH_HALF = 25
ICON_MATCH_THRESHOLD = 0.7
# How many icons to look for at most - answer selection is keyboard-driven
# (down-arrow N times + enter) since it no longer needs a fixed pixel
# position for each possible answer, so raising this doesn't require
# guessing any new coordinates, just counting a bit further.
MAX_APPLES = 6

# Where to click first to skip/complete the instruction text's typewriter
# animation before counting icons - same technique proven necessary for the
# "pick the odd" and "pick the picture" dialogs (a real capture showed the
# full instruction/context not yet displayed on first detection). Pushed
# further right (was x=50) so the click/cursor icon lands past more of the
# instruction text instead of sitting directly on top of a word.
TEXT_CLICK_OFFSET: Tuple[int, int] = (90, 7)


class AppleCountPuzzleSolver:
    """Locates the "Count the apples" dialog via a stable anchor crop of its
    instruction text, then counts how many apple icons are shown (using the
    first one as a template) to determine which vertical option (1-5 apples)
    answers correctly."""

    def __init__(self, anchor_path: str = "apple-count-anchor.png", threshold: float = 0.85):
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

    def count_icons(self, frame: np.ndarray, anchor_pos: Tuple[int, int]) -> Optional[int]:
        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        ax, ay = anchor_pos
        h, w = frame_bgr.shape[:2]

        icx, icy = ax + ICON_START_OFFSET[0], ay + ICON_START_OFFSET[1]
        tx0, ty0 = icx - TEMPLATE_HALF, icy - TEMPLATE_HALF
        tx1, ty1 = icx + TEMPLATE_HALF, icy + TEMPLATE_HALF
        if tx0 < 0 or ty0 < 0 or tx1 > w or ty1 > h:
            return None
        template = frame_bgr[ty0:ty1, tx0:tx1]

        count = 0
        for i in range(MAX_APPLES):
            cx = icx + i * ICON_SPACING
            sx0, sy0 = cx - SEARCH_HALF, icy - SEARCH_HALF
            sx1, sy1 = cx + SEARCH_HALF, icy + SEARCH_HALF
            if sx0 < 0 or sy0 < 0 or sx1 > w or sy1 > h:
                break
            search = frame_bgr[sy0:sy1, sx0:sx1]
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val < ICON_MATCH_THRESHOLD:
                break  # icons fill left-to-right with no gaps - first miss ends the count
            count += 1

        return count if count > 0 else None
