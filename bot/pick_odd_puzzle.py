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
# full-window capture ("pick the odd.jpg", 4 icons shown). The anchor is a
# crop of the constant instruction line "Click the ONE picture that is
# different." (the second line, "All pictures stay visible.", isn't used
# since it can still be mid-typewriter-animation when first detected - seen
# cut off as "All pict" in the calibration capture). Only 4 slots - this
# dialog family (pick-the-picture, arithmetic) consistently shows 4 options,
# and guessing a 5th position risks comparing against background past the
# last real icon (confirmed: this got picked as "the odd one out").
ICON_OFFSETS: List[Tuple[int, int]] = [(33, 67), (97, 63), (167, 63), (237, 63)]
TEMPLATE_HALF = 16
SEARCH_HALF = 22

# Click point offsets - a few px lower than ICON_OFFSETS (closer to the
# option number's height than the icon's visual center). Icons vary in
# height (e.g. a short bowl vs a tall bottle) but stay bottom-aligned on the
# same baseline, so a click point calibrated for a tall icon lands right on
# the top edge/rim of a shorter icon and can miss its hitbox entirely
# (confirmed: "pick the picture-failed5.jpg" - a short bowl icon - clicked
# just above it). ICON_OFFSETS is left untouched since it's also used to
# center the comparison crops, which are already working correctly.
CLICK_OFFSETS: List[Tuple[int, int]] = [(dx, dy + 10) for dx, dy in ICON_OFFSETS]

# Where to click first to skip/complete the instruction text's typewriter
# animation before analyzing the icons - confirmed necessary from a real
# capture where the second line was still mid-render ("All pict..."). Pushed
# further right (was x=50) so the click/cursor icon lands past more of the
# instruction text instead of sitting directly on top of a word.
TEXT_CLICK_OFFSET: Tuple[int, int] = (90, 7)


class PickOddPuzzleSolver:
    """Locates the "click the ONE picture that is different" dialog via a
    stable anchor crop of its instruction text, then compares up to 5 icons
    pairwise to find which one doesn't match the rest (the same "odd one out"
    technique proven for the earlier version of this dialog)."""

    def __init__(self, anchor_path: str = "pick-odd-anchor.png", threshold: float = 0.85):
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

    def find_odd_icon(self, frame: np.ndarray, anchor_pos: Tuple[int, int]) -> Optional[int]:
        """Return the 0-based index of the icon that looks different from the
        others, or None if the icon slots couldn't be cropped.

        Uses a small local search window per comparison (not just the exact
        calibrated centers) - a real capture showed two icons of the *same*
        item scoring as low as 0.14 similarity on an exact fixed-position
        comparison, purely from a few pixels of rendering/alignment jitter,
        which would have been mistaken for a real visual difference."""
        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        ax, ay = anchor_pos
        h, w = frame_bgr.shape[:2]

        templates = []
        searches = []
        for dx, dy in ICON_OFFSETS:
            cx, cy = ax + dx, ay + dy
            tx0, ty0 = cx - TEMPLATE_HALF, cy - TEMPLATE_HALF
            tx1, ty1 = cx + TEMPLATE_HALF, cy + TEMPLATE_HALF
            sx0, sy0 = cx - SEARCH_HALF, cy - SEARCH_HALF
            sx1, sy1 = cx + SEARCH_HALF, cy + SEARCH_HALF
            if sx0 < 0 or sy0 < 0 or sx1 > w or sy1 > h:
                break  # ran past however many icons this step actually has
            templates.append(frame_bgr[ty0:ty1, tx0:tx1])
            searches.append(frame_bgr[sy0:sy1, sx0:sx1])

        n = len(templates)
        if n < 2:
            return None

        avg_similarity = []
        for i in range(n):
            sims = []
            for j in range(n):
                if i == j:
                    continue
                result = cv2.matchTemplate(searches[j], templates[i], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                sims.append(float(max_val))
            avg_similarity.append(sum(sims) / len(sims))

        return int(np.argmin(avg_similarity))
