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
# reference icon - it's inline with the instruction text itself, so it
# stays put regardless of how the options below are laid out.
#
# Re-measured via connected-component analysis across 6 real captures
# (both layouts): the true center is (236, -2), not the previous (234,-10).
# That 8px Y error was always there but small enough not to cause visible
# failures against the old, more forgiving horizontal-layout comparison -
# the vertical layout's tightly-packed rows (see OPTION_OFFSETS_VERTICAL)
# left much less margin for error and is what exposed it.
REFERENCE_OFFSET: Tuple[int, int] = (236, -2)

# This dialog has shown two icon layouts in real captures: 4 candidates side
# by side on one row, or stacked vertically one per row (the same shift
# arithmetic's options list and pick-the-odd's icons went through). Option
# positions are found dynamically (see _locate_icons) rather than assumed at
# a fixed pitch - pick-the-odd hit a real bug from a fixed vertical pitch
# (row spacing depends on icon height, so it drifted over multiple rows and
# corrupted comparisons); locating each icon from its actual pixel content
# sidesteps that entirely, and scales to more than 4 options for free since
# answer selection is keyboard-driven (down-arrow N times + enter) rather
# than needing a precalibrated click position per option.
#
# Search region (relative to the anchor) generous enough to contain either
# layout's full spread for up to MAX_OPTIONS candidates, starting below
# REFERENCE_OFFSET's own row so it can't ever pick up the reference icon.
OPTION_SEARCH_REGION: Tuple[int, int, int, int] = (15, 55, 420, 260)  # (x0, y0, x1, y1)
MIN_ICON_AREA = 120      # real icons run ~250-470px²; the small arrow marker /
                          # option-number text nearby runs ~15-60px² - well clear
MAX_ICON_DIMENSION = 40  # real icons run ~22-32px per side - well clear of
                          # unrelated wide/tall content (chat lines, etc.)
MAX_OPTIONS = 6

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

    @staticmethod
    def _locate_icons(frame_bgr: np.ndarray, anchor_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find where the (up to MAX_OPTIONS) candidate icons actually are,
        whichever layout is active, by locating real icon-sized blobs rather
        than assuming a fixed position/pitch for either layout (see the
        module comment). Returns [(dx, dy), ...] offsets from the anchor,
        ordered start-to-end along whichever axis they're actually spread
        across (x for the horizontal layout, y for the vertical one)."""
        ax, ay = anchor_pos
        rx0, ry0, rx1, ry1 = OPTION_SEARCH_REGION
        h, w = frame_bgr.shape[:2]
        x0, y0 = max(0, ax + rx0), max(0, ay + ry0)
        x1, y1 = min(w, ax + rx1), min(h, ay + ry1)
        if x1 <= x0 or y1 <= y0:
            return []

        region = frame_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        bg = np.median(gray[0:2, :])
        mask = (np.abs(gray.astype(int) - int(bg)) > 20).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        candidates = []
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if area < MIN_ICON_AREA or bw > MAX_ICON_DIMENSION or bh > MAX_ICON_DIMENSION:
                continue
            candidates.append((x0 + bx + bw // 2 - ax, y0 + by + bh // 2 - ay, area))
        if len(candidates) < 1:
            return []

        candidates.sort(key=lambda c: -c[2])
        candidates = candidates[:MAX_OPTIONS]

        xs = [c[0] for c in candidates]
        ys = [c[1] for c in candidates]
        if (max(xs) - min(xs)) >= (max(ys) - min(ys)):
            candidates.sort(key=lambda c: c[0])
        else:
            candidates.sort(key=lambda c: c[1])
        return [(c[0], c[1]) for c in candidates]

    def find_matching_option(self, frame: np.ndarray, anchor_pos: Tuple[int, int]) -> Optional[int]:
        """Return the 0-based index of the candidate that matches the
        reference icon, or None if the icons couldn't be found/cropped."""
        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        ax, ay = anchor_pos
        h, w = frame_bgr.shape[:2]

        rx, ry = ax + REFERENCE_OFFSET[0], ay + REFERENCE_OFFSET[1]
        rx0, ry0 = rx - TEMPLATE_HALF, ry - TEMPLATE_HALF
        rx1, ry1 = rx + TEMPLATE_HALF, ry + TEMPLATE_HALF
        if rx0 < 0 or ry0 < 0 or rx1 > w or ry1 > h:
            return None
        reference = frame_bgr[ry0:ry1, rx0:rx1]

        option_offsets = self._locate_icons(frame_bgr, anchor_pos)
        if not option_offsets:
            return None

        best_index = None
        best_score = -1.0
        for i, (dx, dy) in enumerate(option_offsets):
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
