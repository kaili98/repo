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

# 4 candidate icons side by side on one row (the original/only layout seen
# until now).
OPTION_OFFSETS_HORIZONTAL: List[Tuple[int, int]] = [(35, 69), (110, 69), (180, 70), (245, 70)]

# Real captures ("pick the picture-to-test.jpg/2/3.jpg") showed a second
# layout: 4 icons stacked vertically instead, one per row - the same shift
# arithmetic's options list and pick-the-odd's icons went through. Measured
# via connected-component analysis (not eyeballed).
OPTION_OFFSETS_VERTICAL: List[Tuple[int, int]] = [(40, 72), (40, 100), (40, 129), (40, 157)]

TEMPLATE_HALF = 18
SEARCH_HALF = 24

# Click point offsets for the vertical layout - a few px below the icon's
# visual center (verified directly against the real icon shapes: lands
# solidly inside every one, including the shortest/roundest). The
# horizontal layout has never needed this adjustment (been working fine
# clicking its OPTION_OFFSETS_HORIZONTAL position directly), so that one is
# left alone rather than changing working behavior.
CLICK_OFFSETS_VERTICAL: List[Tuple[int, int]] = [(dx, dy + 8) for dx, dy in OPTION_OFFSETS_VERTICAL]

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
    def _content_score(frame_bgr: np.ndarray, anchor_pos: Tuple[int, int], offset: Tuple[int, int]) -> float:
        """How much real (non-background) content sits at this icon slot -
        the dialog background is a near-flat light gray, while an icon is
        detailed/colorful, so pixel std-dev cleanly tells slot-has-an-icon
        apart from slot-is-empty-background."""
        ax, ay = anchor_pos
        cx, cy = ax + offset[0], ay + offset[1]
        x0, y0 = cx - TEMPLATE_HALF, cy - TEMPLATE_HALF
        x1, y1 = cx + TEMPLATE_HALF, cy + TEMPLATE_HALF
        h, w = frame_bgr.shape[:2]
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return -1.0
        gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        return float(gray.std())

    def _detect_layout(self, frame_bgr: np.ndarray, anchor_pos: Tuple[int, int]):
        """This dialog has shown two different icon layouts in real captures -
        4 icons side by side on one row, or 4 stacked vertically one per row.
        Rather than guess which is active, check which candidate position for
        option 1 actually has real icon content sitting on it (option 0 sits
        close to the same spot in both layouts, so can't tell them apart)."""
        h_score = self._content_score(frame_bgr, anchor_pos, OPTION_OFFSETS_HORIZONTAL[1])
        v_score = self._content_score(frame_bgr, anchor_pos, OPTION_OFFSETS_VERTICAL[1])
        if v_score > h_score:
            return OPTION_OFFSETS_VERTICAL, CLICK_OFFSETS_VERTICAL
        return OPTION_OFFSETS_HORIZONTAL, OPTION_OFFSETS_HORIZONTAL

    def find_matching_option(self, frame: np.ndarray, anchor_pos: Tuple[int, int]) -> Optional[Tuple[int, Tuple[int, int]]]:
        """Return (0-based index, click (dx, dy) offset) for the candidate
        that matches the reference icon, or None if the icons couldn't be
        cropped."""
        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        ax, ay = anchor_pos
        h, w = frame_bgr.shape[:2]

        rx, ry = ax + REFERENCE_OFFSET[0], ay + REFERENCE_OFFSET[1]
        rx0, ry0 = rx - TEMPLATE_HALF, ry - TEMPLATE_HALF
        rx1, ry1 = rx + TEMPLATE_HALF, ry + TEMPLATE_HALF
        if rx0 < 0 or ry0 < 0 or rx1 > w or ry1 > h:
            return None
        reference = frame_bgr[ry0:ry1, rx0:rx1]

        option_offsets, click_offsets = self._detect_layout(frame_bgr, anchor_pos)

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

        if best_index is None:
            return None
        return best_index, click_offsets[best_index]
