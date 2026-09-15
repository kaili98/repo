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

# This dialog has shown two icon layouts in real captures: 4 icons side by
# side on one row, or 4 stacked vertically one per row (the same shift
# arithmetic's options list went through). Icon positions are no longer
# hardcoded (see _locate_icons) - a real capture ("pick-the-picture-
# failed.png") showed the vertical layout's row pitch isn't constant, it
# depends on icon height (a row of short bowls packs ~10px tighter than a
# row with a tall bottle in it), so a fixed pitch calibrated from one
# capture drifted increasingly off over 4 rows in a capture with shorter
# icons, corrupting every comparison (even two identical bowls scored as
# low as 0.26 similarity). Dynamically finding each icon's real position
# from its actual pixel content sidesteps that entirely.
#
# Search region (relative to the anchor) generous enough to contain either
# layout's full icon spread, sized for up to MAX_ICONS options - answer
# selection is keyboard-driven (down-arrow N times + enter), so raising
# MAX_ICONS doesn't require guessing any new pixel positions the way a
# fixed-offset list would, just widening the net _locate_icons already
# scans (real icons are still told apart from unrelated content purely by
# their size, via MIN_ICON_AREA/MAX_ICON_DIMENSION below).
ICON_SEARCH_REGION: Tuple[int, int, int, int] = (10, 40, 420, 260)  # (x0, y0, x1, y1)
MIN_ICON_AREA = 120   # real icons run ~250-470px²; the small arrow marker /
                       # option-number text nearby runs ~15-60px² - well clear
MAX_ICON_DIMENSION = 40  # real icons run ~22-32px per side - well clear of
                          # unrelated wide/tall content (chat lines, etc.)
MAX_ICONS = 6

TEMPLATE_HALF = 16
SEARCH_HALF = 22


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

    @staticmethod
    def _locate_icons(frame_bgr: np.ndarray, anchor_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find where the (up to 4) icons actually are, whichever layout is
        active, by locating real icon-sized blobs rather than assuming a
        fixed position/pitch for either layout - see the module comment for
        why a fixed pitch isn't reliable. Returns [(dx, dy), ...] offsets
        from the anchor, ordered start-to-end along whichever axis they're
        actually spread across (x for the horizontal layout, y for the
        vertical one)."""
        ax, ay = anchor_pos
        rx0, ry0, rx1, ry1 = ICON_SEARCH_REGION
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
            # Real icons run roughly 22-32px in each dimension - excluding
            # anything wider/taller than that rules out unrelated content the
            # search region can extend into below a shorter (horizontal-
            # layout) dialog, e.g. game chat/log text lines, which are much
            # wider than an icon and were otherwise big enough by area alone
            # to outrank and displace real icons from the top-4 selection
            # below (confirmed: "pick the odd-failed.jpg" picked up a
            # 250px-wide chat line instead of two of the real icons).
            if area < MIN_ICON_AREA or bw > MAX_ICON_DIMENSION or bh > MAX_ICON_DIMENSION:
                continue
            candidates.append((x0 + bx + bw // 2 - ax, y0 + by + bh // 2 - ay, area))
        if len(candidates) < 2:
            return []

        candidates.sort(key=lambda c: -c[2])
        candidates = candidates[:MAX_ICONS]

        xs = [c[0] for c in candidates]
        ys = [c[1] for c in candidates]
        if (max(xs) - min(xs)) >= (max(ys) - min(ys)):
            candidates.sort(key=lambda c: c[0])
        else:
            candidates.sort(key=lambda c: c[1])
        return [(c[0], c[1]) for c in candidates]

    @staticmethod
    def _comparison_half_sizes(icon_offsets: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Shrink TEMPLATE_HALF/SEARCH_HALF when icons are packed tighter than
        those defaults allow, so the search window for one icon can't bleed
        into a neighboring row/column - real captures showed row pitch as
        tight as ~24-28px between short icons, well under SEARCH_HALF's
        default 44px-wide (22*2) window."""
        if len(icon_offsets) < 2:
            return TEMPLATE_HALF, SEARCH_HALF
        gaps = []
        for i in range(len(icon_offsets) - 1):
            dx0, dy0 = icon_offsets[i]
            dx1, dy1 = icon_offsets[i + 1]
            gaps.append(abs(dx1 - dx0) + abs(dy1 - dy0))
        min_gap = min(gaps) if gaps else max(TEMPLATE_HALF, SEARCH_HALF) * 2
        template_half = min(TEMPLATE_HALF, max(8, min_gap // 2 - 2))
        search_half = min(SEARCH_HALF, max(template_half + 2, min_gap // 2))
        return template_half, search_half

    def find_odd_icon(self, frame: np.ndarray, anchor_pos: Tuple[int, int]) -> Optional[int]:
        """Return the 0-based index of the icon that looks different from
        the others, or None if the icon slots couldn't be found/cropped.
        Icons are ordered start-to-end along whichever axis they're actually
        spread across (see _locate_icons), matching the on-screen option
        numbering, so this index is directly usable for keyboard selection
        (down-arrow this many times, then enter).

        Uses a small local search window per comparison (not just the exact
        icon centers) - a real capture showed two icons of the *same* item
        scoring as low as 0.14 similarity on an exact fixed-position
        comparison, purely from a few pixels of rendering/alignment jitter,
        which would have been mistaken for a real visual difference."""
        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        ax, ay = anchor_pos
        h, w = frame_bgr.shape[:2]

        icon_offsets = self._locate_icons(frame_bgr, anchor_pos)
        if len(icon_offsets) < 2:
            return None
        template_half, search_half = self._comparison_half_sizes(icon_offsets)

        templates = []
        searches = []
        # icon_offsets is already in on-screen option order (see
        # _locate_icons), and every entry came from within this same frame,
        # so none should fail the bounds check below - but if one somehow
        # does, bail entirely rather than silently comparing a shifted subset
        # under indices that would no longer match the real option numbers.
        for dx, dy in icon_offsets:
            cx, cy = ax + dx, ay + dy
            tx0, ty0 = cx - template_half, cy - template_half
            tx1, ty1 = cx + template_half, cy + template_half
            sx0, sy0 = cx - search_half, cy - search_half
            sx1, sy1 = cx + search_half, cy + search_half
            if sx0 < 0 or sy0 < 0 or sx1 > w or sy1 > h:
                return None
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
