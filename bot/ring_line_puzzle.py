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

# Canvas region (the dark panel containing the ring + colored lines), as
# offsets from the anchor's matched top-left - calibrated from the one real
# capture on hand (click-the-ring-line.png). Generous on all sides since the
# ring-finding step below tolerates extra background/border pixels fine (it
# only reacts to ring-shaped, low-saturation blobs), unlike a tight crop
# that risks clipping the ring or a line's approach into it.
CANVAS_REGION = (-3, 25, 361, 320)  # (x0, y0, x1, y1) offsets from anchor

# A ring-outline arc fragment (the ring gets broken into 2+ disconnected
# arcs by every line crossing it - see find_ring) runs noticeably bigger in
# at least one dimension than this UI's text glyphs (~8-11px square), and
# well short of a whole word - this band keeps both out of the merge below.
RING_ARC_MIN_AREA = 40
RING_ARC_MIN_SIDE = 14
RING_ARC_MAX_SIDE = 35

# After merging nearby arc fragments, the combined ring blob's bounding box
# must land in this size/shape band - measured ring diameter was ~26-28px.
RING_MIN_SIDE = 12
RING_MAX_SIDE = 40
RING_MIN_ASPECT = 0.6
RING_MAX_ASPECT = 1.6
# A hollow ring outline fills roughly a third of its own bounding box right
# at the boundary; a solid/filled blob (mask fill near 1.0) or a sparse
# leftover text cluster (mask fill well under 0.15) are both rejected by
# picking the candidate closest to this fraction rather than just the
# biggest or first.
RING_TARGET_FILL_FRACTION = 0.3

MERGE_PAD = 4  # px gap allowed between two arc fragments before they're merged as one ring

# Ring-outline pixels themselves are low-saturation (white/light gray); the
# 4 line colors are all vividly saturated - this threshold separates them.
RING_SAT_MAX = 60
RING_VAL_MIN = 60

# Line-color sampling: only look within this fraction of the ring's own
# radius (near dead-center) so a second line merely passing near the ring's
# edge (without going through it) can't get counted alongside the real one.
SAMPLE_RADIUS_FRACTION = 0.7
LINE_SAT_MIN = 80
LINE_VAL_MIN = 60
HUE_CLUSTER_WIDTH = 6  # OpenCV hue units (0-179) treated as "the same line color"

# Never click on a guess - require the dominant hue to make up almost all of
# the sampled colorful pixels, and enough of them to not be noise/anti-
# aliasing artifacts, matching this codebase's "never guess wrong, prefer no
# answer" rule for every other auto-solver.
MIN_SAMPLE_PIXELS = 10
MIN_DOMINANT_FRACTION = 0.75


class RingLinePuzzleSolver:
    """Locates the "Click the line that passes through the ring." Verification
    dialog via a stable anchor crop of its instruction text, finds the ring
    (a hollow, low-saturation circle broken into several disconnected arcs by
    whichever line crosses it) and the dominant line color sampled from near
    its exact center, and returns the on-screen point to click.

    Unlike the 4 Human Check puzzle types, this dialog has no multiple-choice
    option list to navigate by keyboard - the answer is a literal point on
    the canvas, so the caller clicks it directly (see RingLinePuzzleSolver
    usage in engine.py)."""

    def __init__(self, anchor_path: str = "click-ring-line-anchor.png", threshold: float = 0.85):
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
    def _find_ring(canvas_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Return (cx, cy, radius) of the ring in `canvas_bgr`-relative
        coordinates, or None if no ring-shaped blob was found.

        The ring's own outline gets cut into multiple disconnected arcs
        wherever a colored line crosses over/under it, so a plain
        connected-component pass sees several small blobs, not one circle -
        this merges nearby ones back together before checking the combined
        shape against the expected ring size/hollowness."""
        hsv = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(int)
        val = hsv[:, :, 2].astype(int)
        mask = ((sat < RING_SAT_MAX) & (val > RING_VAL_MIN)).astype(np.uint8)

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if (area >= RING_ARC_MIN_AREA and bw <= RING_ARC_MAX_SIDE and bh <= RING_ARC_MAX_SIDE
                    and (bw >= RING_ARC_MIN_SIDE or bh >= RING_ARC_MIN_SIDE)):
                boxes.append([int(x), int(y), int(x + bw), int(y + bh)])

        merged = [list(b) for b in boxes]
        changed = True
        while changed:
            changed = False
            out = []
            for b in merged:
                bb = [b[0] - MERGE_PAD, b[1] - MERGE_PAD, b[2] + MERGE_PAD, b[3] + MERGE_PAD]
                placed = False
                for m in out:
                    if bb[0] <= m[2] and bb[2] >= m[0] and bb[1] <= m[3] and bb[3] >= m[1]:
                        m[0] = min(m[0], b[0]); m[1] = min(m[1], b[1])
                        m[2] = max(m[2], b[2]); m[3] = max(m[3], b[3])
                        placed = True
                        changed = True
                        break
                if not placed:
                    out.append(list(b))
            merged = out

        candidates = []
        for m in merged:
            bw, bh = m[2] - m[0], m[3] - m[1]
            if not (RING_MIN_SIDE <= bw <= RING_MAX_SIDE and RING_MIN_SIDE <= bh <= RING_MAX_SIDE):
                continue
            aspect = bw / bh
            if not (RING_MIN_ASPECT <= aspect <= RING_MAX_ASPECT):
                continue
            sub = mask[m[1]:m[3], m[0]:m[2]]
            fill = float(sub.sum()) / (bw * bh)
            candidates.append((m, fill))

        if not candidates:
            return None
        m, _ = min(candidates, key=lambda c: abs(c[1] - RING_TARGET_FILL_FRACTION))
        cx = (m[0] + m[2]) / 2
        cy = (m[1] + m[3]) / 2
        radius = ((m[2] - m[0]) + (m[3] - m[1])) / 4
        return cx, cy, radius

    @staticmethod
    def _dominant_line_point(canvas_bgr: np.ndarray, cx: float, cy: float, radius: float) -> Optional[Tuple[float, float]]:
        """Return the (x, y) centroid, in `canvas_bgr`-relative coordinates,
        of whichever line color dominates a small sample disk around the
        ring's center - or None if that isn't confidently a single color
        (too few colored pixels, or no clear majority hue)."""
        h, w = canvas_bgr.shape[:2]
        rr = radius * SAMPLE_RADIUS_FRACTION
        x0, y0 = max(0, int(cx - rr)), max(0, int(cy - rr))
        x1, y1 = min(w, int(cx + rr) + 1), min(h, int(cy + rr) + 1)
        if x1 <= x0 or y1 <= y0:
            return None

        region_hsv = cv2.cvtColor(canvas_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hue = region_hsv[:, :, 0].astype(int)
        sat = region_hsv[:, :, 1].astype(int)
        val = region_hsv[:, :, 2].astype(int)
        colorful = (sat > LINE_SAT_MIN) & (val > LINE_VAL_MIN)

        ys, xs = np.where(colorful)
        if len(xs) < MIN_SAMPLE_PIXELS:
            return None
        hues = hue[colorful]

        best_frac, best_mask = 0.0, None
        for center_hue in np.unique(hues):
            dist = np.minimum(np.abs(hues - center_hue), 180 - np.abs(hues - center_hue))
            cluster = dist <= HUE_CLUSTER_WIDTH
            frac = cluster.sum() / len(hues)
            if frac > best_frac:
                best_frac, best_mask = frac, cluster

        if best_mask is None or best_frac < MIN_DOMINANT_FRACTION:
            return None

        return float(xs[best_mask].mean() + x0), float(ys[best_mask].mean() + y0)

    def find_click_point(self, frame: np.ndarray, anchor_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Return the on-screen (frame-relative) point to click, or None if
        the ring or the line passing through it couldn't be confidently
        identified."""
        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        h, w = frame_bgr.shape[:2]
        ax, ay = anchor_pos
        x0r, y0r, x1r, y1r = CANVAS_REGION
        x0, y0 = max(0, ax + x0r), max(0, ay + y0r)
        x1, y1 = min(w, ax + x1r), min(h, ay + y1r)
        if x1 <= x0 or y1 <= y0:
            return None
        canvas = frame_bgr[y0:y1, x0:x1]

        ring = self._find_ring(canvas)
        if ring is None:
            return None
        cx, cy, radius = ring

        point = self._dominant_line_point(canvas, cx, cy, radius)
        if point is None:
            return None
        px, py = point
        return int(round(x0 + px)), int(round(y0 + py))
