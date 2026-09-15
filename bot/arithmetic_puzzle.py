import os
import sys
from typing import List, Optional, Tuple
import numpy as np
import cv2
import joblib

from bot.digit_features import hog_features

def _resource_path(filename: str) -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "images", filename)

# Filename-prefix -> character label, for the non-alphanumeric template names
# (used by train_digit_model.py, which builds the classifiers below from the
# real crops in images/digits/eq and images/digits/opt).
_LABEL_FOR_PREFIX = {"plus": "+", "minus": "-", "eq": "=", "q": "?"}

# A trained classifier's predicted probability for its top label, below which
# a read is treated as "unrecognized" (None) rather than trusted. Calibrated
# via leave-one-source-out validation across every real template on file: at
# 0.70, zero wrong predictions ever scored this high in either scale (worst
# wrong: 0.640 eq-scale, 0.683 opt-scale), while plenty of correct ones
# clear it easily - same "never guess wrong, prefer no answer" principle as
# the old margin-based check, just calibrated for probability outputs
# instead of raw pixel-correlation scores.
ML_TRUST_THRESHOLD = 0.70

# Equation/options region offsets (dx, dy) from the anchor's matched top-left,
# calibrated from 3 real captures (arithmetic.png, arithmetic1.jpg,
# arithmetic2.jpg). The anchor is a crop of the constant "Click the answer:"
# instruction text.
EQUATION_REGION = (0, 16, 90, 38)     # (x0, y0, x1, y1) offsets from anchor
# x1 widened from 150 to 170: when all 4 options are 2-digit numbers (e.g.
# "13, 14, 15, 11"), the 4th option's digits extend to x=151-152 - past the
# old boundary, silently clipping them off and dropping that whole option
# from _cluster_options (single leftover arrow-only box gets filtered out),
# which then made solve() fail to find a matching option even though the
# equation itself was read correctly.
OPTIONS_REGION = (10, 53, 170, 64)
# Real captures showed this dialog has a *second* options layout: instead of
# 4 options side-by-side on one row, some occurrences stack them vertically
# (one per line, same x). OPTION_ROW_HEIGHT is the measured spacing between
# those stacked rows; OPTIONS_REGION's own height is reused as each row's
# scan height. Scanning each row as its own narrow strip (rather than one
# tall region) matters: below the *horizontal* layout's single row sits the
# "Mistakes left: N" line, which contains a real digit - segmenting row by
# row (and reusing the same per-character classifier used for equation
# text, which safely rejects letter shapes) keeps that line from ever being
# read as a 5th option, since _cluster_options discards a row's whole
# reading the moment any character in it fails digit classification, and a
# word like "stakes" fails long before its trailing digit is ever reached.
OPTION_ROW_HEIGHT = 18
# How many rows to scan at most (vertical layout only - the horizontal
# layout has no such cap, it clusters however many groups it finds in its
# one shared row). Answer selection is keyboard-driven (down-arrow N times +
# enter), so raising this doesn't require guessing any new pixel positions,
# just scanning a couple more row-bands - the extra bands safely find no
# content (and contribute nothing) on any real capture we've seen, which all
# show exactly 4 options.
OPTION_ROW_COUNT = 6
OPTION_CLUSTER_GAP = 10                # x-gap (px) that separates one option from the next
BG_THRESHOLD = 15

# Where to click first to skip/complete the instruction text's typewriter
# animation before reading the equation/options - same technique proven
# necessary for the other 3 dialog types. Lands within the "Click the
# answer:" anchor line itself (y=7 of 16px tall), well above EQUATION_REGION
# (starts at y=16), so it can't land on the equation/options text. Pushed
# further right (was x=50) - lands just past the end of "answer:" in blank
# space instead of on top of the word.
TEXT_CLICK_OFFSET: Tuple[int, int] = (90, 7)
# Only merge components whose x-ranges actually overlap (e.g. the two strokes
# of '=' or the curve+dot of '?') - a positive gap here also merged adjacent
# digits within a 2-digit option (e.g. '1' and '8' in "18"), misreading it as
# a single glyph ("8").
MERGE_GAP = 0


def _segment_chars(region_bgr: np.ndarray, bg_thresh: int = BG_THRESHOLD,
                    merge_gap: int = MERGE_GAP) -> List[List[int]]:
    """Return [x0,y0,x1,y1] boxes for each character-like connected component
    in `region_bgr`, left-to-right, merging components whose x-ranges overlap
    (e.g. the two strokes of '=' or the curve+dot of '?')."""
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    bg = np.median(gray[0:2, :])
    mask = (np.abs(gray.astype(int) - int(bg)) > bg_thresh).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 3:
            continue
        boxes.append([x, y, x + w, y + h])
    boxes.sort(key=lambda b: b[0])

    merged: List[List[int]] = []
    for b in boxes:
        placed = False
        for m in merged:
            if b[0] <= m[2] + merge_gap and b[2] >= m[0] - merge_gap:
                m[0] = min(m[0], b[0]); m[1] = min(m[1], b[1])
                m[2] = max(m[2], b[2]); m[3] = max(m[3], b[3])
                placed = True
                break
        if not placed:
            merged.append(list(b))
    merged.sort(key=lambda b: b[0])
    return merged


GLYPH_PAD = 2  # margin added around each glyph's tight bbox before matching -
                # a bare '-' is only 1-2px tall, and matching that degenerate a
                # crop directly (with zero/near-zero variance) gives useless,
                # near-random correlation scores.


def _crop_glyph(region: np.ndarray, box: List[int], pad: int = GLYPH_PAD) -> np.ndarray:
    h, w = region.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
    return region[y0:y1, x0:x1]


CANON_SIZE = 24


def _canonicalize(gray: np.ndarray, size: int = CANON_SIZE) -> np.ndarray:
    """Resize a tight glyph crop to fit within a fixed size x size canvas
    while preserving aspect ratio (pad, don't stretch to fill). This dialog's
    equation text renders at a visibly larger font (~12px tall) than its
    options text (~7px tall); comparing across those scales by stretching one
    to match the other's raw pixel dimensions distorts proportions badly
    (e.g. a tall-narrow '4' squished into a short-wide box), which is why an
    otherwise-clear digit could fail to match anything. Canonicalizing both
    the live glyph and every stored template to this same padded size keeps
    proportions faithful regardless of source scale, so a digit seen so far
    only in one context can still be recognized via the other."""
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.uint8)
    margin = 2
    scale = min((size - margin) / h, (size - margin) / w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(gray, (new_w, new_h))
    # Use the crop's most common pixel value as the padding fill, not a tiny
    # corner sample - a 2x2 corner can land on a JPEG-compression-noised or
    # anti-aliased pixel and end up meaningfully darker/lighter than the true
    # background, which then reads as a totally different backdrop from a
    # template's (clean PNG) padding and tanks the correlation score even
    # when the glyph's own ink shape matches well. Background pixels vastly
    # outnumber ink pixels in a tight glyph crop, so the mode is reliable.
    values, counts = np.unique(gray, return_counts=True)
    bg = int(values[np.argmax(counts)])
    canvas = np.full((size, size), bg, dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _cluster_options(boxes: List[List[int]], gap_thresh: int = OPTION_CLUSTER_GAP) -> List[List[List[int]]]:
    """Group character boxes into per-option clusters by large x-gaps, then
    drop the leading separator arrow ('>' or the selection cursor) from each
    cluster."""
    if not boxes:
        return []
    clusters = [[boxes[0]]]
    for b in boxes[1:]:
        if b[0] - clusters[-1][-1][2] > gap_thresh:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    return [c[1:] for c in clusters if len(c) > 1]


class ArithmeticPuzzleSolver:
    """Locates the arithmetic Human Check dialog via a stable anchor crop of
    its instruction text, reads the equation (addition/subtraction only) and
    the up-to-4 answer options via a pair of small trained classifiers (one
    per font scale - see below), computes the correct answer, and finds
    which option matches it.

    The digit/operator classifiers are HOG-features + SVM (see
    bot/digit_features.py and train_digit_model.py), trained offline from the
    real, hand-verified glyph crops in images/digits/eq and images/digits/opt
    and saved as images/digits/eq_model.joblib and opt_model.joblib. This
    replaced an earlier raw-pixel-correlation matcher (cv2.matchTemplate
    against those same crops directly) once real captures kept showing the
    same digit rendered at meaningfully different pixel widths (e.g. a "3"
    seen at both 7px and 10px wide) - pixel correlation doesn't generalize
    across that, but gradient-orientation features do much better (measured
    via leave-one-source-out validation: eq-scale accuracy went from 67% to
    84%, opt-scale from 93% to 97%)."""

    # Lower than the other 3 solvers' 0.85: real captures showed a cursor/
    # tooltip icon hovering near this anchor's text with an inconsistent
    # pose from one capture to the next (sometimes barely overlapping,
    # sometimes covering much of "answer:") - even with a clean, cursor-free
    # anchor image, real matches against a heavily-overlapped capture still
    # only scored ~0.88. The worst false-positive seen against the other 3
    # dialog types was 0.735, so 0.80 keeps a solid margin on both sides.
    def __init__(self, anchor_path: str = "arithmetic-anchor.png", threshold: float = 0.80):
        self.threshold = threshold
        self.anchor = None
        # Equation text renders at a visibly larger size (~12px tall) than
        # options text (~7px tall) in this dialog, so each scale gets its own
        # classifier rather than one trained across both.
        self._eq_model = self._load_model("eq_model.joblib")
        self._opt_model = self._load_model("opt_model.joblib")

        path = _resource_path(anchor_path)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self.anchor = img

    def _load_model(self, filename: str):
        path = _resource_path(os.path.join("digits", filename))
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    def _classify(self, glyph_bgr: np.ndarray, model) -> Optional[str]:
        """Return the model's predicted label for a single segmented
        character crop, or None if it's not confident enough.

        Requires the top class's predicted probability to clear
        ML_TRUST_THRESHOLD - a confident-but-wrong read (e.g. '+' misread as
        '-') silently flips the whole computed answer while still looking
        like a valid equation, which is worse than admitting "I don't know"
        (that safely falls through to the failure-notification + poll
        fallback). See ML_TRUST_THRESHOLD's own comment for how that cutoff
        was calibrated."""
        if model is None:
            return None
        gray = cv2.cvtColor(glyph_bgr, cv2.COLOR_BGR2GRAY)
        gh, gw = gray.shape[:2]
        if gh == 0 or gw == 0:
            return None
        canon_glyph = _canonicalize(gray)
        features = hog_features(canon_glyph).reshape(1, -1)
        proba = model.predict_proba(features)[0]
        best_idx = int(np.argmax(proba))
        best_score = float(proba[best_idx])
        if best_score < ML_TRUST_THRESHOLD:
            return None
        return str(model.classes_[best_idx])

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

    def _read_equation(self, frame_bgr: np.ndarray, anchor_pos: Tuple[int, int]) -> Optional[int]:
        """Read the equation and return its computed result, or None."""
        ax, ay = anchor_pos
        x0, y0, x1, y1 = EQUATION_REGION
        region = frame_bgr[ay + y0:ay + y1, ax + x0:ax + x1]
        boxes = _segment_chars(region)

        chars = []
        for box in boxes:
            glyph = _crop_glyph(region, box)
            label = self._classify(glyph, self._eq_model)
            if label == "=":
                break  # nothing past '=' matters
            if label is not None:
                chars.append(label)

        expr = "".join(chars)
        for op in ("+", "-"):
            if op in expr:
                left, _, right = expr.partition(op)
                if left.isdigit() and right.isdigit():
                    a, b = int(left), int(right)
                    return a + b if op == "+" else a - b
        return None

    def _read_options(self, frame_bgr: np.ndarray, anchor_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return [(value, on-screen option index), ...] for each option
        read, skipping any that couldn't be read cleanly.

        Scans up to OPTION_ROW_COUNT row bands rather than one fixed region,
        since this dialog has two different options layouts: 4 options on
        one horizontal row (all the content lands in row 0), or one option
        per row stacked vertically (each row band holds exactly one). Both
        are handled by the same per-row clustering - see OPTION_ROW_HEIGHT's
        comment for why scanning row by row (instead of one tall region) is
        what keeps the "Mistakes left: N" line safe from being misread.

        The option index is `row_i + cluster_i`, not the read option's
        position in the returned list - the two layouts never both have a
        nonzero value at once (vertical: row_i varies, exactly one cluster
        per row; horizontal: row_i is always 0, cluster_i varies), so the
        sum gives the true on-screen index either way, and stays correct
        even if some other option earlier in scan order failed to read and
        got skipped (which would silently shift a plain list-position index)."""
        ax, ay = anchor_pos
        x0, y0, x1, y1 = OPTIONS_REGION
        row_h = y1 - y0

        options = []
        for row_i in range(OPTION_ROW_COUNT):
            band_y0 = y0 + row_i * OPTION_ROW_HEIGHT
            band_y1 = band_y0 + row_h
            region = frame_bgr[ay + band_y0:ay + band_y1, ax + x0:ax + x1]
            if region.shape[0] == 0 or region.shape[1] == 0:
                continue
            boxes = _segment_chars(region)
            clusters = _cluster_options(boxes)

            for cluster_i, cluster in enumerate(clusters):
                digit_str = ""
                for box in cluster:
                    glyph = _crop_glyph(region, box)
                    label = self._classify(glyph, self._opt_model)
                    if label is None:
                        digit_str = None
                        break
                    digit_str += label
                if not digit_str:
                    continue
                value = int(digit_str)
                options.append((value, row_i + cluster_i))
        return options

    def solve(self, frame: np.ndarray, anchor_pos: Tuple[int, int]) -> Optional[int]:
        """Return the 0-based index of the correct answer option, or None if
        the equation/options couldn't be read or no option matches the
        computed result."""
        frame_bgr = np.ascontiguousarray(frame[:, :, :3])
        result = self._read_equation(frame_bgr, anchor_pos)
        if result is None:
            return None
        options = self._read_options(frame_bgr, anchor_pos)
        for value, option_index in options:
            if value == result:
                return option_index
        return None
