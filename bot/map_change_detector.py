from typing import Optional
import numpy as np
import cv2
from bot.window_manager import WindowManager

# Fixed region (relative to the game window's client area) covering just the
# minimap's map-name text line ("Henesys Ruins", "Knight Stronghold", etc) -
# measured directly from real captures. Deliberately excludes the character
# avatar icon to its left (its idle-animation pose shifts frame to frame,
# which would cause false "map changed" positives) and the sub-area line
# below it (e.g. "Henesys Ruins Market" / "Armory 1"), keeping only the
# top-level map name. Always at the same position regardless of window size
# (same fixed-UI-pixel-size behavior confirmed throughout this project for
# every other in-game panel).
REGION = (45, 46, 108, 22)  # (x, y, w, h)


class MapChangeDetector:
    """Snapshots the minimap's map-name text once (set_baseline), then checks
    later captures against it - not by reading the name (no OCR needed, just
    like the other icon-comparison solvers), only by comparing pixels, since
    all that's needed is knowing whether the text changed at all."""

    def __init__(self, wm: WindowManager, threshold: float = 0.90):
        self.wm = wm
        self.threshold = threshold
        self.baseline: Optional[np.ndarray] = None

    def reset(self):
        """Discard the current baseline - call whenever the bot (re)starts,
        so each run compares against a freshly captured map name instead of
        a stale one from a previous session."""
        self.baseline = None

    def set_baseline(self) -> bool:
        """Capture and store the current map-name region as the reference to
        compare future captures against. Returns True iff it succeeded."""
        frame = self.wm.capture_region(*REGION)
        if frame is None:
            return False
        self.baseline = np.ascontiguousarray(frame[:, :, :3])
        return True

    def changed(self) -> bool:
        """Return True iff the current map-name region no longer matches the
        stored baseline (i.e. the player is now on a different map). Returns
        False (not-changed) if there's no baseline yet or a capture fails -
        callers should establish a baseline before relying on this."""
        if self.baseline is None:
            return False
        frame = self.wm.capture_region(*REGION)
        if frame is None:
            return False
        current = np.ascontiguousarray(frame[:, :, :3])
        if current.shape != self.baseline.shape:
            return False
        result = cv2.matchTemplate(current, self.baseline, cv2.TM_CCOEFF_NORMED)
        score = float(result[0, 0])
        return score < self.threshold
