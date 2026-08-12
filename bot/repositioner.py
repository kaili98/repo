import threading
import time
from typing import Optional
from bot.char_detector import CharDetector
from bot.input_handler import InputHandler

REPOSITION_TIMEOUT = 10.0
POLL_INTERVAL = 0.05

class Repositioner:
    def __init__(self, detector: CharDetector, inp: InputHandler):
        self.detector = detector
        self.inp = inp

        self.enabled = True
        self.calib_x: Optional[int] = None
        self.l_offset = 5
        self.r_offset = 5
        self.tolerance = 2

        self.facing_l_key = "left"
        self.facing_r_key = "right"
        self.current_facing = "right"

        self.preferred_facing = "right"
        self.snap_facing = True

        self.simple_reposition = True
        self.aux_key = "a"
        self.hold_aux = False

        self._lock = threading.Lock()
        self._repositioning = False

    def calibrate(self) -> Optional[int]:
        x = self.detector.detect_char_x()
        if x is not None:
            self.calib_x = x
        return self.calib_x

    def check_direction(self) -> Optional[str]:
        """Return 'left' or 'right' if repositioning is needed, else None."""
        if not (self.enabled and self.calib_x is not None):
            return None

        x = self.detector.detect_char_x()
        if x is None:
            return None

        home = self.calib_x
        if x < home - self.l_offset - self.tolerance:
            return "right"
        if x > home + self.r_offset + self.tolerance:
            return "left"
        return None

    def do_reposition(self, direction: str) -> bool:
        move_key = self.facing_l_key if direction == "left" else self.facing_r_key

        with self._lock:
            self._repositioning = True
            try:
                self.inp.key_down(move_key)
                if self.hold_aux and self.aux_key:
                    self.inp.key_down(self.aux_key)

                deadline = time.time() + REPOSITION_TIMEOUT
                while time.time() < deadline:
                    time.sleep(POLL_INTERVAL)
                    if self.check_direction() is None:
                        break

                if self.hold_aux and self.aux_key:
                    self.inp.key_up(self.aux_key)
                self.inp.key_up(move_key)
                self.current_facing = direction

                if self.snap_facing and self.preferred_facing != direction:
                    time.sleep(0.15)
                    if self.check_direction() is None:
                        snap_key = self.facing_l_key if self.preferred_facing == "left" else self.facing_r_key
                        self.inp.key_press(snap_key, 0.05)
                        self.current_facing = self.preferred_facing
            finally:
                self._repositioning = False

    @property
    def is_repositioning(self) -> bool:
        return self._repositioning
