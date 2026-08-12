from typing import List, Optional, Tuple
import numpy as np
import win32gui
import mss

class WindowManager:
    def __init__(self):
        self._hwnd: Optional[int] = None

    def list_windows(self) -> List[Tuple[int, str]]:
        windows = []

        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title.strip():
                    windows.append((hwnd, title))

        win32gui.EnumWindows(_cb, None)
        return windows

    def set_target(self, hwnd: int):
        self._hwnd = hwnd

    def get_target(self) -> Optional[int]:
        return self._hwnd

    def is_valid(self) -> bool:
        if not self._hwnd:
            return False
        return bool(win32gui.IsWindow(self._hwnd) and win32gui.IsWindowVisible(self._hwnd))

    def get_client_rect_screen(self) -> Optional[Tuple[int, int, int, int]]:
        """Return (left, top, right, bottom) of client area in screen coords."""
        if not self._hwnd:
            return None
        try:
            rect = win32gui.GetClientRect(self._hwnd)
            tl = win32gui.ClientToScreen(self._hwnd, (rect[0], rect[1]))
            br = win32gui.ClientToScreen(self._hwnd, (rect[2], rect[3]))
            return (tl[0], tl[1], br[0], br[1])
        except Exception:
            return None

    def capture_region(self, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        """Capture a region relative to the game window client area. Returns BGRA array."""
        rect = self.get_client_rect_screen()
        if rect is None:
            return None

        abs_x = rect[0] + x
        abs_y = rect[1] + y

        try:
            with mss.mss() as sct:
                mon = {"left": abs_x, "top": abs_y, "width": max(1, w), "height": max(1, h)}
                shot = sct.grab(mon)
                return np.array(shot)
        except Exception:
            return None
