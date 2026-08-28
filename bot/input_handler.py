import ctypes
from ctypes import wintypes as wt
import time
from typing import Optional
import win32api
import win32con

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 2
KEYEVENTF_SCANCODE = 8
KEYEVENTF_EXTENDEDKEY = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_long),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("padding", ctypes.c_byte * 32),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wt.DWORD),
        ("union", _INPUT_UNION),
    ]

# key -> (scan code, extended)
SCAN_MAP = {
    "esc": (1, False), "escape": (1, False),
    "1": (2, False), "2": (3, False), "3": (4, False), "4": (5, False), "5": (6, False),
    "6": (7, False), "7": (8, False), "8": (9, False), "9": (10, False), "0": (11, False),
    "backspace": (14, False),
    "tab": (15, False),
    "q": (16, False), "w": (17, False), "e": (18, False), "r": (19, False), "t": (20, False),
    "y": (21, False), "u": (22, False), "i": (23, False), "o": (24, False), "p": (25, False),
    "a": (30, False), "s": (31, False), "d": (32, False), "f": (33, False), "g": (34, False),
    "h": (35, False), "j": (36, False), "k": (37, False), "l": (38, False),
    "enter": (28, False),
    "shift": (42, False), "lshift": (42, False),
    "ctrl": (29, False), "lctrl": (29, False),
    "alt": (56, False), "lalt": (56, False),
    "space": (57, False),
    "z": (44, False), "x": (45, False), "c": (46, False), "v": (47, False), "b": (48, False),
    "n": (49, False), "m": (50, False),
    "f1": (59, False), "f2": (60, False), "f3": (61, False), "f4": (62, False), "f5": (63, False),
    "f6": (64, False), "f7": (65, False), "f8": (66, False), "f9": (67, False), "f10": (68, False),
    "f11": (87, False), "f12": (88, False),
    "up": (72, True), "down": (80, True), "left": (75, True), "right": (77, True),
    "insert": (82, True), "delete": (83, True), "home": (71, True), "end": (79, True),
    "pageup": (73, True), "pagedown": (81, True),
    "`": (41, False),
}

MAPVK_VSC_TO_VK = 1

def _make_input(scan: int, extended: bool, key_up: bool) -> INPUT:
    flags = KEYEVENTF_SCANCODE
    if extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    if key_up:
        flags |= KEYEVENTF_KEYUP

    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki.wVk = 0
    inp.union.ki.wScan = scan
    inp.union.ki.dwFlags = flags
    inp.union.ki.time = 0
    inp.union.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    return inp

def _send(scan: int, extended: bool, key_up: bool) -> None:
    inp = _make_input(scan, extended, key_up)
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

def _make_mouse_input(dx: int, dy: int, mouse_data: int, flags: int) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi.dx = dx
    inp.union.mi.dy = dy
    inp.union.mi.mouseData = mouse_data
    inp.union.mi.dwFlags = flags
    inp.union.mi.time = 0
    inp.union.mi.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    return inp

def _send_mouse(dx: int, dy: int, mouse_data: int, flags: int) -> None:
    inp = _make_mouse_input(dx, dy, mouse_data, flags)
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

def _to_absolute(x: int, y: int):
    sw = ctypes.windll.user32.GetSystemMetrics(0)
    sh = ctypes.windll.user32.GetSystemMetrics(1)
    return int(x * 65535 / max(1, sw - 1)), int(y * 65535 / max(1, sh - 1))

def _vk_from_scan(scan: int) -> int:
    return ctypes.windll.user32.MapVirtualKeyW(scan, MAPVK_VSC_TO_VK)

def _make_lparam(scan: int, extended: bool, key_up: bool) -> int:
    lparam = 1
    lparam |= (scan & 0xFF) << 16
    if extended:
        lparam |= 0x1000000
    if key_up:
        lparam |= 0xC0000000
    return lparam

class InputHandler:
    def __init__(self):
        self.method = "sendinput"
        self.hwnd: Optional[int] = None

    def _scan(self, key: str):
        return SCAN_MAP.get(key.lower().strip())

    def key_press(self, key: str, hold: float = 0.05):
        entry = self._scan(key)
        if entry is None:
            return
        scan, ext = entry

        if self.method == "postmessage" and self.hwnd:
            self._pm_send(scan, ext, False)
            time.sleep(hold)
            self._pm_send(scan, ext, True)
            return

        _send(scan, ext, False)
        time.sleep(hold)
        _send(scan, ext, True)

    def key_down(self, key: str):
        entry = self._scan(key)
        if entry is None:
            return
        scan, ext = entry

        if self.method == "postmessage" and self.hwnd:
            self._pm_send(scan, ext, False)
            return

        _send(scan, ext, False)

    def key_up(self, key: str):
        entry = self._scan(key)
        if entry is None:
            return
        scan, ext = entry

        if self.method == "postmessage" and self.hwnd:
            self._pm_send(scan, ext, True)
            return

        _send(scan, ext, True)

    def move_mouse(self, x: int, y: int):
        """Move the cursor to an absolute screen position. Foreground (SendInput)
        only - no PostMessage equivalent, since games rarely honor synthetic mouse
        messages the way they honor keyboard ones."""
        ax, ay = _to_absolute(x, y)
        _send_mouse(ax, ay, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)

    def scroll_at(self, x: int, y: int, notches: int = 3):
        """Move the cursor to (x, y) and scroll the wheel down by `notches` clicks."""
        if self.method == "postmessage":
            return
        self.move_mouse(x, y)
        time.sleep(0.05)
        _send_mouse(0, 0, -WHEEL_DELTA * notches, MOUSEEVENTF_WHEEL)

    def _pm_send(self, scan: int, extended: bool, key_up: bool):
        vk = _vk_from_scan(scan)
        lparam = _make_lparam(scan, extended, key_up)
        msg = win32con.WM_KEYUP if key_up else win32con.WM_KEYDOWN
        win32api.PostMessage(self.hwnd, msg, vk, lparam)
