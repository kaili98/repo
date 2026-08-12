import time
from typing import List
import win32gui
from bot.input_handler import InputHandler

class TimedAction:
    def __init__(self, name: str, key: str, interval: float, hold: float, post_delay: float,
                 enabled: bool = True, pre_delay: float = 0.0):
        self.name = name
        self.key = key
        self.interval = interval
        self.enabled = enabled
        self.pre_delay = pre_delay
        self.hold = hold
        self.post_delay = post_delay
        self._last_used = 0.0

    def is_ready(self) -> bool:
        if not (self.enabled and self.key.strip()):
            return False
        return time.time() - self._last_used >= self.interval

    def mark_used(self):
        self._last_used = time.time()

    def time_remaining(self) -> float:
        return max(0.0, self.interval - (time.time() - self._last_used))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "key": self.key,
            "interval": self.interval,
            "enabled": self.enabled,
            "pre_delay": self.pre_delay,
            "hold": self.hold,
            "post_delay": self.post_delay,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TimedAction":
        return cls(
            name=d.get("name", "Buff"),
            key=d.get("key", ""),
            interval=float(d.get("interval", 30.0)),
            enabled=bool(d.get("enabled", True)),
            pre_delay=float(d.get("pre_delay", 0.0)),
            hold=float(d.get("hold", 0.05)),
            post_delay=float(d.get("post_delay", 0.5)),
        )

class TimedActionManager:
    def __init__(self, inp: InputHandler):
        self.inp = inp
        self.actions: List[TimedAction] = []

    def any_ready(self) -> bool:
        return any(a.is_ready() for a in self.actions)

    def _focus_game(self):
        if self.inp.method == "postmessage":
            return
        hwnd = self.inp.hwnd
        if not hwnd:
            return
        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, 9)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.15)
        except Exception:
            return

    def check_and_execute(self):
        for action in self.actions:
            if not action.is_ready():
                continue
            self._focus_game()
            if action.pre_delay > 0:
                time.sleep(action.pre_delay)
            self.inp.key_press(action.key, action.hold)
            action.mark_used()
            if action.post_delay > 0:
                time.sleep(action.post_delay)
