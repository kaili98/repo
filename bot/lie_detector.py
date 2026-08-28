import time
from typing import List, Optional
import numpy as np
from bot.input_handler import InputHandler
from bot.telegram_notifier import TelegramNotifier

SCREENSHOT_CAPTION = "Lie Detector detected!"
POLL_QUESTION = "Lie Detector detected - pick the answer"
POLL_OPTIONS = [
    "1st option", "2nd option", "3rd option", "4th option",
    "5th option", "6th option", "7th option", "8th option",
    "Others: type your answer in chat",
]
OTHERS_INDEX = 8

TYPE_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789 ")

# How long to wait for an answer before allowing the poll to be re-sent
# (e.g. if it was never answered, or the app was restarted).
ANSWER_TIMEOUT = 300.0


class LieDetectorFlow:
    """Sends a Telegram poll when a Lie Detector check is detected, and relays the
    chosen answer (or freeform "Others" text reply) back into the game as keystrokes."""

    def __init__(self, telegram: TelegramNotifier, inp: InputHandler, focus_fn):
        self.telegram = telegram
        self.inp = inp
        self.focus_fn = focus_fn
        self._awaiting = False
        self._triggered_at = 0.0

    def trigger(self, frames: Optional[List[Optional[np.ndarray]]] = None):
        """Send the screenshot(s) followed by the poll. No-ops while a previous
        poll/answer for this detection is still in flight. `frames` may contain
        more than one screenshot (e.g. a second one after scrolling the dialog to
        reveal options that didn't fit on screen)."""
        if not self.telegram.enabled:
            return
        if self._awaiting and time.time() - self._triggered_at < ANSWER_TIMEOUT:
            return
        self._awaiting = True
        self._triggered_at = time.time()
        self.telegram.send_photos_then_poll(
            frames or [], SCREENSHOT_CAPTION, POLL_QUESTION, POLL_OPTIONS, self._on_answer
        )

    def cancel(self):
        """Call when the Lie Detector check is no longer detected - closes any
        outstanding poll/text wait so a stale one can't be answered late."""
        if not self._awaiting:
            return
        self._awaiting = False
        self.telegram.close_active_poll()
        self.telegram.cancel_text_wait()

    def _on_answer(self, option_index: int):
        if option_index == OTHERS_INDEX:
            self.telegram.request_text(self._on_text)
            return
        try:
            self._submit_downs(option_index)
        finally:
            self._awaiting = False

    def _on_text(self, text: str):
        try:
            self._type_text(text)
        finally:
            self._awaiting = False

    def _submit_downs(self, n: int):
        self.focus_fn()
        for _ in range(n):
            self.inp.key_press("down", 0.05)
            time.sleep(0.05)
        self.inp.key_press("enter", 0.05)

    def _type_text(self, text: str):
        self.focus_fn()
        for ch in text.lower():
            if ch in TYPE_CHARS:
                self.inp.key_press(ch, 0.04)
                time.sleep(0.03)
        self.inp.key_press("enter", 0.05)
