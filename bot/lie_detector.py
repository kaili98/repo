import time
from typing import List, Optional
import numpy as np
from bot.input_handler import InputHandler
from bot.telegram_notifier import TelegramNotifier

SCREENSHOT_CAPTION = "Lie Detector detected!"
POLL_QUESTION = "Lie Detector detected - pick the answer"
POLL_OPTIONS = [
    "1st option", "2nd option", "3rd option", "4th option",
    "5th option", "6th option", "7th option", "jump,moveleft",
    "2 Actions: send 2 key names in chat",
    "Others: type your answer in chat",
]
JUMP_MOVE_LEFT_INDEX = 7
TWO_ACTIONS_INDEX = 8
OTHERS_INDEX = 9

TYPE_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789 ")

# Left/right need a real hold (e.g. to actually walk into a checkpoint), unlike a
# quick tap for something like a jump.
HELD_ACTION_KEYS = {"left", "right"}
HELD_ACTION_DURATION = 1.0
TAP_ACTION_DURATION = 0.05

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
        self._answered = False
        self._triggered_at = 0.0

    def trigger(self, frames: Optional[List[Optional[np.ndarray]]] = None) -> bool:
        """Send the screenshot(s) followed by the poll. No-ops while a previous
        poll/answer for this detection is still in flight. `frames` may contain
        more than one screenshot (e.g. a second one after scrolling the dialog to
        reveal options that didn't fit on screen). Returns True iff it actually
        sent (vs no-op'd), so callers can gate follow-up work (like a scrolled
        2nd screenshot) on this being a genuinely new poll, not a repeat tick
        while the same one is still outstanding."""
        if not self.telegram.enabled:
            return False
        if self._awaiting and time.time() - self._triggered_at < ANSWER_TIMEOUT:
            return False
        self._awaiting = True
        self._answered = False
        self._triggered_at = time.time()
        self.telegram.send_photos_then_poll(
            frames or [], SCREENSHOT_CAPTION, POLL_QUESTION, POLL_OPTIONS, self._on_answer
        )
        return True

    def cancel(self):
        """Call when the Lie Detector check is no longer detected - closes an
        outstanding, *unanswered* poll so a stale one can't be voted on late.
        Once you've actually picked an option, this becomes a no-op: the check's
        on-screen banner can disappear (e.g. right after the dismiss keypress for
        "2 Actions") well before you've finished replying, and we don't want that
        to cancel your still-pending exchange out from under you."""
        if not self._awaiting or self._answered:
            return
        self._awaiting = False
        self.telegram.close_active_poll()
        self.telegram.cancel_text_wait()

    def _on_answer(self, option_index: int):
        self._answered = True
        if option_index == OTHERS_INDEX:
            self.telegram.request_text(self._on_text)
            return
        if option_index == TWO_ACTIONS_INDEX:
            self._start_two_actions()
            return
        if option_index == JUMP_MOVE_LEFT_INDEX:
            try:
                self._do_jump_move_left()
            finally:
                self._awaiting = False
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

    def _start_two_actions(self):
        """The check needs two sequential in-game actions (e.g. jump, then move)
        rather than picking from a list. Enter dismisses the check's initial
        popup, then each of the next two chat messages is treated as one key
        name (not typed as literal characters) and pressed in order."""
        try:
            self.focus_fn()
            self.inp.key_press("enter", 0.05)
        except Exception:
            pass
        self.telegram.request_text(self._on_action_1)

    def _on_action_1(self, text: str):
        try:
            self._press_key_action(text)
        finally:
            self.telegram.request_text(self._on_action_2)

    def _on_action_2(self, text: str):
        try:
            self._press_key_action(text)
        finally:
            self._awaiting = False

    def auto_solve_jump_left(self):
        """Directly perform the 'jump,moveleft' macro (Enter, Space, hold Left
        1s) without going through Telegram at all - used when auto-solve is
        enabled and the jump+left LD variant is detected. No-ops if a poll-driven
        exchange is already in flight."""
        if self._awaiting:
            return
        self._awaiting = True
        self._answered = True  # nothing to answer - not a poll-driven flow
        try:
            self._do_jump_move_left()
        finally:
            self._awaiting = False

    def auto_solve_puzzle_click(self, screen_x: int, screen_y: int):
        """Directly click the answer icon at its on-screen position in the
        Human Check picture-difference puzzle - the dialog shows a
        pointer-cursor hint suggesting this is the intended interaction, and
        it sidesteps any risk of synthetic keyboard presses (Right Arrow x N +
        Enter) not registering reliably for this specific dialog. No further
        Telegram interaction needed."""
        if self._awaiting:
            return
        self._awaiting = True
        self._answered = True  # nothing to answer - not a poll-driven flow
        try:
            self.focus_fn()
            # The dialog needs a beat to become fully interactive - clicking
            # right away (especially right after the previous step's dialog
            # just closed) risks landing before the game is ready for it.
            time.sleep(1.0)
            self.focus_fn()
            self.inp.click_at(screen_x, screen_y)
            # A synthetic (SendInput) click doesn't reliably grant the window
            # OS focus the way a real click does - re-assert it once more right
            # after, so normal play (attacking etc.) resumes on a window that's
            # definitely focused rather than whatever had focus before.
            time.sleep(0.2)
            self.focus_fn()
        finally:
            self._awaiting = False

    def _do_jump_move_left(self):
        """Fixed macro for the 'jump,moveleft' option - dismiss with Enter, jump,
        then hold left to walk off. No further Telegram interaction needed."""
        self.focus_fn()
        self.inp.key_press("enter", 0.05)
        # The popup->gameplay transition needs a beat before the game actually
        # registers a jump - a 0.05s gap sometimes dropped the space press
        # entirely (left still landed since its hold is much longer at 1s).
        # 0.15s matches the gap already proven reliable for the double-jump
        # elsewhere in this codebase.
        time.sleep(0.15)
        self.focus_fn()
        self.inp.key_press("space", 0.1)
        time.sleep(0.15)
        self._press_key_action("left")

    def _press_key_action(self, key_name: str):
        key = key_name.strip().lower()
        hold = HELD_ACTION_DURATION if key in HELD_ACTION_KEYS else TAP_ACTION_DURATION
        self.focus_fn()
        self.inp.key_press(key, hold)

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
