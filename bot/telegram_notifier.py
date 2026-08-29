import json
import threading
import time
import urllib.request
from typing import Callable, Dict, Optional
import numpy as np
import cv2


class TelegramNotifier:
    """Sends photos/polls to a Telegram chat via the Bot API, and listens for poll
    answers and text replies via long polling, all off the calling thread."""

    def __init__(self, token: str = "", chat_id: str = "", cooldown: float = 30.0):
        self.token = token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self._last_sent = 0.0

        self._poll_callbacks: Dict[str, Callable[[int], None]] = {}
        self._text_callback: Optional[Callable[[str], None]] = None
        self._text_wait_since = 0.0
        self._active_poll_id: Optional[str] = None
        self._active_poll_message_id: Optional[int] = None
        self._callback_lock = threading.Lock()

        self._polling_thread: Optional[threading.Thread] = None
        self._polling_stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    # ---- Screenshot ----

    def notify_async(self, frame_bgra: np.ndarray, caption: str = ""):
        """Fire-and-forget: encode and POST the screenshot on a background thread.
        Silently no-ops if not configured or still within the cooldown window."""
        if not self.enabled or frame_bgra is None:
            return
        now = time.time()
        if now - self._last_sent < self.cooldown:
            return
        self._last_sent = now
        threading.Thread(target=self._send, args=(frame_bgra.copy(), caption), daemon=True).start()

    def _send(self, frame_bgra: np.ndarray, caption: str):
        self._encode_and_post_photo(frame_bgra, caption)

    def _encode_and_post_photo(self, frame_bgra: np.ndarray, caption: str):
        try:
            frame_bgr = np.ascontiguousarray(frame_bgra[:, :, :3])
            ok, buf = cv2.imencode(".png", frame_bgr)
            if not ok:
                return
            self._post_photo(buf.tobytes(), caption)
        except Exception:
            pass

    def _post_photo(self, png_bytes: bytes, caption: str):
        boundary = "----BotNotifierBoundary"

        def field(name: str, value: str) -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")

        parts = [field("chat_id", self.chat_id)]
        if caption:
            parts.append(field("caption", caption))
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="photo"; filename="screenshot.png"\r\n'
                f"Content-Type: image/png\r\n\r\n"
            ).encode("utf-8")
        )
        parts.append(png_bytes)
        parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(parts)

        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        urllib.request.urlopen(req, timeout=15)

    def send_photo(self, frame_bgra: Optional[np.ndarray], caption: str = ""):
        """Fire-and-forget: send a single screenshot right away, on its own
        background thread. Not gated by the notify_async cooldown."""
        if not self.enabled or frame_bgra is None:
            return
        threading.Thread(
            target=self._encode_and_post_photo, args=(frame_bgra.copy(), caption), daemon=True
        ).start()

    # ---- Poll flow ----

    def send_photos_then_poll(self, frames, caption: str,
                               question: str, options, on_answer: Callable[[int], None]):
        """Fire-and-forget: send each screenshot in order, then the poll, all as
        sequential requests on one background thread so they land in the chat in
        that order. Not gated by the notify_async cooldown - callers throttle
        their own resends."""
        if not self.enabled:
            return
        self.start_polling()
        frame_copies = [f.copy() for f in frames if f is not None]
        threading.Thread(
            target=self._send_photos_then_poll,
            args=(frame_copies, caption, question, list(options), on_answer),
            daemon=True,
        ).start()

    def _send_photos_then_poll(self, frames, caption: str,
                                question: str, options, on_answer: Callable[[int], None]):
        n = len(frames)
        for i, frame in enumerate(frames, start=1):
            frame_caption = f"{caption} ({i}/{n})" if n > 1 else caption
            self._encode_and_post_photo(frame, frame_caption)
        self._send_poll(question, options, on_answer)

    def send_poll(self, question: str, options, on_answer: Callable[[int], None]):
        """Fire-and-forget: send a single-choice poll and remember on_answer for
        when a poll_answer update for it arrives."""
        if not self.enabled:
            return
        self.start_polling()
        threading.Thread(target=self._send_poll, args=(question, list(options), on_answer), daemon=True).start()

    def _send_poll(self, question: str, options, on_answer: Callable[[int], None]):
        try:
            body = json.dumps({
                "chat_id": self.chat_id,
                "question": question,
                "options": options,
                "is_anonymous": False,
                "allows_multiple_answers": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{self.token}/sendPoll", data=body, method="POST"
            )
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            poll_id = data["result"]["poll"]["id"]
            message_id = data["result"]["message_id"]
            with self._callback_lock:
                self._poll_callbacks[poll_id] = on_answer
                self._active_poll_id = poll_id
                self._active_poll_message_id = message_id
        except Exception:
            pass

    def request_text(self, on_text: Callable[[str], None]):
        """Call on_text with the text of the next chat message received."""
        if not self.enabled:
            return
        self.start_polling()
        with self._callback_lock:
            self._text_wait_since = time.time()
            self._text_callback = on_text

    def close_active_poll(self):
        """Stop whichever poll is currently outstanding, if any (e.g. it's no
        longer relevant because the thing it asked about went away unanswered)."""
        with self._callback_lock:
            message_id = self._active_poll_message_id
            poll_id = self._active_poll_id
            self._active_poll_id = None
            self._active_poll_message_id = None
            if poll_id is not None:
                self._poll_callbacks.pop(poll_id, None)
        if message_id is not None:
            self._stop_poll_async(message_id)

    def cancel_text_wait(self):
        with self._callback_lock:
            self._text_callback = None

    def _stop_poll_async(self, message_id: int):
        threading.Thread(target=self._stop_poll, args=(message_id,), daemon=True).start()

    def _stop_poll(self, message_id: int):
        try:
            body = json.dumps({"chat_id": self.chat_id, "message_id": message_id}).encode("utf-8")
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{self.token}/stopPoll", data=body, method="POST"
            )
            req.add_header("Content-Type", "application/json")
            urllib.request.urlopen(req, timeout=15)
        except Exception:
            pass

    # ---- Long polling for updates ----

    def start_polling(self):
        if not self.enabled or (self._polling_thread and self._polling_thread.is_alive()):
            return
        self._polling_stop.clear()
        self._polling_thread = threading.Thread(target=self._poll_updates, daemon=True)
        self._polling_thread.start()

    def stop_polling(self):
        self._polling_stop.set()

    def _poll_updates(self):
        offset = None
        while not self._polling_stop.is_set():
            if not self.enabled:
                time.sleep(1)
                continue
            try:
                url = f"https://api.telegram.org/bot{self.token}/getUpdates?timeout=25"
                if offset is not None:
                    url += f"&offset={offset}"
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read())
                for upd in data.get("result", []):
                    offset = upd["update_id"] + 1
                    self._handle_update(upd)
            except Exception:
                time.sleep(2)

    def _handle_update(self, upd: dict):
        poll_answer = upd.get("poll_answer")
        if poll_answer:
            poll_id = poll_answer.get("poll_id")
            option_ids = poll_answer.get("option_ids") or []
            with self._callback_lock:
                cb = self._poll_callbacks.pop(poll_id, None)
                message_id = None
                if self._active_poll_id == poll_id:
                    message_id = self._active_poll_message_id
                    self._active_poll_id = None
                    self._active_poll_message_id = None
            if message_id is not None:
                self._stop_poll_async(message_id)
            if cb and option_ids:
                cb(option_ids[0])
            return

        message = upd.get("message")
        if message and "text" in message:
            if str(message.get("chat", {}).get("id")) != str(self.chat_id):
                return
            if message.get("date", 0) < self._text_wait_since:
                return
            with self._callback_lock:
                cb = self._text_callback
                self._text_callback = None
            if cb:
                cb(message["text"])
