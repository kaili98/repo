import threading
import time
import urllib.request
import numpy as np
import cv2


class TelegramNotifier:
    """Sends a screenshot to a Telegram chat via the Bot API, off the calling thread."""

    def __init__(self, token: str = "", chat_id: str = "", cooldown: float = 30.0):
        self.token = token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self._last_sent = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

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
