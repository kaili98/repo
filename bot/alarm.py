import threading
import winsound

class Alarm:
    """Plays a repeating beep on a background thread until stopped."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.active:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                winsound.Beep(2000, 400)
            except Exception:
                pass
            if self._stop_event.wait(0.3):
                break
