import os
import threading
import winsound

class Alarm:
    """Plays a repeating beep, or a looping custom .wav if set, until stopped."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self.sound_path = ""

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
        path = self.sound_path
        use_custom = bool(path) and os.path.isfile(path)

        if use_custom:
            try:
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                self._stop_event.wait()
            except Exception:
                use_custom = False
            finally:
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:
                    pass
            if use_custom:
                return

        while not self._stop_event.is_set():
            try:
                winsound.Beep(2000, 400)
            except Exception:
                pass
            if self._stop_event.wait(0.3):
                break
