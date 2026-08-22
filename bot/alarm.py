import io
import math
import struct
import threading
import wave
import winsound

TONE_HZ = 2000
TONE_MS = 400
SAMPLE_RATE = 44100

class Alarm:
    """Plays a repeating tone at an adjustable volume until stopped."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self.volume = 1.0  # 0.0 - 1.0

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
        data = self._generate_tone_wav()
        try:
            winsound.PlaySound(data, winsound.SND_MEMORY | winsound.SND_ASYNC | winsound.SND_LOOP)
            self._stop_event.wait()
            return
        except Exception:
            pass
        finally:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

        # Last-resort fallback if wave generation/playback failed for any reason.
        # Volume isn't controllable here - it's the raw system beep.
        while not self._stop_event.is_set():
            try:
                winsound.Beep(TONE_HZ, TONE_MS)
            except Exception:
                pass
            if self._stop_event.wait(0.3):
                break

    def _generate_tone_wav(self) -> bytes:
        n = int(SAMPLE_RATE * TONE_MS / 1000)
        amp = int(32767 * max(0.0, min(1.0, self.volume)))
        samples = [int(amp * math.sin(2 * math.pi * TONE_HZ * i / SAMPLE_RATE)) for i in range(n)]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(struct.pack(f"<{n}h", *samples))
        return buf.getvalue()
