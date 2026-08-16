import threading
import time
from typing import Callable, Optional
from bot.config_manager import ConfigManager
from bot.window_manager import WindowManager
from bot.char_detector import CharDetector
from bot.input_handler import InputHandler
from bot.repositioner import Repositioner
from bot.timed_action_manager import TimedAction, TimedActionManager
from bot.gm_detector import GMDetector
from bot.alarm import Alarm
from bot.telegram_notifier import TelegramNotifier

GM_CHECK_INTERVAL = 2.0

class Status:
    def __init__(self):
        self.running = False
        self.state = "Idle"
        self.char_x = None
        self.facing = "left"
        self.calib_x = None
        self.gm_detected = False

class BotEngine:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.window = WindowManager()
        self.detector = CharDetector(self.window)
        self.inp = InputHandler()
        self.repositioner = Repositioner(self.detector, self.inp)
        self.timed = TimedActionManager(self.inp)
        self.gm = GMDetector(self.window)
        self.alarm = Alarm()
        self.telegram = TelegramNotifier()
        self.status = Status()
        self._thread = None
        self._stop_event = threading.Event()
        self._last_attack = 0.0
        self._idle_baseline_x = None
        self._idle_since = time.time()
        self._last_gm_check = 0.0
        self._apply_config()

    def _apply_config(self):
        cfg = self.config.data

        self.detector.set_region(cfg["minimap_x"], cfg["minimap_y"], cfg["minimap_w"], cfg["minimap_h"])
        self.detector.r_min = cfg["yellow_r_min"]
        self.detector.g_min = cfg["yellow_g_min"]
        self.detector.b_max = cfg["yellow_b_max"]

        r = self.repositioner
        r.enabled = cfg["reposition_enabled"]
        r.calib_x = cfg["calib_x"]
        r.l_offset = cfg["l_offset"]
        r.r_offset = cfg["r_offset"]
        r.tolerance = cfg["tolerance"]
        r.facing_l_key = cfg["facing_l_key"]
        r.facing_r_key = cfg["facing_r_key"]
        r.preferred_facing = cfg.get("preferred_facing", "right")
        r.snap_facing = cfg.get("snap_facing", True)
        r.simple_reposition = cfg["simple_reposition"]
        r.aux_key = cfg["aux_move_key"]
        r.hold_aux = cfg["hold_aux_key"]

        self.inp.method = "postmessage" if cfg["background_mode"] else cfg["input_method"]
        if self.window.get_target():
            self.inp.hwnd = self.window.get_target()

        self.timed.actions = [TimedAction.from_dict(a) for a in cfg.get("timed_actions", [])]

        self.telegram.token = cfg.get("telegram_token", "")
        self.telegram.chat_id = cfg.get("telegram_chat_id", "")
        self.telegram.cooldown = cfg.get("telegram_cooldown", 30.0)

    def start(self):
        if self.status.running:
            return

        self._apply_config()
        self._stop_event.clear()
        self.status.running = True
        self.status.state = "Running"
        self._idle_baseline_x = None
        self._idle_since = time.time()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.status.running = False
        self.status.state = "Stopped"
        self.alarm.stop()

    def idle_countdown(self) -> Optional[float]:
        """Seconds until the next idle move fires, or None if disabled."""
        if not self.config.data.get("idle_move_enabled") or self.config.data.get("background_mode"):
            return None
        remaining = self.config.data["idle_move_every"] - (time.time() - self._idle_since)
        return max(0.0, remaining)

    def calibrate(self) -> Optional[int]:
        x = self.repositioner.calibrate()
        if x is not None:
            self.config.set("calib_x", x)
            self.config.save()
            self.status.calib_x = x
        return x

    def _loop(self):
        cfg = self.config.data
        tick = 0.05

        try:
            while not self._stop_event.is_set():
                t0 = time.time()

                if not self.window.is_valid():
                    self.status.state = "No window"
                    time.sleep(0.5)
                    continue

                if cfg.get("gm_alarm_enabled") and t0 - self._last_gm_check >= GM_CHECK_INTERVAL:
                    self._last_gm_check = t0
                    was_detected = self.status.gm_detected
                    self.status.gm_detected = self.gm.check()
                    if self.status.gm_detected:
                        self.alarm.start()
                        if not was_detected and cfg.get("telegram_alert_enabled"):
                            self.telegram.notify_async(self.gm.last_frame, caption="kin.png detected!")
                    else:
                        self.alarm.stop()

                if self.status.gm_detected:
                    self.status.state = "GM / ANTI-BOT CHECK - PAUSED"
                    time.sleep(0.2)
                    continue

                char_x = self.detector.detect_char_x()
                self.status.char_x = char_x
                self.status.facing = self.repositioner.current_facing
                self.status.calib_x = self.repositioner.calib_x

                if cfg["idle_move_enabled"] and not cfg["background_mode"]:
                    now = time.time()
                    if char_x is not None:
                        if self._idle_baseline_x is None or abs(char_x - self._idle_baseline_x) >= 1:
                            self._idle_baseline_x = char_x
                            self._idle_since = now

                    if now - self._idle_since >= cfg["idle_move_every"]:
                        self.status.state = "Idle Move"
                        self.timed._focus_game()
                        r = self.repositioner
                        idle_key = r.facing_r_key if r.preferred_facing == "right" else r.facing_l_key
                        self.inp.key_press(idle_key, 0.5)
                        self._idle_baseline_x = None
                        self._idle_since = now
                        self._last_attack = time.time()
                        continue

                if self.timed.any_ready():
                    self.status.state = "Buffing"
                    self.timed.check_and_execute()
                    continue

                if not cfg["background_mode"]:
                    direction = self.repositioner.check_direction()
                    if direction and not self.repositioner.is_repositioning:
                        self.status.state = f"Repositioning {direction}"
                        self.repositioner.do_reposition(direction)
                        continue

                if time.time() - self._last_attack >= cfg["attack_interval"]:
                    if not cfg["background_mode"]:
                        self.timed._focus_game()
                    self.inp.key_press(cfg["attack_key"], 0.05)
                    self._last_attack = time.time()
                    self.status.state = "Attacking"

                elapsed = time.time() - t0
                remaining = tick - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status.state = f"Error: {e}"
        finally:
            self.alarm.stop()
            if not self.status.state.startswith("Error"):
                self.status.state = "Stopped"
            self.status.running = False
