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
from bot.player_detector import PlayerDetector
from bot.jump_left_detector import JumpLeftDetector
from bot.alarm import Alarm
from bot.telegram_notifier import TelegramNotifier
from bot.lie_detector import LieDetectorFlow

GM_CHECK_INTERVAL = 0.5          # cadence while nothing is detected - kept fast so a fresh
                                  # LD dialog gets caught quickly from a cold (idle) state
GM_CHECK_INTERVAL_ACTIVE = 0.3   # faster cadence while a check is currently up, so a
                                  # split-second close+reopen (failed LD attempt) isn't missed
GM_CLEAR_CONFIRM_TICKS = 3       # consecutive "not detected" ticks required before treating
                                  # the check as truly closed - a single miss is usually just
                                  # a mid-render/animation frame, not a real close, and acting
                                  # on it prematurely cancels/discards the active poll
GM_TRIGGER_CONFIRM_TICKS = 2     # consecutive "detected" ticks required before actually sending
                                  # a poll (the pause/alarm still react on the very first tick) -
                                  # a single stray false-positive match (e.g. right after the real
                                  # dialog closed) shouldn't be enough to fire a bogus new poll

PLAYER_CHECK_INTERVAL = 1.0      # cadence for scanning the minimap for other players

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
        self.player_detector = PlayerDetector(self.window)
        self.jump_left_detector = JumpLeftDetector()
        self.alarm = Alarm()
        self.player_alarm = Alarm()
        self.telegram = TelegramNotifier()
        self.telegram.set_command_handler(self._handle_telegram_command)
        self.lie_detector = LieDetectorFlow(self.telegram, self.inp, self.timed._focus_game)
        self.status = Status()
        self._thread = None
        self._detect_thread = None
        self._stop_event = threading.Event()
        self._last_attack = 0.0
        self._idle_baseline_x = None
        self._idle_since = time.time()
        self._last_gm_check = 0.0
        self._gm_clear_streak = 0
        self._gm_detect_streak = 0
        self._lie_detector_generation = 0
        self._lie_detector_pending = False
        self._seen_miss_since_trigger = True  # allow the very first trigger unconditionally
        self._last_double_jump = time.time()
        self._last_player_check = 0.0
        self._player_alarm_until = 0.0
        self._apply_config()

    def _apply_config(self):
        cfg = self.config.data

        self.detector.set_region(cfg["minimap_x"], cfg["minimap_y"], cfg["minimap_w"], cfg["minimap_h"])
        self.detector.r_min = cfg["yellow_r_min"]
        self.detector.g_min = cfg["yellow_g_min"]
        self.detector.b_max = cfg["yellow_b_max"]

        self.player_detector.set_region(cfg["minimap_x"], cfg["minimap_y"], cfg["minimap_w"], cfg["minimap_h"])

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

        self.alarm.volume = cfg.get("alarm_volume", 100) / 100.0
        self.player_alarm.volume = cfg.get("alarm_volume", 100) / 100.0

        self.telegram.token = cfg.get("telegram_token", "")
        self.telegram.chat_id = cfg.get("telegram_chat_id", "")
        self.telegram.cooldown = cfg.get("telegram_cooldown", 30.0)
        if self.telegram.enabled:
            # Listen for chat commands (e.g. /screenshot) even while the bot
            # itself isn't running - not gated behind start()/stop().
            self.telegram.start_polling()

    def start(self):
        if self.status.running:
            return

        self._apply_config()
        self._stop_event.clear()
        self.status.running = True
        self.status.state = "Running"
        self._idle_baseline_x = None
        self._idle_since = time.time()
        self._last_double_jump = time.time()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._detect_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detect_thread.start()

    def stop(self):
        self._stop_event.set()
        self.status.running = False
        self.status.state = "Stopped"
        self.alarm.stop()
        self.player_alarm.stop()
        self._player_alarm_until = 0.0

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

    def _capture_full_game_frame(self):
        rect = self.window.get_client_rect_screen()
        if rect is None:
            return None
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        if w <= 0 or h <= 0:
            return None
        return self.window.capture_region(0, 0, w, h)

    def _handle_telegram_command(self, text: str):
        cmd = text.split()[0].lower()
        if cmd == "/screenshot":
            frame = self._capture_full_game_frame()
            self.telegram.send_photo(frame, "Screenshot")

    def _scroll_dialog_target(self, cfg) -> Optional[tuple]:
        """Absolute screen (x, y) to scroll at - a fixed point within the game
        window's client area (0,0 = top-left), or None if there's no window."""
        rect = self.window.get_client_rect_screen()
        if rect is None:
            return None
        return rect[0] + cfg.get("dialog_scroll_x", 840), rect[1] + cfg.get("dialog_scroll_y", 350)

    def _trigger_lie_detector_async(self, cfg):
        """Click+scroll the dialog, capture a single screenshot once it settles,
        then send that screenshot with the poll - all off the main loop thread so
        detection keeps running. Guarded by a generation counter + a pending flag
        so a dialog closing/reopening before this finishes drops the (now stale)
        attempt instead of sending a mismatched screenshot or double-firing."""
        if self.lie_detector._awaiting or self._lie_detector_pending:
            return
        self._seen_miss_since_trigger = False  # require a fresh miss before the next resend
        self._lie_detector_pending = True
        self._lie_detector_generation += 1
        generation = self._lie_detector_generation
        threading.Thread(
            target=self._capture_and_trigger_lie_detector, args=(cfg, generation), daemon=True
        ).start()

    def _capture_and_trigger_lie_detector(self, cfg, generation):
        try:
            can_scroll = cfg.get("dialog_scroll_enabled", True) and self.inp.method != "postmessage"
            target = self._scroll_dialog_target(cfg) if can_scroll else None
            if target is not None:
                self.inp.scroll_at(target[0], target[1], cfg.get("dialog_scroll_notches", 12))
                # Give the in-game scroll animation time to actually finish before
                # capturing, otherwise the screenshot shows a mid-scroll, half-settled
                # state.
                time.sleep(cfg.get("dialog_scroll_delay", 2.0))
                if generation != self._lie_detector_generation:
                    return  # dialog closed/reopened while scrolling - stale, drop it
                frame = self._capture_full_game_frame()
            else:
                frame = self.gm.last_frame
            if generation != self._lie_detector_generation:
                return

            # The "jump+left" instruction banner only appears after the dialog has
            # been scrolled into view, so this check has to happen on the
            # post-scroll frame above, not the initial (pre-scroll) detection frame.
            if cfg.get("auto_solve_enabled") and frame is not None and self.jump_left_detector.check_frame(frame):
                self.lie_detector.auto_solve_jump_left()
                # Give the game a moment to settle after the move before
                # capturing, so the screenshot shows the result of the solve
                # rather than a mid-walk/mid-transition frame.
                time.sleep(2.0)
                if generation != self._lie_detector_generation:
                    return
                confirm_frame = self._capture_full_game_frame()
                self.telegram.send_photo(confirm_frame, "Auto-solved Jump + Move Left")
                return

            if cfg.get("telegram_alert_enabled"):
                self.lie_detector.trigger([frame])
        finally:
            self._lie_detector_pending = False

    def test_scroll(self) -> bool:
        """Perform the configured dialog scroll right now, so the position can be
        visually checked/tuned. Returns False if there's no target window."""
        cfg = self.config.data
        target = self._scroll_dialog_target(cfg)
        if target is None:
            return False
        self.inp.scroll_at(target[0], target[1], cfg.get("dialog_scroll_notches", 12))
        return True

    def _detection_loop(self):
        """Runs the anti-bot banner and player-marker scans on their own thread,
        decoupled from the action loop below. Both involve a screen capture plus
        several template matches, which can take long enough on slower hardware
        to noticeably stall attack/reposition timing if run inline with it - this
        keeps that cost from ever blocking the action loop, regardless of how slow
        a scan is on any given machine."""
        cfg = self.config.data

        try:
            while not self._stop_event.is_set():
                t0 = time.time()

                if not self.window.is_valid():
                    time.sleep(0.5)
                    continue

                # Poll faster while a check is currently up: this makes cancel() close the
                # poll promptly once the dialog actually clears, and makes a resend for a
                # fresh dialog prompt happen without delay once the flow goes idle again.
                check_interval = GM_CHECK_INTERVAL_ACTIVE if self.status.gm_detected else GM_CHECK_INTERVAL
                if cfg.get("gm_alarm_enabled") and t0 - self._last_gm_check >= check_interval:
                    self._last_gm_check = t0
                    detected_now = self.gm.check()
                    if detected_now:
                        self._gm_clear_streak = 0
                        self._gm_detect_streak += 1
                        self.status.gm_detected = True
                    else:
                        self._gm_detect_streak = 0
                        self._gm_clear_streak += 1
                        # Raw (undebounced) signal that the dialog visually went away at least
                        # once, even just for a single tick - required before a resend is
                        # allowed (see below). Deliberately NOT gated behind the slower 3-tick
                        # GM_CLEAR_CONFIRM_TICKS debounce: requiring a fully confirmed close
                        # first would miss a close+reopen that happens faster than that debounce
                        # window, which is exactly the failure mode this is avoiding.
                        self._seen_miss_since_trigger = True
                        if self._gm_clear_streak >= GM_CLEAR_CONFIRM_TICKS:
                            self.status.gm_detected = False
                        # else: a single miss is likely just a mid-render frame, not treated
                        # as a real close yet - keep the existing gm_detected/poll state.

                    if self.status.gm_detected:
                        # Gated on _gm_detect_streak (not the very first tick) so a lone
                        # false-positive blip rings the alarm for a moment with nothing to show
                        # for it in Telegram - the pause/PAUSED state above still reacts
                        # instantly regardless, since staying cautious on an unconfirmed hit is
                        # the safe default even if it turns out to be nothing.
                        if self._gm_detect_streak >= GM_TRIGGER_CONFIRM_TICKS:
                            self.alarm.start()
                        # _trigger_lie_detector_async no-ops while a poll/answer is already in
                        # flight, and re-arms itself the instant it's answered. Gated on
                        # _gm_detect_streak (unlike the alarm/pause above, which react on the
                        # very first tick) so a lone false-positive blip can't fire a bogus new
                        # poll on its own, AND on _seen_miss_since_trigger so a resend only
                        # happens once the dialog has actually been seen to go away (even
                        # briefly) since the last poll - not just because the previous poll got
                        # answered while the same dialog was still continuously showing. Fires
                        # whenever either auto-solve or the Telegram poll could apply - which of
                        # the two actually happens is decided after the scroll+capture, since the
                        # jump+left banner (if any) only appears post-scroll.
                        if (cfg.get("auto_solve_enabled") or cfg.get("telegram_alert_enabled")) \
                                and self._gm_detect_streak >= GM_TRIGGER_CONFIRM_TICKS \
                                and self._seen_miss_since_trigger:
                            self._trigger_lie_detector_async(cfg)
                    else:
                        self.alarm.stop()
                        self.lie_detector.cancel()
                        self._lie_detector_generation += 1  # drop any still-pending capture

                if cfg.get("player_detect_enabled") and t0 - self._last_player_check >= PLAYER_CHECK_INTERVAL:
                    self._last_player_check = t0
                    if self.player_detector.check():
                        if self._player_alarm_until == 0.0:
                            self.player_alarm.start()
                        # Extends the ring window while the player stays visible, instead of
                        # going silent after 3s with the threat still on screen.
                        self._player_alarm_until = t0 + cfg.get("player_alarm_duration", 3.0)

                if self._player_alarm_until and t0 >= self._player_alarm_until:
                    self.player_alarm.stop()
                    self._player_alarm_until = 0.0

                time.sleep(0.05)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            self.alarm.stop()
            self.player_alarm.stop()

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

                if self.status.gm_detected or self.lie_detector._awaiting:
                    # Also stays paused while a Lie Detector answer is still in flight
                    # (e.g. mid-exchange on "2 Actions" or "Others"), even if the check's
                    # on-screen banner itself has already cleared - so normal play doesn't
                    # resume out from under a reply you haven't finished sending yet.
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

                if cfg.get("double_jump_enabled") and \
                        time.time() - self._last_double_jump >= cfg.get("double_jump_interval", 40.0):
                    self.status.state = "Double Jump"
                    if not cfg["background_mode"]:
                        self.timed._focus_game()
                    self.inp.key_press("space", 0.05)
                    time.sleep(0.15)
                    self.inp.key_press("space", 0.05)
                    self._last_double_jump = time.time()
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
