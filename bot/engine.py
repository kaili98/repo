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
from bot.apple_count_puzzle import AppleCountPuzzleSolver, OPTION_OFFSETS as APPLE_OPTION_OFFSETS, TEXT_CLICK_OFFSET as APPLE_TEXT_CLICK_OFFSET
from bot.pick_picture_puzzle import PickPicturePuzzleSolver, OPTION_OFFSETS as PICK_OPTION_OFFSETS, TEXT_CLICK_OFFSET as PICK_TEXT_CLICK_OFFSET
from bot.pick_odd_puzzle import PickOddPuzzleSolver, CLICK_OFFSETS as ODD_CLICK_OFFSETS, TEXT_CLICK_OFFSET as ODD_TEXT_CLICK_OFFSET
from bot.arithmetic_puzzle import ArithmeticPuzzleSolver, TEXT_CLICK_OFFSET as ARITHMETIC_TEXT_CLICK_OFFSET
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
        self.apple_count_solver = AppleCountPuzzleSolver()
        self.pick_picture_solver = PickPicturePuzzleSolver()
        self.pick_odd_solver = PickOddPuzzleSolver()
        self.arithmetic_solver = ArithmeticPuzzleSolver()
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

    def _click_puzzle_answer(self, rect, screen_x, screen_y, generation, caption) -> bool:
        """Shared click+cleanup+confirm sequence for the puzzle auto-solvers."""
        self.lie_detector.auto_solve_puzzle_click(screen_x, screen_y)
        # Move the cursor off the icon row afterward so it doesn't sit on top of
        # (and visually block) a picture in the confirm screenshot below or the
        # next step's puzzle detection. The dialog is always centered, so
        # parking in the window's top-left corner is guaranteed clear of it
        # regardless of resolution.
        self.inp.move_mouse(rect[0] + 10, rect[1] + 10)
        time.sleep(2.0)
        if generation != self._lie_detector_generation:
            return True
        confirm_frame = self._capture_full_game_frame()
        self.telegram.send_photo(confirm_frame, caption)
        return True

    def _try_apple_count_puzzle(self, cfg, frame, generation) -> bool:
        """The "Count the apples" variant. `frame` is expected to already be
        settled (the click-on-dialog-to-skip-animation step happens once,
        centrally, in `_capture_and_trigger_lie_detector` before any solver
        is tried) - this just reads it."""
        anchor_pos = self.apple_count_solver.locate(frame)
        if anchor_pos is None:
            return False
        count = self.apple_count_solver.count_icons(frame, anchor_pos)
        if count is None:
            return False
        rect = self.window.get_client_rect_screen()
        if rect is None:
            return False
        dx, dy = APPLE_OPTION_OFFSETS[count - 1]
        screen_x = rect[0] + anchor_pos[0] + dx
        screen_y = rect[1] + anchor_pos[1] + dy
        return self._click_puzzle_answer(
            rect, screen_x, screen_y, generation,
            f"Auto-solved Human Check (counted {count} apples, clicked option {count})",
        )

    def _try_pick_picture_puzzle(self, cfg, frame, generation) -> bool:
        """The "click the picture that matches this one" variant. `frame` is
        expected to already be settled (see `_try_apple_count_puzzle`)."""
        anchor_pos = self.pick_picture_solver.locate(frame)
        if anchor_pos is None:
            return False
        match_index = self.pick_picture_solver.find_matching_option(frame, anchor_pos)
        if match_index is None:
            return False
        rect = self.window.get_client_rect_screen()
        if rect is None:
            return False
        dx, dy = PICK_OPTION_OFFSETS[match_index]
        screen_x = rect[0] + anchor_pos[0] + dx
        screen_y = rect[1] + anchor_pos[1] + dy
        return self._click_puzzle_answer(
            rect, screen_x, screen_y, generation,
            f"Auto-solved Human Check (matched picture, clicked option {match_index + 1})",
        )

    def _try_arithmetic_puzzle(self, cfg, frame, generation) -> bool:
        """The arithmetic (addition/subtraction) variant. `frame` is expected
        to already be settled (see `_try_apple_count_puzzle`)."""
        anchor_pos = self.arithmetic_solver.locate(frame)
        if anchor_pos is None:
            return False
        offset = self.arithmetic_solver.solve(frame, anchor_pos)
        if offset is None:
            return False
        rect = self.window.get_client_rect_screen()
        if rect is None:
            return False
        screen_x = rect[0] + anchor_pos[0] + offset[0]
        screen_y = rect[1] + anchor_pos[1] + offset[1]
        return self._click_puzzle_answer(
            rect, screen_x, screen_y, generation,
            "Auto-solved Human Check (arithmetic)",
        )

    def _try_pick_odd_puzzle(self, cfg, frame, generation) -> bool:
        """The "click the ONE picture that is different" variant (no
        reference icon shown - compare the options against each other).
        `frame` is expected to already be settled (see
        `_try_apple_count_puzzle`)."""
        anchor_pos = self.pick_odd_solver.locate(frame)
        if anchor_pos is None:
            return False
        odd_index = self.pick_odd_solver.find_odd_icon(frame, anchor_pos)
        if odd_index is None:
            return False
        rect = self.window.get_client_rect_screen()
        if rect is None:
            return False
        dx, dy = ODD_CLICK_OFFSETS[odd_index]
        screen_x = rect[0] + anchor_pos[0] + dx
        screen_y = rect[1] + anchor_pos[1] + dy
        return self._click_puzzle_answer(
            rect, screen_x, screen_y, generation,
            f"Auto-solved Human Check (picked the odd one out, clicked option {odd_index + 1})",
        )

    def _try_auto_solve_puzzle(self, cfg, frame, generation) -> bool:
        """If any auto-solvable Human Check variant is in `frame`, solve it and
        send a confirmation screenshot. Returns True iff one fired (so the
        caller can skip everything else for this detection).

        An unexpected exception anywhere in a solver (image-processing/
        classification code, not yet battle-tested against every real-world
        capture) is caught here rather than left to propagate - otherwise it
        would kill this async attempt entirely, skipping both the retry and
        the failure-notification/poll fallback below it, silently leaving the
        dialog unanswered with no safety net at all."""
        if not cfg.get("auto_solve_puzzle_enabled") or frame is None:
            return False
        try:
            if self._try_apple_count_puzzle(cfg, frame, generation):
                return True
            if self._try_pick_picture_puzzle(cfg, frame, generation):
                return True
            if self._try_pick_odd_puzzle(cfg, frame, generation):
                return True
            if self._try_arithmetic_puzzle(cfg, frame, generation):
                return True
        except Exception:
            import traceback
            traceback.print_exc()
        return False

    def _locate_puzzle_anchor(self, frame):
        """Check each of the 4 auto-solvable dialog types' anchors against
        `frame`, in the same order `_try_auto_solve_puzzle` tries them.
        Returns (anchor_pos, text_click_offset, label) for whichever one
        matches, or (None, None, None) if none do - used both to click near
        the dialog before anything else, and to report what the puzzle type
        actually was (e.g. for the failure notification)."""
        if frame is None:
            return None, None, None
        pos = self.apple_count_solver.locate(frame)
        if pos is not None:
            return pos, APPLE_TEXT_CLICK_OFFSET, "apple counting"
        pos = self.pick_picture_solver.locate(frame)
        if pos is not None:
            return pos, PICK_TEXT_CLICK_OFFSET, "pick-the-picture"
        pos = self.pick_odd_solver.locate(frame)
        if pos is not None:
            return pos, ODD_TEXT_CLICK_OFFSET, "pick-the-odd-one-out"
        pos = self.arithmetic_solver.locate(frame)
        if pos is not None:
            return pos, ARITHMETIC_TEXT_CLICK_OFFSET, "arithmetic"
        return None, None, None

    def _notify_auto_solve_failed(self, frame, puzzle_type: str):
        if not self.telegram.enabled:
            return
        self.telegram.send_photo(
            frame,
            f"Auto-solve failed to answer the '{puzzle_type}' Human Check - sending the poll as a fallback.",
        )

    def _capture_and_trigger_lie_detector(self, cfg, generation):
        try:
            frame = self.gm.last_frame
            rect = self.window.get_client_rect_screen()
            anchor_pos, text_offset, puzzle_type = (
                self._locate_puzzle_anchor(frame) if rect is not None else (None, None, None)
            )

            if anchor_pos is None and rect is not None:
                # `self.gm.last_frame` is whatever the GM-banner scan last
                # captured, which can catch one of these 4 dialogs mid-render
                # (box/banner already up, instruction text not yet drawn at
                # all) - confirmed from a real capture ("pick the odd-
                # failed3.jpg") where the anchor matched at 0.996 once
                # settled, yet the live run fell all the way through to the
                # generic "unrecognized dialog" poll with no auto-solve
                # attempted. Give it a moment and recapture once before
                # concluding this isn't one of the 4 known types.
                time.sleep(1.0)
                if generation != self._lie_detector_generation:
                    return
                settled = self._capture_full_game_frame()
                if settled is not None:
                    frame = settled
                anchor_pos, text_offset, puzzle_type = self._locate_puzzle_anchor(frame)

            if anchor_pos is not None:
                # One of the 4 auto-solvable dialog types - click on the
                # dialog itself (near its anchor) FIRST, before anything
                # else. This both skips/completes the instruction text's
                # typewriter animation (confirmed still mid-render on first
                # detection in real captures, e.g. "All pict...") and makes
                # sure something actually lands on the dialog rather than
                # jumping straight to reading a not-yet-settled frame.
                text_x = rect[0] + anchor_pos[0] + text_offset[0]
                text_y = rect[1] + anchor_pos[1] + text_offset[1]
                self.timed._focus_game()
                self.inp.click_at(text_x, text_y)
                time.sleep(1.0)
                if generation != self._lie_detector_generation:
                    return
                settled = self._capture_full_game_frame()
                if settled is not None:
                    frame = settled

                # Give the user visibility into every occurrence (not just
                # ones auto-solve fails on) - send the normal screenshot+poll
                # now, before auto-solve tries to answer it directly.
                if cfg.get("auto_solve_puzzle_enabled") or cfg.get("telegram_alert_enabled"):
                    self.lie_detector.trigger([frame])
                    if generation != self._lie_detector_generation:
                        return
                    if cfg.get("auto_solve_puzzle_enabled"):
                        # About to answer this directly below - close the
                        # poll just sent (it was for visibility only) so it
                        # can't be double-answered, and so it doesn't block
                        # the auto-solver's own click just below
                        # (auto_solve_puzzle_click no-ops while a poll is
                        # still awaiting an answer).
                        self.lie_detector.cancel()

                if self._try_auto_solve_puzzle(cfg, frame, generation):
                    return

                if cfg.get("auto_solve_puzzle_enabled"):
                    self._notify_auto_solve_failed(frame, puzzle_type)
                    # The poll sent above was closed right before this attempt
                    # (see the cancel() above) since auto-solve was expected to
                    # answer directly - since it couldn't, re-open a fresh one
                    # so the user actually has a working fallback to answer,
                    # instead of pointing them at an already-closed poll.
                    self.lie_detector.trigger([frame])
                return

            # None of the 4 known anchors matched - could be an older/other
            # dialog type (e.g. the numbered "1st option".."8th option" list)
            # whose options don't all fit on screen without scrolling.
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
            if generation != self._lie_detector_generation:
                return

            if cfg.get("telegram_alert_enabled"):
                self.lie_detector.trigger([frame])
        except Exception:
            import traceback
            traceback.print_exc()
            # Last-resort safety net: something unexpected broke outside the
            # solvers themselves (scrolling, capturing, etc.) - still try to
            # give the user a way to respond via the normal poll, rather than
            # leaving the dialog completely unanswered with nothing sent at
            # all (which would otherwise leave the bot stuck paused on a
            # check no one - human or auto-solver - ever got a chance to
            # answer).
            try:
                if cfg.get("auto_solve_puzzle_enabled") or cfg.get("telegram_alert_enabled"):
                    self.lie_detector.trigger([self.gm.last_frame])
            except Exception:
                traceback.print_exc()
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
                        # poll on its own.
                        if self._gm_detect_streak >= GM_TRIGGER_CONFIRM_TICKS and (
                            cfg.get("auto_solve_puzzle_enabled")
                            # Also required for the plain Telegram-poll path (no auto-solve), so a
                            # resend only happens once the dialog has actually been seen to go
                            # away (even briefly) since the last poll - avoids re-pinging a human
                            # about the same still-open dialog just because they already answered
                            # it while it kept showing. Auto-solve doesn't have that concern - a
                            # multi-step check (e.g. the 2nd Human Check question) can replace one
                            # step with the next without ever registering as absent in between on
                            # slower hardware, which would otherwise leave it permanently stuck
                            # waiting for a "miss" that never comes.
                            or (cfg.get("telegram_alert_enabled") and self._seen_miss_since_trigger)
                        ):
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
