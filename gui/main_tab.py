import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from typing import Callable
from PIL import Image, ImageTk
import numpy as np

class MainTab(ttk.Frame):
    def __init__(self, parent, engine, config, on_toggle: Callable):
        super().__init__(parent)
        self.engine = engine
        self.config = config
        self.on_toggle = on_toggle
        self._preview_win = None
        self._windows = []
        self._filtered_windows = []
        self._build()
        self._poll()

    def _build(self):
        data = self.config.data

        # ---- Status ----
        status_frame = tk.LabelFrame(self, text="Status")
        status_frame.pack(fill="x", padx=5, pady=5)

        self._lbl_status = tk.Label(status_frame, text="Status: Stopped", font=("Segoe UI", 10, "bold"), anchor="w")
        self._lbl_status.pack(anchor="w", padx=5, pady=(5, 0))
        self._lbl_state = tk.Label(status_frame, text="Idle", anchor="w")
        self._lbl_state.pack(anchor="w", padx=5)
        self._lbl_charpos = tk.Label(status_frame, text="Char Pos: N/A", anchor="w")
        self._lbl_charpos.pack(anchor="w", padx=5)
        self._lbl_idle = tk.Label(status_frame, text="", anchor="w", fg="gray")
        self._lbl_idle.pack(anchor="w", padx=5)
        self._lbl_gm = tk.Label(status_frame, text="", anchor="w", fg="red", font=("Segoe UI", 9, "bold"))
        self._lbl_gm.pack(anchor="w", padx=5, pady=(0, 5))

        # ---- Controls ----
        controls_frame = tk.LabelFrame(self, text="Controls")
        controls_frame.pack(fill="x", padx=5, pady=5)

        self._btn_start = tk.Button(controls_frame, text="Start (F9)", width=20, command=self.on_toggle)
        self._btn_start.pack(padx=5, pady=(5, 2))
        tk.Button(controls_frame, text="Recalibrate Position", width=20, command=self._calibrate).pack(padx=5, pady=2)
        tk.Button(controls_frame, text="Screenshot Char Region", width=20, command=self._show_preview).pack(padx=5, pady=(2, 5))

        self._mode_var = tk.StringVar(value="background" if data.get("background_mode") else "normal")
        mode_frame = tk.Frame(controls_frame)
        mode_frame.pack(pady=(0, 5))
        tk.Radiobutton(mode_frame, text="Normal", variable=self._mode_var, value="normal",
                        command=self._on_mode_change).pack(side="left")
        tk.Radiobutton(mode_frame, text="Background", variable=self._mode_var, value="background",
                        command=self._on_mode_change).pack(side="left")

        self._gm_alarm_var = tk.BooleanVar(value=data.get("gm_alarm_enabled", True))
        tk.Checkbutton(controls_frame, text="GM / Anti-Bot Alarm (kin.png)", variable=self._gm_alarm_var,
                        command=lambda: self._save_bool("gm_alarm_enabled", self._gm_alarm_var.get())
                        ).pack(pady=(0, 5))

        # ---- Target ----
        target_frame = tk.LabelFrame(self, text="Target")
        target_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(target_frame, text="Filter:").pack(side="left", padx=(5, 2))
        self._filter_var = tk.StringVar(value=data.get("window_title_filter", ""))
        self._filter_var.trace_add("write", lambda *a: self._apply_filter())
        tk.Entry(target_frame, textvariable=self._filter_var, width=15).pack(side="left", padx=2)

        self._window_var = tk.StringVar()
        self._window_cb = ttk.Combobox(target_frame, textvariable=self._window_var, state="readonly", width=35)
        self._window_cb.pack(side="left", expand=True, fill="x", padx=2)
        self._window_cb.bind("<<ComboboxSelected>>", self._on_window_select)

        tk.Button(target_frame, text="Refresh", command=self._refresh_windows).pack(side="left", padx=(2, 5))

        # ---- Main Attack ----
        attack_frame = tk.LabelFrame(self, text="Main Attack")
        attack_frame.pack(fill="x", padx=5, pady=5)
        tk.Label(attack_frame, text="Key:").pack(side="left", padx=(5, 2), pady=5)
        self._atk_key = tk.Entry(attack_frame, width=8)
        self._atk_key.insert(0, str(data.get("attack_key", "")))
        self._atk_key.pack(side="left", padx=2, pady=5)
        self._atk_key.bind("<FocusOut>", lambda e: self._save_field("attack_key", self._atk_key.get()))

        # ---- Repositioning ----
        repo_frame = tk.LabelFrame(self, text="Repositioning")
        repo_frame.pack(fill="x", padx=5, pady=5)
        self._repose_var = tk.BooleanVar(value=data.get("reposition_enabled", True))
        tk.Checkbutton(repo_frame, text="Enable Reposition", variable=self._repose_var,
                        command=self._on_repose_toggle).pack(side="left", padx=5, pady=5)
        self._lbl_calib = tk.Label(repo_frame, text=f"Calib X: {data.get('calib_x')}")
        self._lbl_calib.pack(side="left", padx=(15, 5), pady=5)

        # ---- Repo Control ----
        repo_ctrl = tk.LabelFrame(self, text="Repo Control")
        repo_ctrl.pack(fill="x", padx=5, pady=5)

        offsets_row = tk.Frame(repo_ctrl)
        offsets_row.pack(fill="x", padx=5, pady=(5, 2))

        def _int_entry(label, key, default):
            tk.Label(offsets_row, text=label).pack(side="left", padx=(0, 3))
            e = tk.Entry(offsets_row, width=5)
            e.insert(0, str(data.get(key, default)))
            e.pack(side="left", padx=(0, 10))
            e.bind("<FocusOut>", lambda ev: self._save_int(key, e.get()))
            return e

        _int_entry("L Off:", "l_offset", 5)
        _int_entry("R Off:", "r_offset", 5)
        _int_entry("Tol:", "tolerance", 2)

        facing_row = tk.Frame(repo_ctrl)
        facing_row.pack(fill="x", padx=5, pady=(2, 5))
        tk.Label(facing_row, text="Facing:").pack(side="left", padx=(0, 5))
        preferred = data.get("preferred_facing", "right")
        self._facing_l_btn = tk.Button(facing_row, text="L", width=3, command=lambda: self._set_facing("left"),
                                        relief="sunken" if preferred == "left" else "raised")
        self._facing_l_btn.pack(side="left")
        self._facing_r_btn = tk.Button(facing_row, text="R", width=3, command=lambda: self._set_facing("right"),
                                        relief="sunken" if preferred == "right" else "raised")
        self._facing_r_btn.pack(side="left", padx=(3, 10))
        self._lbl_facing = tk.Label(facing_row, text=f"  Cur: {self.engine.repositioner.current_facing}")
        self._lbl_facing.pack(side="left")

        self._simple_var = tk.BooleanVar(value=data.get("simple_reposition", True))
        tk.Checkbutton(repo_ctrl, text="Simple Reposition (Hold Arrow to Target)", variable=self._simple_var,
                        command=lambda: self._save_bool("simple_reposition", self._simple_var.get())
                        ).pack(anchor="w", padx=5, pady=(0, 2))

        aux_row = tk.Frame(repo_ctrl)
        aux_row.pack(fill="x", padx=5, pady=(0, 5))
        tk.Label(aux_row, text="Aux Move Key:").pack(side="left", padx=(0, 3))
        self._aux_key = tk.Entry(aux_row, width=6)
        self._aux_key.insert(0, str(data.get("aux_move_key", "")))
        self._aux_key.pack(side="left", padx=(0, 10))
        self._aux_key.bind("<FocusOut>", lambda e: self._save_field("aux_move_key", self._aux_key.get()))
        self._hold_aux_var = tk.BooleanVar(value=data.get("hold_aux_key", False))
        tk.Checkbutton(aux_row, text="Hold Move + Aux Key", variable=self._hold_aux_var,
                        command=lambda: self._save_bool("hold_aux_key", self._hold_aux_var.get())
                        ).pack(side="left")

        self._idle_var = tk.BooleanVar(value=data.get("idle_move_enabled", True))
        tk.Checkbutton(repo_ctrl, text="Idle Move (hold facing if no movement for 30s)", variable=self._idle_var,
                        command=lambda: self._save_bool("idle_move_enabled", self._idle_var.get())
                        ).pack(anchor="w", padx=5, pady=(0, 5))

        self._refresh_windows()

    # ---- Window targeting ----

    def _refresh_windows(self):
        self._windows = self.engine.window.list_windows()
        self._apply_filter()

    def _apply_filter(self):
        q = self._filter_var.get().lower()
        if q:
            self._filtered_windows = [(h, t) for h, t in self._windows if q in t.lower()]
        else:
            self._filtered_windows = list(self._windows)
        self._window_cb["values"] = [t for _, t in self._filtered_windows]

    def _on_window_select(self, _event):
        idx = self._window_cb.current()
        if not (0 <= idx < len(self._filtered_windows)):
            return
        hwnd, title = self._filtered_windows[idx]
        self.engine.window.set_target(hwnd)
        self.engine.inp.hwnd = hwnd
        self._window_var.set(title)

    # ---- Controls ----

    def _calibrate(self):
        x = self.engine.calibrate()
        if x is None:
            messagebox.showwarning("Calibration", "No yellow pixel found in minimap region.\nCheck the region settings and try again.")
            return
        self._lbl_calib.config(text=f"Calib X: {str(x)}")

    def _show_preview(self):
        if not self.engine.window.is_valid():
            messagebox.showwarning("Preview", "No target window selected or capture failed.")
            return

        if self._preview_win and self._preview_win.winfo_exists():
            self._preview_win.destroy()

        win = tk.Toplevel(self)
        win.title("Char Region Preview  (auto-refreshes)")
        win.resizable(False, False)
        self._preview_win = win

        self._preview_img_lbl = tk.Label(win)
        self._preview_img_lbl.pack()
        self._preview_info_lbl = tk.Label(win, text="", font=("Courier", 9))
        self._preview_info_lbl.pack(pady=2)
        tk.Label(
            win,
            text="Red = yellow pixels detected  |  Green line = measured X position\n"
                 "Adjust minimap region in Settings until the green line tracks the character dot.",
            fg="gray"
        ).pack(pady=4)

        self._preview_refresh(win)

    def _preview_refresh(self, win):
        if not win.winfo_exists():
            return

        img_arr, detected_x = self.engine.detector.capture_annotated_preview()

        if img_arr is not None:
            rgb = img_arr[:, :, :3][:, :, ::-1]
            pil_img = Image.fromarray(rgb.astype(np.uint8))
            scale = max(2, 400 // pil_img.width)
            pil_img = pil_img.resize((pil_img.width * scale, pil_img.height * scale), Image.NEAREST)
            tk_img = ImageTk.PhotoImage(pil_img)
            self._preview_img_lbl.config(image=tk_img)
            self._preview_img_lbl.image = tk_img

            calib = self.engine.repositioner.calib_x
            info = f"Detected X: {detected_x}   Calib X: {calib}"
            if detected_x is not None and calib is not None:
                diff = detected_x - calib
                info += f"   Diff: {diff:+d}"
            self._preview_info_lbl.config(text=info)
        else:
            self._preview_info_lbl.config(text="Capture failed — is the game window selected?")

        win.after(300, lambda: self._preview_refresh(win))

    def _on_mode_change(self):
        bg = self._mode_var.get() == "background"
        self.config.set("background_mode", bg)
        self.config.save()

    def _on_repose_toggle(self):
        self._save_bool("reposition_enabled", self._repose_var.get())
        self.engine.repositioner.enabled = self._repose_var.get()

    def _set_facing(self, direction: str):
        self.engine.repositioner.preferred_facing = direction
        self.engine.repositioner.current_facing = direction
        self._lbl_facing.config(text=f"  Cur: {direction}")
        if direction == "left":
            self._facing_l_btn.config(relief="sunken")
            self._facing_r_btn.config(relief="raised")
        else:
            self._facing_l_btn.config(relief="raised")
            self._facing_r_btn.config(relief="sunken")
        self._save_field("preferred_facing", direction)

    def _save_field(self, key: str, value: str):
        self.config.set(key, value.strip())
        self.config.save()

    def _save_int(self, key: str, value: str):
        try:
            self.config.set(key, int(value))
            self.config.save()
        except ValueError:
            return

    def _save_float(self, key: str, value: str):
        try:
            self.config.set(key, float(value))
            self.config.save()
        except ValueError:
            return

    def _save_bool(self, key: str, value: bool):
        self.config.set(key, value)
        self.config.save()

    def _poll(self):
        s = self.engine.status

        if s.running:
            self._lbl_status.config(text="Status: Running")
            self._btn_start.config(text="Stop (F9)")
        else:
            self._lbl_status.config(text="Status: Stopped")
            self._btn_start.config(text="Start (F9)")

        self._lbl_state.config(text=s.state)

        pos_text = str(s.char_x) if s.char_x is not None else "N/A"
        if s.calib_x is not None:
            pos_text += f"  (home: {s.calib_x})"
        self._lbl_charpos.config(text=pos_text)
        self._lbl_facing.config(text=f"  Cur: {s.facing}")

        if s.calib_x is not None:
            self._lbl_calib.config(text=f"Calib X: {str(s.calib_x)}")

        countdown = self.engine.idle_countdown()
        if countdown is None:
            self._lbl_idle.config(text="")
        else:
            self._lbl_idle.config(text=f"Idle move in: {countdown:.0f}s")

        self._lbl_gm.config(text="⚠ GM / ANTI-BOT CHECK DETECTED" if s.gm_detected else "")

        self.after(200, self._poll)

    def update_start_button(self, running: bool):
        self._btn_start.config(text="Stop (F9)" if running else "Start (F9)")
