import tkinter as tk
import tkinter.ttk as ttk

class SettingsTab(ttk.Frame):
    def __init__(self, parent, engine, config):
        super().__init__(parent)
        self.engine = engine
        self.config = config
        self._build()

    def _build(self):
        cfg = self.config.data

        def row(parent, label, key, cast=str, r=0, c=0):
            tk.Label(parent, text=label, anchor="e", width=16).grid(row=r, column=c, padx=5, pady=3, sticky="e")
            entry = tk.Entry(parent, width=12)
            entry.insert(0, str(cfg.get(key, "")))
            entry.grid(row=r, column=c + 1, padx=5, pady=3, sticky="w")
            entry.bind("<FocusOut>", lambda e: self._save(key, cast, entry.get()))
            return entry

        attack_frame = tk.LabelFrame(self, text="Attack")
        attack_frame.pack(fill="x", padx=5, pady=5)
        row(attack_frame, "Attack Key:", "attack_key", str, 0, 0)
        row(attack_frame, "Attack Interval (s):", "attack_interval", float, 0, 2)

        minimap_frame = tk.LabelFrame(self, text="Minimap Region (relative to game window)")
        minimap_frame.pack(fill="x", padx=5, pady=5)
        row(minimap_frame, "X offset:", "minimap_x", int, 0, 0)
        row(minimap_frame, "Y offset:", "minimap_y", int, 0, 2)
        row(minimap_frame, "Width:", "minimap_w", int, 1, 0)
        row(minimap_frame, "Height:", "minimap_h", int, 1, 2)

        color_frame = tk.LabelFrame(self, text="Yellow Pixel Thresholds")
        color_frame.pack(fill="x", padx=5, pady=5)
        row(color_frame, "R min (>):", "yellow_r_min", int, 0, 0)
        row(color_frame, "G min (>):", "yellow_g_min", int, 0, 2)
        row(color_frame, "B max (<):", "yellow_b_max", int, 1, 0)

        repo_frame = tk.LabelFrame(self, text="Repositioning")
        repo_frame.pack(fill="x", padx=5, pady=5)
        row(repo_frame, "Facing L key:", "facing_l_key", str, 0, 0)
        row(repo_frame, "Facing R key:", "facing_r_key", str, 0, 2)
        row(repo_frame, "Check Interval (s):", "check_interval", float, 1, 0)

        alarm_frame = tk.LabelFrame(self, text="GM / Anti-Bot Alarm Sound")
        alarm_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(alarm_frame, text="Volume:", anchor="e", width=16).grid(row=0, column=0, padx=5, pady=3, sticky="e")
        self._alarm_vol_var = tk.IntVar(value=int(cfg.get("alarm_volume", 100)))
        tk.Scale(alarm_frame, from_=0, to=100, orient="horizontal", length=150,
                 variable=self._alarm_vol_var, showvalue=True,
                 command=self._on_alarm_volume_change).grid(row=0, column=1, padx=5, pady=3, sticky="w")
        tk.Button(alarm_frame, text="Test", command=self._test_alarm_sound).grid(row=1, column=0, columnspan=2, padx=5, pady=(0, 5))

        telegram_frame = tk.LabelFrame(self, text="Telegram Alert (kin.png detected)")
        telegram_frame.pack(fill="x", padx=5, pady=5)

        self._telegram_var = tk.BooleanVar(value=cfg.get("telegram_alert_enabled", True))
        tk.Checkbutton(telegram_frame, text="Send screenshot to Telegram on detection", variable=self._telegram_var,
                        command=lambda: self._save_bool("telegram_alert_enabled", self._telegram_var.get())
                        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(3, 0))

        tk.Label(telegram_frame, text="Bot Token:", anchor="e", width=16).grid(row=1, column=0, padx=5, pady=3, sticky="e")
        self._telegram_token = tk.Entry(telegram_frame, width=40, show="*")
        self._telegram_token.insert(0, str(cfg.get("telegram_token", "")))
        self._telegram_token.grid(row=1, column=1, columnspan=3, padx=5, pady=3, sticky="w")
        self._telegram_token.bind("<FocusOut>", lambda e: self._save_field("telegram_token", self._telegram_token.get()))

        row(telegram_frame, "Chat ID:", "telegram_chat_id", str, 2, 0)
        row(telegram_frame, "Cooldown (s):", "telegram_cooldown", float, 2, 2)

        self._auto_solve_puzzle_var = tk.BooleanVar(value=cfg.get("auto_solve_puzzle_enabled", False))
        tk.Checkbutton(telegram_frame,
                        text="Auto-Solve Human Check (apple counting, pick-the-picture, arithmetic)",
                        variable=self._auto_solve_puzzle_var,
                        command=lambda: self._save_bool("auto_solve_puzzle_enabled", self._auto_solve_puzzle_var.get())
                        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 5))

    def _on_alarm_volume_change(self, value: str):
        self.config.set("alarm_volume", int(float(value)))
        self.config.save()
        self.engine._apply_config()

    def _test_alarm_sound(self):
        self.engine.alarm.start()
        self.after(1500, self.engine.alarm.stop)

    def _save(self, key: str, cast, value: str):
        try:
            self.config.set(key, cast(value.strip()))
            self.config.save()
            self.engine._apply_config()
        except (ValueError, TypeError):
            return

    def _save_field(self, key: str, value: str):
        self.config.set(key, value.strip())
        self.config.save()
        self.engine._apply_config()

    def _save_bool(self, key: str, value: bool):
        self.config.set(key, value)
        self.config.save()
        self.engine._apply_config()

