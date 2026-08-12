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

        input_frame = tk.LabelFrame(self, text="Input Method")
        input_frame.pack(fill="x", padx=5, pady=5)
        self._input_var = tk.StringVar(value=cfg.get("input_method", "sendinput"))
        tk.Radiobutton(input_frame, text="SendInput (default, window must be focused)",
                        variable=self._input_var, value="sendinput",
                        command=self._on_input_change).pack(anchor="w", padx=5, pady=2)
        tk.Radiobutton(input_frame, text="PostMessage (background, works unfocused, may not work for all games)",
                        variable=self._input_var, value="postmessage",
                        command=self._on_input_change).pack(anchor="w", padx=5, pady=2)

    def _save(self, key: str, cast, value: str):
        try:
            self.config.set(key, cast(value.strip()))
            self.config.save()
            self.engine._apply_config()
        except (ValueError, TypeError):
            return

    def _on_input_change(self):
        self.config.set("input_method", self._input_var.get())
        self.config.save()
        self.engine.inp.method = self._input_var.get()
