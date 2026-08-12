import tkinter as tk
import tkinter.ttk as ttk
from bot.timed_action_manager import TimedAction

COLUMNS = ("Name", "Key", "Interval (s)", "Pre Delay (s)", "Hold (s)", "Post Delay (s)", "Remaining (s)", "Enabled")

class TimedTab(ttk.Frame):
    def __init__(self, parent, engine, config):
        super().__init__(parent)
        self.engine = engine
        self.config = config
        self._build()
        self._load_actions()
        self._poll()

    def _build(self):
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=5, pady=5)
        tk.Button(btn_frame, text="Add Action", command=self._add_action).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Remove Selected", command=self._remove_action).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Save", command=self._save).pack(side="right", padx=2)

        self._tree = ttk.Treeview(self, columns=COLUMNS, show="headings", selectmode="browse")
        for col in COLUMNS:
            self._tree.heading(col, text=col)
            self._tree.column(col, width=90)
        self._tree.pack(fill="both", expand=True, padx=5, pady=5)
        self._tree.bind("<Double-1>", self._on_double_click)

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)

        tk.Label(self, text="Double-click a row to edit it.", fg="gray").pack(pady=2)

    def _load_actions(self):
        self._tree.delete(*self._tree.get_children())
        for a in self.engine.timed.actions:
            self._tree.insert("", "end", values=(
                a.name, a.key, a.interval, a.pre_delay, a.hold, a.post_delay,
                f"{a.time_remaining():.1f}", "Yes" if a.enabled else "No"
            ))

    def _add_action(self):
        action = TimedAction(name="New Buff", key="", interval=30.0, hold=0.05, post_delay=0.5)
        self.engine.timed.actions.append(action)
        self._load_actions()
        self._save()
        self._edit_action(len(self.engine.timed.actions) - 1)

    def _remove_action(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        if 0 <= idx < len(self.engine.timed.actions):
            self.engine.timed.actions.pop(idx)
        self._load_actions()
        self._save()

    def _on_double_click(self, _event):
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        self._edit_action(idx)

    def _edit_action(self, idx: int):
        if not (0 <= idx < len(self.engine.timed.actions)):
            return
        action = self.engine.timed.actions[idx]

        win = tk.Toplevel(self)
        win.title("Edit Buff")
        win.resizable(False, False)
        win.grab_set()

        enabled_var = tk.BooleanVar(value=action.enabled)
        entries = {}

        fields = [
            ("Name", "name", action.name),
            ("Key", "key", action.key),
            ("Interval (s)", "interval", action.interval),
            ("Pre Delay (s)", "pre_delay", action.pre_delay),
            ("Hold (s)", "hold", action.hold),
            ("Post Delay (s)", "post_delay", action.post_delay),
        ]
        for i, (label, key, value) in enumerate(fields):
            tk.Label(win, text=f"{label}:", anchor="e", width=14).grid(row=i, column=0, padx=5, pady=3, sticky="e")
            e = tk.Entry(win, width=16)
            e.insert(0, str(value))
            e.grid(row=i, column=1, padx=5, pady=3, sticky="w")
            entries[key] = e

        tk.Checkbutton(win, text="Enabled", variable=enabled_var).grid(row=len(fields), column=0, columnspan=2, pady=5)

        def _confirm():
            try:
                action.name = entries["name"].get().strip() or "Buff"
                action.key = entries["key"].get().strip()
                action.interval = float(entries["interval"].get())
                action.pre_delay = float(entries["pre_delay"].get())
                action.hold = float(entries["hold"].get())
                action.post_delay = float(entries["post_delay"].get())
                action.enabled = enabled_var.get()
            except ValueError:
                return
            self._load_actions()
            self._save()
            win.destroy()

        btn_row = tk.Frame(win)
        btn_row.grid(row=len(fields) + 1, column=0, columnspan=2, pady=(0, 8))
        tk.Button(btn_row, text="OK", command=_confirm).pack(side="left", padx=5)
        tk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="left", padx=5)

    def _save(self):
        self.config.set_timed_actions([a.to_dict() for a in self.engine.timed.actions])
        self.config.save()

    def _poll(self):
        actions = self.engine.timed.actions
        for i, item in enumerate(self._tree.get_children()):
            if i >= len(actions):
                continue
            a = actions[i]
            self._tree.set(item, "Remaining (s)", f"{a.time_remaining():.1f}")
            self._tree.set(item, "Enabled", "Yes" if a.enabled else "No")

        self.after(300, self._poll)
