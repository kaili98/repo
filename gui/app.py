import ctypes
import os
import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

def _resource(filename: str) -> str:
    """Return the correct path for bundled resources (works both frozen and in dev)."""
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename)

try:
    import keyboard as _keyboard
    _HAS_KEYBOARD = True
except Exception:
    _HAS_KEYBOARD = False

from bot.config_manager import ConfigManager
from bot.engine import BotEngine
from gui.main_tab import MainTab
from gui.timed_tab import TimedTab
from gui.settings_tab import SettingsTab

def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Telegram Desktop")
        self.root.resizable(False, False)
        for icon_name in ("telegram.ico", "app.ico"):
            try:
                self.root.iconbitmap(_resource(icon_name))
                break
            except Exception:
                pass

        self.config = ConfigManager()
        self.engine = BotEngine(self.config)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self.main_tab = MainTab(notebook, self.engine, self.config, self._toggle)
        self.timed_tab = TimedTab(notebook, self.engine, self.config)
        self.settings_tab = SettingsTab(notebook, self.engine, self.config)

        notebook.add(self.main_tab, text="Main")
        notebook.add(self.timed_tab, text="Timed Actions")
        notebook.add(self.settings_tab, text="Settings")

        root.update_idletasks()
        width = max(200, root.winfo_width() - 60)
        height = root.winfo_height()
        root.geometry(f"{width}x{height}")

        root.bind("<F9>", lambda e: self._toggle())
        root.bind("<F6>", lambda e: self._recalibrate())

        self._global_hotkey_ok = False
        self._hotkey_error = None
        if _HAS_KEYBOARD:
            try:
                _keyboard.add_hotkey("f9", self._toggle_threadsafe)
                _keyboard.add_hotkey("f6", self._recalibrate_threadsafe)
                self._global_hotkey_ok = True
            except Exception as e:
                self._hotkey_error = str(e)
        else:
            self._hotkey_error = "The 'keyboard' package is not installed."

        if not self._global_hotkey_ok:
            root.after(500, self._warn_no_global_hotkey)

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _toggle_threadsafe(self):
        self.root.after(0, self._toggle)

    def _toggle(self):
        if self.engine.status.running:
            self.engine.stop()
            return
        self.engine.start()

    def _recalibrate_threadsafe(self):
        self.root.after(0, self._recalibrate)

    def _recalibrate(self):
        self.main_tab._calibrate()

    def _warn_no_global_hotkey(self):
        detail = f"\n\nDetails: {self._hotkey_error}" if self._hotkey_error else ""
        admin_hint = "" if _is_admin() else "\n\nTry running the bot as Administrator."
        messagebox.showinfo(
            "Global Hotkeys",
            "Global F9 (Start/Stop) and F6 (Recalibrate) hotkeys are not active.\n\n"
            "They still work while this window is focused, but not while the game "
            "window has focus.\n\n"
            "If another program (e.g. NVIDIA overlay, OBS, Discord, Xbox Game Bar) is "
            "also bound to F9 or F6, it may be intercepting the key first — try "
            "changing or disabling that program's hotkey."
            + admin_hint + detail
        )

    def _on_close(self):
        self.engine.stop()
        if _HAS_KEYBOARD and self._global_hotkey_ok:
            try:
                _keyboard.remove_hotkey("f9")
                _keyboard.remove_hotkey("f6")
            except Exception:
                pass
        self.root.destroy()
