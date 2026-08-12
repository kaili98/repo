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
        self.root.title("MapleStory Bot")
        self.root.resizable(False, False)
        try:
            self.root.iconbitmap(_resource("app.ico"))
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

        root.bind("<F9>", lambda e: self._toggle())

        self._global_hotkey_ok = False
        if _HAS_KEYBOARD:
            try:
                _keyboard.add_hotkey("f9", self._toggle_threadsafe)
                self._global_hotkey_ok = True
            except Exception:
                pass

        if not self._global_hotkey_ok and not _is_admin():
            root.after(500, self._warn_no_global_hotkey)

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _toggle_threadsafe(self):
        self.root.after(0, self._toggle)

    def _toggle(self):
        if self.engine.status.running:
            self.engine.stop()
            return
        self.engine.start()

    def _warn_no_global_hotkey(self):
        messagebox.showinfo(
            "F9 Hotkey",
            "Global F9 hotkey is not active.\n\n"
            "To use F9 from the game window, run the bot as Administrator.\n\n"
            "For now, click Start or press F9 while this window is focused."
        )

    def _on_close(self):
        self.engine.stop()
        if _HAS_KEYBOARD and self._global_hotkey_ok:
            try:
                _keyboard.remove_hotkey("f9")
            except Exception:
                pass
        self.root.destroy()
