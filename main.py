import ctypes
import sys
import tkinter as tk
from gui.app import App

def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False

def _relaunch_as_admin():
    params = " ".join(f'"{a}"' for a in sys.argv)
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, None, 1)

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    if not _is_admin():
        _relaunch_as_admin()
        sys.exit()
    main()
