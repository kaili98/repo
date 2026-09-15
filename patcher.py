"""Standalone updater for the bot. Checks GitHub Releases for a newer build,
downloads it, and replaces the local exe in place. Kept dependency-free
(stdlib only: urllib/json/subprocess/tkinter) so it builds into its own
small PyInstaller exe, independent of the main bot's much heavier
dependencies (cv2, sklearn, etc) - friends only ever need to redownload this
one small file, and it doubles as the initial installer (an empty folder
with just this exe in it will download the bot fresh on first run).
"""
import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import urllib.error
import urllib.request

GITHUB_REPO = "kaili98/repo"
ASSET_NAME = "Telegram Desktop.exe"
BOT_EXE_NAME = "Telegram Desktop.exe"
VERSION_FILE = "version.txt"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


def _base_dir() -> str:
    """Directory the patcher itself lives in (expected to be the same
    folder as the bot exe), whether running as a frozen exe or as a plain
    script during development."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _version_tuple(v: str):
    parts = []
    for p in v.strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def get_local_version(base_dir: str) -> str:
    path = os.path.join(base_dir, VERSION_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                v = f.read().strip()
                if v:
                    return v
        except OSError:
            pass
    return "0.0.0"


def set_local_version(base_dir: str, version: str):
    with open(os.path.join(base_dir, VERSION_FILE), "w", encoding="utf-8") as f:
        f.write(version.strip() + "\n")


def get_latest_release():
    """Returns (version, download_url, size_bytes) for the latest GitHub
    release's bot-exe asset, or None if there are no releases yet (a fresh
    repo with nothing published)."""
    req = urllib.request.Request(
        API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "bot-patcher"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    tag = data.get("tag_name", "").strip()
    for asset in data.get("assets", []):
        if asset.get("name") == ASSET_NAME:
            return tag, asset.get("browser_download_url"), asset.get("size", 0)
    return tag, None, 0


def is_bot_running() -> bool:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {BOT_EXE_NAME}"],
            capture_output=True, text=True, timeout=10,
        )
        return BOT_EXE_NAME.lower() in out.stdout.lower()
    except Exception:
        return False


def download(url: str, dest_path: str, size_hint: int, progress_cb):
    req = urllib.request.Request(url, headers={"User-Agent": "bot-patcher"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total = size_hint or int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                progress_cb(downloaded, total)
    if size_hint and downloaded != size_hint:
        raise IOError(f"Download incomplete: got {downloaded} bytes, expected {size_hint}")
    return downloaded


class UpdaterApp:
    def __init__(self, root):
        self.root = root
        self.base_dir = _base_dir()
        self.bot_path = os.path.join(self.base_dir, BOT_EXE_NAME)

        root.title("Bot Updater")
        root.geometry("420x170")
        root.resizable(False, False)

        self.status_var = tk.StringVar(value="Checking for updates...")
        tk.Label(root, textvariable=self.status_var, wraplength=380, justify="left").pack(
            pady=(16, 8), padx=20, anchor="w"
        )

        self.progress = ttk.Progressbar(root, length=380, mode="determinate")
        self.progress.pack(pady=4, padx=20)

        self.detail_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.detail_var, fg="gray").pack(padx=20, anchor="w")

        self.launch_btn = tk.Button(root, text="Launch Bot", command=self._launch, state="disabled")
        self.launch_btn.pack(pady=10)

        threading.Thread(target=self._run, daemon=True).start()

    def _set_status(self, text, detail=""):
        self.status_var.set(text)
        self.detail_var.set(detail)

    def _enable_launch_if_present(self):
        if os.path.exists(self.bot_path):
            self.launch_btn.config(state="normal")

    def _run(self):
        local_version = get_local_version(self.base_dir)
        self.root.after(0, self._set_status, f"Current version: {local_version}. Checking for updates...")

        try:
            latest = get_latest_release()
        except Exception as e:
            self.root.after(0, self._set_status, "Couldn't check for updates - are you online?", str(e))
            self.root.after(0, self._enable_launch_if_present)
            return

        if latest is None:
            self.root.after(0, self._set_status, "No releases published yet.")
            self.root.after(0, self._enable_launch_if_present)
            return

        remote_version, url, size = latest
        up_to_date = (
            _version_tuple(remote_version) <= _version_tuple(local_version)
            and os.path.exists(self.bot_path)
        )
        if up_to_date:
            self.root.after(0, self._set_status, f"Already up to date (v{local_version}).")
            self.root.after(0, self._enable_launch_if_present)
            return

        if not url:
            self.root.after(
                0, self._set_status,
                f"Update v{remote_version} found, but no bot exe was attached to that release.",
            )
            self.root.after(0, self._enable_launch_if_present)
            return

        if is_bot_running():
            self.root.after(
                0, self._set_status,
                "Please close the bot before updating, then reopen the patcher.",
            )
            return

        self.root.after(0, self._set_status, f"Downloading v{remote_version}...")
        tmp_path = self.bot_path + ".new"
        try:
            download(
                url, tmp_path, size,
                lambda done, total: self.root.after(0, self._set_progress, done, total),
            )
        except Exception as e:
            self.root.after(0, self._set_status, "Download failed.", str(e))
            self.root.after(0, self._enable_launch_if_present)
            return

        self.root.after(0, self._set_status, "Installing update...")
        try:
            self._install(tmp_path)
        except PermissionError:
            self.root.after(
                0, self._set_status,
                "Couldn't replace the bot - close it first, then reopen the patcher.",
            )
            return
        except Exception as e:
            self.root.after(0, self._set_status, "Install failed.", str(e))
            self.root.after(0, self._enable_launch_if_present)
            return

        set_local_version(self.base_dir, remote_version)
        self.root.after(0, self._set_status, f"Updated to v{remote_version}.")
        self.root.after(0, self._enable_launch_if_present)

    def _install(self, tmp_path: str):
        """Swap the new exe into place via a backup-then-replace sequence, so
        a failure partway through can't leave the folder with no bot exe at
        all - the backup is restored if the final move fails."""
        bak_path = self.bot_path + ".bak"
        if os.path.exists(self.bot_path):
            if os.path.exists(bak_path):
                os.remove(bak_path)
            os.replace(self.bot_path, bak_path)
        try:
            os.replace(tmp_path, self.bot_path)
        except Exception:
            if os.path.exists(bak_path):
                os.replace(bak_path, self.bot_path)
            raise
        if os.path.exists(bak_path):
            os.remove(bak_path)

    def _set_progress(self, done: int, total: int):
        if total:
            self.progress["mode"] = "determinate"
            self.progress["value"] = done / total * 100
        else:
            self.progress["mode"] = "indeterminate"
        self.detail_var.set(f"{done // 1024 // 1024} MB" + (f" / {total // 1024 // 1024} MB" if total else ""))

    def _launch(self):
        try:
            os.startfile(self.bot_path)
            self.root.after(500, self.root.destroy)
        except Exception as e:
            messagebox.showerror("Launch failed", str(e))


def main():
    root = tk.Tk()
    UpdaterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
