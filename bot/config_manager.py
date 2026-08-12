import json
import os
import sys
from typing import List, Optional

DEFAULT_CONFIG = {
    "background_mode": False,
    "attack_key": "f10",
    "attack_interval": 0.15,
    "reposition_enabled": True,
    "calib_x": None,
    "minimap_x": 5,
    "minimap_y": 68,
    "minimap_w": 150,
    "minimap_h": 190,
    "yellow_r_min": 200,
    "yellow_g_min": 200,
    "yellow_b_max": 80,
    "l_offset": 5,
    "r_offset": 5,
    "tolerance": 2,
    "facing_l_key": "left",
    "facing_r_key": "right",
    "preferred_facing": "right",
    "snap_facing": True,
    "simple_reposition": True,
    "aux_move_key": "a",
    "hold_aux_key": False,
    "idle_move_enabled": True,
    "idle_move_every": 30.0,
    "idle_move_hold": 0.3,
    "timed_actions": [],
    "input_method": "sendinput",
    "check_interval": 0.5,
    "gm_alarm_enabled": True,
}

def _config_path():
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "config.json")
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")

CONFIG_FILE = _config_path()

class ConfigManager:
    def __init__(self):
        self.data = dict(DEFAULT_CONFIG)
        self.load()

    def load(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    saved = json.load(f)
                for key in DEFAULT_CONFIG:
                    if key in saved:
                        self.data[key] = saved[key]
            except Exception:
                return None
        return None

    def save(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            return None

    def get(self, key: str):
        return self.data.get(key, DEFAULT_CONFIG.get(key))

    def set(self, key: str, value):
        self.data[key] = value

    def get_timed_actions(self) -> List[dict]:
        return self.data.get("timed_actions", [])

    def set_timed_actions(self, actions: List[dict]):
        self.data["timed_actions"] = actions
