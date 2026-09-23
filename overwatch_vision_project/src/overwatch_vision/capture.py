import ctypes
from ctypes import wintypes
import time

import cv2
import mss
import numpy as np

from overwatch_vision.models import GameFrame


user32 = ctypes.windll.user32


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


EnumWindowsProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HWND,
    wintypes.LPARAM,
)


def _find_window_contains(title_fragment):
    title_fragment = title_fragment.lower()
    matches = []

    @EnumWindowsProc
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value

        if title_fragment in title.lower():
            matches.append(hwnd)
            return False

        return True

    user32.EnumWindows(callback, 0)
    return matches[0] if matches else None


def _client_screen_rect(hwnd):
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None

    top_left = POINT(rect.left, rect.top)
    bottom_right = POINT(rect.right, rect.bottom)

    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        return None

    width = bottom_right.x - top_left.x
    height = bottom_right.y - top_left.y

    if width <= 0 or height <= 0:
        return None

    return {
        "left": top_left.x,
        "top": top_left.y,
        "width": width,
        "height": height,
    }


class OverwatchCapture:
    def __init__(self, config):
        capture_cfg = config["capture"]
        self.window_title = str(capture_cfg.get("window_title_contains", "Overwatch"))
        self.fallback_monitor_index = int(capture_cfg.get("fallback_monitor_index", 1))
        self.sct = mss.mss()
        self.hwnd = None
        self.last_window_lookup = 0.0

    def _capture_rect(self):
        now = time.monotonic()

        if self.hwnd is None or now - self.last_window_lookup > 2.0:
            self.hwnd = _find_window_contains(self.window_title)
            self.last_window_lookup = now

        if self.hwnd:
            rect = _client_screen_rect(self.hwnd)
            if rect:
                return rect

        index = self.fallback_monitor_index
        if index >= len(self.sct.monitors):
            index = 1

        return self.sct.monitors[index]

    def grab(self):
        rect = self._capture_rect()
        raw = np.asarray(self.sct.grab(rect))
        image = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        h, w = image.shape[:2]

        return GameFrame(
            image=image,
            timestamp=time.monotonic(),
            width=w,
            height=h,
        )
