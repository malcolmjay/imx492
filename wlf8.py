#!/usr/bin/env python3
import atexit
import ctypes
import json
import os
import queue
import signal
import shutil
import subprocess
import struct
import tempfile
import textwrap
import time
import threading
import sys
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import glob as _glob_module
import cv2
import numpy as np

# --- Early splash screen: show a loading indicator immediately so the user
#     sees feedback within ~100ms instead of a black screen for several seconds
#     while the camera hardware initialises.
cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.moveWindow("Camera", 0, 0)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
_splash = np.zeros((480, 800, 3), dtype=np.uint8)
cv2.putText(_splash, "Starting...", (330, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 1, cv2.LINE_AA)
cv2.imshow("Camera", cv2.flip(_splash, -1))
cv2.waitKey(1)
del _splash


def _update_splash(msg):
    """Update the splash screen with a progress message."""
    s = np.zeros((480, 800, 3), dtype=np.uint8)
    cv2.putText(s, msg, (300, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.imshow("Camera", cv2.flip(s, -1))
    cv2.waitKey(1)


# ---- TrueType font rendering (Pillow) ----
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_FONT_PATHS = [
    "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
    "/usr/share/fonts/truetype/inter/Inter-Medium.ttf",
    "/usr/share/fonts/truetype/ibm-plex/IBMPlexSans-Regular.ttf",
    "/usr/share/fonts/truetype/ibm-plex/IBMPlexSans-Medium.ttf",
    "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]
_MONO_FONT_PATHS = [
    "/usr/share/fonts/truetype/robotomono/RobotoMono-Regular.ttf",
    "/usr/share/fonts/truetype/ibm-plex/IBMPlexMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
]
_pil_font_cache = {}


def _load_pil_font(size, mono=False):
    """Load and cache a TrueType font at the given pixel size."""
    key = (size, mono)
    cached = _pil_font_cache.get(key)
    if cached is not None:
        return cached
    if not _PIL_AVAILABLE:
        return None
    paths = _MONO_FONT_PATHS if mono else _FONT_PATHS
    for p in paths:
        try:
            font = ImageFont.truetype(p, size)
            _pil_font_cache[key] = font
            return font
        except Exception:
            continue
    try:
        font = ImageFont.load_default(size=size)
        _pil_font_cache[key] = font
        return font
    except Exception:
        pass
    return None


_ui_measure_cache = {}
_UI_MEASURE_CACHE_MAX = 128
_ui_measure_dummy = None


def _ui_measure_text(text, font_size):
    """Measure text using PIL TrueType font if available, else cv2. Returns (w, h)."""
    global _ui_measure_dummy
    key = (text, font_size)
    cached = _ui_measure_cache.get(key)
    if cached is not None:
        return cached
    if _PIL_AVAILABLE:
        font = _load_pil_font(font_size)
        if font is not None:
            if _ui_measure_dummy is None:
                _ui_measure_dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            bbox = _ui_measure_dummy.textbbox((0, 0), text, font=font)
            result = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            if len(_ui_measure_cache) >= _UI_MEASURE_CACHE_MAX:
                _ui_measure_cache.clear()
            _ui_measure_cache[key] = result
            return result
    scale = font_size / 26.0
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    result = (w, h + base)
    if len(_ui_measure_cache) >= _UI_MEASURE_CACHE_MAX:
        _ui_measure_cache.clear()
    _ui_measure_cache[key] = result
    return result


def _ui_draw_text(img, text, x, y, font_size, color_bgr, outline_bgr=None):
    """Draw text at top-left (x, y) using PIL TrueType font if available, else cv2.

    Optimized: converts only the text bounding-box ROI instead of the full frame.
    """
    if _PIL_AVAILABLE:
        font = _load_pil_font(font_size)
        if font is not None:
            tw, th = _ui_measure_text(text, font_size)
            stroke_pad = 2 if outline_bgr is not None else 0
            # Compute ROI with padding for stroke and descent
            roi_x1 = max(0, x - stroke_pad)
            roi_y1 = max(0, y - stroke_pad)
            roi_x2 = min(img.shape[1], x + tw + stroke_pad * 2 + 4)
            roi_y2 = min(img.shape[0], y + th + stroke_pad * 2 + 4)
            if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
                return
            roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
            pil_roi = Image.fromarray(roi[:, :, ::-1])
            draw = ImageDraw.Draw(pil_roi)
            bbox = draw.textbbox((0, 0), text, font=font)
            adj_x = (x - roi_x1) - bbox[0]
            adj_y = (y - roi_y1) - bbox[1]
            r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
            kwargs = {}
            if outline_bgr is not None:
                sr, sg, sb = int(outline_bgr[2]), int(outline_bgr[1]), int(outline_bgr[0])
                kwargs["stroke_width"] = 1
                kwargs["stroke_fill"] = (sr, sg, sb)
            draw.text((adj_x, adj_y), text, fill=(r, g, b), font=font, **kwargs)
            img[roi_y1:roi_y2, roi_x1:roi_x2] = np.array(pil_roi)[:, :, ::-1]
            return
    scale = font_size / 26.0
    thickness = 1
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2_y = y + th
    if outline_bgr is not None:
        cv2.putText(img, text, (x, cv2_y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    outline_bgr, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, cv2_y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color_bgr, thickness, cv2.LINE_AA)


def _ui_draw_text_batch(img, items, font_size, outline_bgr=None):
    """Draw multiple text items in one PIL pass using a tight bounding-box ROI.

    items is a list of (text, x, y, color_bgr) tuples.
    Converts only the region covering all text items, not the full frame.
    """
    if not items:
        return
    if _PIL_AVAILABLE:
        font = _load_pil_font(font_size)
        if font is not None:
            stroke_pad = 2 if outline_bgr is not None else 0
            # Compute bounding box covering all items
            roi_x1, roi_y1 = img.shape[1], img.shape[0]
            roi_x2, roi_y2 = 0, 0
            measured = []
            for text, x, y, color_bgr in items:
                tw, th = _ui_measure_text(text, font_size)
                ix1 = max(0, x - stroke_pad)
                iy1 = max(0, y - stroke_pad)
                ix2 = min(img.shape[1], x + tw + stroke_pad * 2 + 4)
                iy2 = min(img.shape[0], y + th + stroke_pad * 2 + 4)
                roi_x1 = min(roi_x1, ix1)
                roi_y1 = min(roi_y1, iy1)
                roi_x2 = max(roi_x2, ix2)
                roi_y2 = max(roi_y2, iy2)
                measured.append((text, x, y, color_bgr, tw, th))
            if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
                return
            roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
            pil_roi = Image.fromarray(roi[:, :, ::-1])
            draw = ImageDraw.Draw(pil_roi)
            for text, x, y, color_bgr, tw, th in measured:
                bbox = draw.textbbox((0, 0), text, font=font)
                adj_x = (x - roi_x1) - bbox[0]
                adj_y = (y - roi_y1) - bbox[1]
                r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
                kwargs = {}
                if outline_bgr is not None:
                    sr, sg, sb = int(outline_bgr[2]), int(outline_bgr[1]), int(outline_bgr[0])
                    kwargs["stroke_width"] = 1
                    kwargs["stroke_fill"] = (sr, sg, sb)
                draw.text((adj_x, adj_y), text, fill=(r, g, b), font=font, **kwargs)
            img[roi_y1:roi_y2, roi_x1:roi_x2] = np.array(pil_roi)[:, :, ::-1]
            return
    # Fallback: call individual draw
    for text, x, y, color_bgr in items:
        _ui_draw_text(img, text, x, y, font_size, color_bgr, outline_bgr=outline_bgr)


# ---- Performance caches ----
_IP_CACHE = {'ts': 0.0, 'ip': 'Unavailable'}
_ICON_BTN_CACHE = {}
_ICON_BTN_CACHE_MAX = 64

CONFIG_PATH = Path(__file__).with_name("camera_settings.json")
_SETTINGS_LOCK = threading.Lock()


def _load_persistent_settings():
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            if isinstance(data, dict):
                return data
    except Exception as exc:
        print("Settings load error:", exc)
    return {}


def _bounded_int(value, default, lower, upper):
    try:
        value = int(value)
    except Exception:
        return default
    return max(lower, min(upper, value))


def _bool_value(value, default=False):
    return bool(value) if isinstance(value, (bool, int)) else default


def _save_persistent_settings():
    g = globals()
    payload = {
        "wifi_enabled": bool(g.get("wifi_enabled", False)),
        "flip_mode": int(g.get("flip_mode", 0)),
        "rangefinder_assist_enabled": bool(g.get("rangefinder_assist_enabled", False)),
        "focus_peaking_enabled": bool(g.get("focus_peaking_enabled", True)),
        "focus_mode_idx": int(g.get("_focus_mode_idx", 0)),
        "ui_minimal_mode": bool(g.get("_ui_minimal_mode", False)),
        "aspect_idx": int(g.get("_aspect_idx", 0)),
        "film_idx": int(g.get("_film_idx", 0)),
        "iso_idx": int(g.get("_iso_idx", 0)),
        "shutter_idx": int(g.get("current_shutter_idx", 0)),
        "exp_comp_idx": int(g.get("_exp_comp_idx", 2)),
        "grid_overlay_idx": int(g.get("_grid_overlay_idx", 0)),
        "video_resolution_idx": int(g.get("_video_resolution_idx", 0)),
    }
    try:
        with _SETTINGS_LOCK:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = CONFIG_PATH.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            tmp_path.replace(CONFIG_PATH)
    except Exception as exc:
        print("Settings save error:", exc)


_PERSISTED_SETTINGS = _load_persistent_settings()

# ---- Debounced settings save (reduces SD card writes on rapid button presses) ----
_SETTINGS_SAVE_DELAY = 1.0  # seconds to wait before writing
_settings_save_timer = None
_settings_save_timer_lock = threading.Lock()


def _debounced_save_settings():
    """Schedule a settings save after a short delay, coalescing rapid changes."""
    global _settings_save_timer
    with _settings_save_timer_lock:
        if _settings_save_timer is not None:
            _settings_save_timer.cancel()
        _settings_save_timer = threading.Timer(_SETTINGS_SAVE_DELAY, _save_persistent_settings)
        _settings_save_timer.daemon = True
        _settings_save_timer.start()

# gpiozero and picamera2 are imported lazily at first use (~200-400ms each)
# to avoid delaying the splash screen.  See camera setup and button wiring
# sections below.

# -------------------- Optional battery (X1200) --------------------
user_exit = False
try:
    import smbus2
    I2C_AVAILABLE = True
except Exception:
    I2C_AVAILABLE = False

BATTERY_UPDATE_INTERVAL = 30
FG_ADDR = 0x36
batt_percent_cached = None
batt_voltage_cached = None
batt_cell_voltage_cached = None
batt_pack_voltage_cached = None
_batt_lock = threading.Lock()
_batt_voltage_samples = deque(maxlen=5)
_batt_last_good_ts = 0.0
_batt_fail_start_ts = 0.0
_batt_fail_count = 0
_batt_low_voltage_start = None
_batt_low_voltage_samples = 0

LOW_VOLT_PACK_THRESHOLD = 6.50
LOW_VOLT_SAMPLE_COUNT = 3
LOW_VOLT_HOLD_S = 10.0
BATT_CLEAR_FAIL_S = 60.0
BATT_VOLT_MIN = 3.0
BATT_VOLT_MAX = 4.35

shutdown_initiated = False
_shutdown_lock = threading.Lock()

def safe_shutdown(reason):
    global shutdown_initiated
    with _shutdown_lock:
        if shutdown_initiated:
            return
        shutdown_initiated = True
    try:
        print(f"SAFE SHUTDOWN: {reason}")
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    try:
        subprocess.run(["sync"], check=False)
    except Exception:
        pass
    try:
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
    except Exception:
        pass
    try:
        os._exit(0)
    except Exception:
        pass

def _read_capacity(bus):
    rd = bus.read_word_data(FG_ADDR, 0x04)
    swapped = struct.unpack("<H", struct.pack(">H", rd))[0]
    return swapped / 256.0


def _read_voltage(bus):
    rd = bus.read_word_data(FG_ADDR, 0x02)
    swapped = struct.unpack("<H", struct.pack(">H", rd))[0]
    return swapped * 1.25 / 1000.0 / 16.0

def _update_battery_cache(cap, volt, now=None):
    global batt_percent_cached, batt_voltage_cached, batt_cell_voltage_cached, batt_pack_voltage_cached
    global _batt_last_good_ts, _batt_fail_start_ts, _batt_fail_count
    global _batt_low_voltage_start, _batt_low_voltage_samples
    if now is None:
        now = time.monotonic()
    valid_voltage = None
    if volt is not None and BATT_VOLT_MIN <= float(volt) <= BATT_VOLT_MAX:
        valid_voltage = float(volt)
        _batt_voltage_samples.append(valid_voltage)
        _batt_last_good_ts = now
        _batt_fail_start_ts = 0.0
        _batt_fail_count = 0
    else:
        _batt_fail_count += 1
        if _batt_fail_start_ts == 0.0:
            _batt_fail_start_ts = now
    if valid_voltage is not None and _batt_voltage_samples:
        avg_voltage = float(sum(_batt_voltage_samples)) / float(len(_batt_voltage_samples))
    else:
        avg_voltage = None
    with _batt_lock:
        if cap is not None:
            batt_percent_cached = max(0.0, min(100.0, float(cap)))
        if avg_voltage is not None:
            batt_cell_voltage_cached = avg_voltage
            batt_pack_voltage_cached = avg_voltage * 2.0
            batt_voltage_cached = avg_voltage
    if avg_voltage is None:
        if _batt_last_good_ts and (now - _batt_last_good_ts) >= BATT_CLEAR_FAIL_S:
            with _batt_lock:
                batt_cell_voltage_cached = None
                batt_pack_voltage_cached = None
                batt_voltage_cached = None
                if _batt_fail_count > 0:
                    batt_percent_cached = None
        return
    pack_voltage = avg_voltage * 2.0
    if pack_voltage <= LOW_VOLT_PACK_THRESHOLD:
        _batt_low_voltage_samples += 1
        if _batt_low_voltage_start is None:
            _batt_low_voltage_start = now
        if (_batt_low_voltage_samples >= LOW_VOLT_SAMPLE_COUNT or
                (now - _batt_low_voltage_start) >= LOW_VOLT_HOLD_S):
            safe_shutdown("low battery voltage")
    else:
        _batt_low_voltage_samples = 0
        _batt_low_voltage_start = None

# ---- Disk space monitoring ----
# Defined before _battery_thread so the helper is available when the
# daemon thread is scheduled immediately after start().
_DISK_CHECK_INTERVAL_S = 30
_DISK_WARN_THRESHOLD_MB = 2048      # 2 GB — show persistent "LOW DISK" warning
_DISK_CRITICAL_THRESHOLD_MB = 1024  # 1 GB — block all captures/recordings

_disk_free_mb = None
_disk_lock = threading.Lock()
_disk_level = "ok"   # "ok" | "warning" | "critical"
_disk_last_check = 0.0


def _check_disk_space(force=False):
    """Lightweight disk space check. Samples both the root filesystem (boot
    protection) and the active capture directory (write protection) and uses
    the lower of the two free-space values for level determination.

    Called from the battery thread every 30 s and before captures."""
    global _disk_free_mb, _disk_level, _disk_last_check
    now = time.monotonic()
    if not force and (now - _disk_last_check) < _DISK_CHECK_INTERVAL_S:
        with _disk_lock:
            return _disk_free_mb, _disk_level

    # Root filesystem — protects boot/journals/tmp
    root_free_mb = None
    try:
        root_free_mb = shutil.disk_usage("/").free / (1024 * 1024)
    except Exception:
        pass

    # Active capture target — may be USB or the same SD card.
    # _capture_dir_lock / _capture_dir are defined later during module load,
    # so the first battery-thread call (before the module finishes loading)
    # gracefully falls back to root-only checking.
    capture_free_mb = None
    try:
        with _capture_dir_lock:
            cap_dir = _capture_dir
        capture_free_mb = shutil.disk_usage(cap_dir).free / (1024 * 1024)
    except Exception:
        pass

    # Use the worse (lower) of the two; ignore paths that errored
    candidates = [v for v in (root_free_mb, capture_free_mb) if v is not None]
    free_mb = min(candidates) if candidates else None

    with _disk_lock:
        _disk_free_mb = free_mb
        _disk_last_check = now
        if free_mb is None:
            _disk_level = "ok"  # fail-open: don't block captures on read error
        elif free_mb < _DISK_CRITICAL_THRESHOLD_MB:
            _disk_level = "critical"
        elif free_mb < _DISK_WARN_THRESHOLD_MB:
            _disk_level = "warning"
        else:
            _disk_level = "ok"
        return _disk_free_mb, _disk_level


def _disk_can_capture():
    """Return True if there is enough disk space to allow a new capture."""
    with _disk_lock:
        return _disk_level != "critical"


def _batt_can_record():
    """Return True if battery is high enough to allow video recording."""
    with _batt_lock:
        pct = batt_percent_cached
    if pct is None:
        return True  # fail-open: don't block if sensor unavailable
    return pct >= _BATT_CRITICAL_PCT


def _battery_thread():
    global batt_percent_cached, batt_voltage_cached
    global batt_cell_voltage_cached, batt_pack_voltage_cached
    if not I2C_AVAILABLE:
        return
    try:
        bus = smbus2.SMBus(1)
    except Exception:
        return
    while True:
        now = time.monotonic()
        try:
            cap = _read_capacity(bus)
            volt = _read_voltage(bus)
            _update_battery_cache(cap, volt, now=now)
        except Exception:
            _update_battery_cache(None, None, now=now)
        try:
            _check_disk_space()
        except Exception:
            pass
        time.sleep(BATTERY_UPDATE_INTERVAL)

threading.Thread(target=_battery_thread, daemon=True).start()

# -------------------- Charging mode helpers --------------------

_charge_button_rect = (0, 0, 0, 0)
_trash_button_rect = (0, 0, 0, 0)
_delete_confirm_active = False
_delete_confirm_yes_rect = (0, 0, 0, 0)
_delete_confirm_no_rect = (0, 0, 0, 0)
charge_mode_active = False
_charge_shutdown_triggered = False
_charge_stats_cache = {
    "timestamp": 0.0,
    "lines": ["Charging mode", "Gathering battery data..."],
    "percent": None,
}
_CHARGE_STATS_REFRESH = 5.0

# -------------------- Sleep mode (Option A: "fake sleep") --------------------
# Goal: instant wake (no reboot) by stopping the camera pipeline, blanking the DSI backlight,
# dropping CPU power, disabling Wi‑Fi, and forcing fan off while we idle.
_sleep_button_rect = (0, 0, 0, 0)
sleep_mode_active = False
_sleep_state = {}
_sleep_wake_event = threading.Event()

# Backlight paths are panel-dependent; we discover them at runtime.
_backlight_paths = None
_backlight_prev = {}

# CPU governor/max frequency restore
_cpu_prev_governor = {}
_cpu_prev_max_freq = {}

# Battery indicator render cache — only re-render when pct/charge state changes
_batt_render_cache = None
_batt_render_key = None

# Fan restore (Pi 5 active cooler exposes a cooling_device)
_fan_prev_states = {}
_fan_paths_cache = None

def _discover_backlight_paths():
    """Find writable backlight brightness sysfs nodes.

    Waveshare 4" DSI panels often expose a single backlight device like
    /sys/class/backlight/10-0045/brightness. We prefer known-good nodes first,
    then fall back to any other backlight entries.
    """
    global _backlight_paths
    if _backlight_paths is not None:
        return _backlight_paths

    preferred = [
        "/sys/class/backlight/10-0045/brightness",
        "/sys/waveshare/rpi_backlight/brightness",
        "/sys/class/backlight/rpi_backlight/brightness",
    ]

    paths = []
    try:
        # Add preferred paths if present
        for p in preferred:
            if Path(p).exists():
                paths.append(p)

        # Add any other backlight nodes
        for p in _glob_module.glob("/sys/class/backlight/*/brightness"):
            if p not in paths:
                paths.append(p)
    except Exception:
        pass

    _backlight_paths = paths
    return _backlight_paths

def _set_backlight_brightness(value):
    """Best-effort backlight control for DSI panels via /sys/class/backlight."""
    paths = _discover_backlight_paths()
    for p in paths:
        try:
            # cache previous value once
            if p not in _backlight_prev:
                try:
                    _backlight_prev[p] = int(Path(p).read_text().strip())
                except Exception:
                    _backlight_prev[p] = None
            Path(p).write_text(str(int(value)))
        except Exception:
            continue

def _restore_backlight():
    for p, prev in list(_backlight_prev.items()):
        if prev is None:
            continue
        try:
            Path(p).write_text(str(int(prev)))
        except Exception:
            pass


def _get_backlight_max_for_path(brightness_path):
    """Return max_brightness for a given brightness sysfs node, defaulting to 255."""
    try:
        max_path = str(Path(brightness_path).with_name("max_brightness"))
        return int(Path(max_path).read_text().strip())
    except Exception:
        return 255

def _wake_backlight_target():
    """Compute a deterministic wake brightness target (50% of max)."""
    paths = _discover_backlight_paths()
    if not paths:
        return None
    mx = _get_backlight_max_for_path(paths[0])
    # 50% of max, clamp to [1, max]
    target = int(round(mx * 0.5))
    if target < 1:
        target = 1
    if target > mx:
        target = mx
    return target

def _fade_backlight_to(target, duration_s=0.25, steps=12):
    """Fade backlight from current (or 0) to target in small steps."""
    if target is None:
        return
    # Ensure target is int
    try:
        target = int(target)
    except Exception:
        return
    if steps < 1:
        steps = 1
    dt = float(duration_s) / float(steps)
    # Start from 0 to avoid relying on cached previous brightness.
    for i in range(steps + 1):
        v = int(round(target * (i / float(steps))))
        _set_backlight_brightness(v)
        if dt > 0:
            time.sleep(dt)

def _wake_backlight_fade_in():
    """Wake backlight at a deterministic 50% level with a short fade-in."""
    target = _wake_backlight_target()
    _fade_backlight_to(target, duration_s=0.25, steps=12)

def _set_wifi_block(block=True):
    """Block/unblock Wi‑Fi using rfkill (best-effort)."""
    try:
        subprocess.run(["sudo", "rfkill", "block" if block else "unblock", "wifi"], check=False)
    except Exception:
        pass

_cpu_policy_paths = None

def _get_cpu_policy_paths():
    global _cpu_policy_paths
    if _cpu_policy_paths is None:
        _cpu_policy_paths = _glob_module.glob("/sys/devices/system/cpu/cpufreq/policy*")
    return _cpu_policy_paths

def _set_cpu_low_power(enable=True):
    """Switch CPU governor and optionally clamp max freq (best-effort)."""
    try:
        pols = _get_cpu_policy_paths()
        for pol in pols:
            gov = Path(pol) / "scaling_governor"
            mx = Path(pol) / "scaling_max_freq"
            mn = Path(pol) / "scaling_min_freq"
            if gov.exists():
                if pol not in _cpu_prev_governor:
                    try:
                        _cpu_prev_governor[pol] = gov.read_text().strip()
                    except Exception:
                        _cpu_prev_governor[pol] = None
            if mx.exists() and pol not in _cpu_prev_max_freq:
                try:
                    _cpu_prev_max_freq[pol] = mx.read_text().strip()
                except Exception:
                    _cpu_prev_max_freq[pol] = None

            if enable:
                # governor
                try:
                    gov.write_text("powersave")
                except Exception:
                    pass
                # clamp max freq down to min freq
                try:
                    if mx.exists() and mn.exists():
                        mx.write_text(mn.read_text().strip())
                except Exception:
                    pass
            else:
                # restore
                try:
                    prev_g = _cpu_prev_governor.get(pol)
                    if prev_g:
                        gov.write_text(prev_g)
                except Exception:
                    pass
                try:
                    prev_m = _cpu_prev_max_freq.get(pol)
                    if prev_m:
                        mx.write_text(prev_m)
                except Exception:
                    pass
    except Exception:
        pass

# ---- CPU thermal cap (active during normal operation) ----
_CPU_THERMAL_CAP_KHZ = 1200000  # 1.2 GHz cap (Pi 5 max is 2.4 GHz)
_CPU_4K_BOOST_KHZ   = 1800000  # 1.8 GHz boost for 4K video recording

def _apply_cpu_thermal_cap():
    """Cap CPU max frequency to reduce heat during normal operation (best-effort).

    Uses 'ondemand' governor so the CPU still scales down when idle, but
    never exceeds _CPU_THERMAL_CAP_KHZ.  This runs once at startup; the
    existing _set_cpu_low_power sleep-mode logic saves/restores whatever
    the current (capped) state is, so sleep/wake still work correctly.
    """
    try:
        pols = _get_cpu_policy_paths()
        for pol in pols:
            gov = Path(pol) / "scaling_governor"
            mx = Path(pol) / "scaling_max_freq"
            if not mx.exists():
                continue
            # Set governor to ondemand (dynamic scaling) or schedutil as fallback
            if gov.exists():
                try:
                    gov.write_text("ondemand")
                except Exception:
                    try:
                        gov.write_text("schedutil")
                    except Exception:
                        pass
            # Cap max frequency
            try:
                mx.write_text(str(_CPU_THERMAL_CAP_KHZ))
            except Exception:
                pass
        print(f"[CPU] Thermal cap applied: {_CPU_THERMAL_CAP_KHZ // 1000} MHz")
    except Exception:
        pass

def _set_cpu_4k_boost(enable=True):
    """Raise CPU max frequency to 1.8 GHz for 4K video, or restore to normal cap."""
    target_khz = _CPU_4K_BOOST_KHZ if enable else _CPU_THERMAL_CAP_KHZ
    try:
        pols = _get_cpu_policy_paths()
        for pol in pols:
            mx = Path(pol) / "scaling_max_freq"
            if mx.exists():
                try:
                    mx.write_text(str(target_khz))
                except Exception:
                    pass
        label = "boost" if enable else "restore"
        print(f"[CPU] 4K {label}: max freq set to {target_khz // 1000} MHz")
    except Exception:
        pass


def _fan_sysfs_paths():
    """Return a list of writable fan control nodes.

    On Pi 5 (and some hats/cases), the primary control is usually exposed as one or more
    /sys/class/thermal/cooling_device*/cur_state entries. The driver may override manual
    writes if the SoC is hot, so we treat this as best-effort.
    """
    global _fan_paths_cache
    if _fan_paths_cache is not None:
        return _fan_paths_cache

    paths = []
    base = Path("/sys/class/thermal")
    if base.exists():
        for cd in sorted(base.glob("cooling_device*")):
            t = (cd / "type")
            cs = (cd / "cur_state")
            if not (t.exists() and cs.exists()):
                continue
            try:
                typ = t.read_text().strip().lower()
            except Exception:
                typ = ""
            # Heuristics: prefer fan / pwm-fan devices
            if ("fan" in typ) or ("pwm" in typ):
                paths.append(str(cs))

    # Fallback to cooling_device0 if present
    p0 = Path("/sys/class/thermal/cooling_device0/cur_state")
    if str(p0) not in paths and p0.exists():
        paths.append(str(p0))

    _fan_paths_cache = paths
    return paths

def _set_fan_state(state):
    """Best-effort set fan cooling state (0=off)."""
    try:
        s = str(int(state))
    except Exception:
        s = "0"
    for p in _fan_sysfs_paths():
        try:
            Path(p).write_text(s)
        except Exception:
            pass

def _snapshot_fan_state():
    """Capture current fan state(s) for restore on wake."""
    global _fan_prev_states
    for p in _fan_sysfs_paths():
        if p in _fan_prev_states:
            continue
        try:
            _fan_prev_states[p] = Path(p).read_text().strip()
        except Exception:
            pass

def _restore_fan_state():
    """Restore fan state(s) captured before sleep."""
    global _fan_prev_states
    for p, v in list(_fan_prev_states.items()):
        try:
            Path(p).write_text(str(v))
        except Exception:
            pass
    _fan_prev_states = {}

def _enter_sleep_mode():
    """Enter low-power 'fake sleep'."""
    global sleep_mode_active
    if sleep_mode_active:
        return

    # Stop any active video recording before sleeping
    if _video_mode_active:
        _exit_video_mode()

    sleep_mode_active = True
    _sleep_wake_event.clear()

    # Stop camera pipeline to reduce ISP/CPU load
    try:
        picam2.stop()
    except Exception:
        pass

    # Blank backlight (DSI panels typically expose /sys/class/backlight)
    _set_backlight_brightness(0)

    # Reduce power
    _set_wifi_block(True)
    _set_cpu_low_power(True)
    # Force fan off (best-effort; driver may override if hot)
    _snapshot_fan_state()
    _set_fan_state(0)

def _exit_sleep_mode():
    """Wake from sleep: restore power settings and restart preview."""
    global sleep_mode_active, _current_backlight_level, _backlight_dimmed
    if not sleep_mode_active:
        return
    sleep_mode_active = False

    # Restore settings
    _wake_backlight_fade_in()
    # Sync internal brightness state to 50% after fade-in completes
    _current_backlight_level = _BACKLIGHT_NORMAL_LEVEL
    _backlight_dimmed = False
    _set_cpu_low_power(False)
    _set_wifi_block(False)
    _restore_fan_state()

    # Restart camera pipeline in the live-view config (still_config on
    # small sensors, lightweight preview config on large sensors)
    try:
        picam2.configure(_preview_running_config)
        picam2.start()
        picam2.set_controls({"FrameDurationLimits": _PREVIEW_FRAME_LIMITS})
        time.sleep(0.1)
        _apply_shutter_controls()
    except Exception as exc:
        print("Wake: preview restart error:", exc)



# Global icon appearance
ICON_GLOBAL_OPACITY = 0.75
ICON_PRESSED_OPACITY = 1.0
ICON_PRESSED_BRIGHTEN = 1.35  # multiplier applied to icon pixels when pressed
LEFT_COLUMN_MARGIN = 24

# ---- Unified UI color palette (RGB) ----
UI_ACCENT_EXPOSURE = (235, 165, 75)     # Warm orange – shutter / ISO
UI_ACCENT_CREATIVE = (235, 165, 75)     # Warm orange – film / aspect / double-exp
UI_ACCENT_SYSTEM   = (215, 210, 195)    # Warm neutral – flip / wifi / charge / sleep
UI_ACCENT_FOCUS    = (190, 225, 200)    # Soft green – rangefinder
UI_ACCENT_DANGER   = (200, 130, 130)    # Soft red – trash / delete
UI_INACTIVE        = (155, 155, 155)    # Unified inactive gray
UI_TEXT_PRIMARY     = (235, 235, 235)    # Primary icon/text color
UI_TEXT_DIM         = (175, 175, 175)    # Dimmed icon/text color

# ---- PNG icon loading system ----
_ICON_DIR = Path(__file__).with_name("icons")
_icon_png_cache = {}


def _load_icon_png(name, size):
    """Load a PNG icon from the icons/ directory, resize, and cache it.

    Returns an (size, size, 3) BGR numpy array, or None if not found.
    The icon is tinted white on transparent and composited onto black.
    """
    key = (name, size)
    cached = _icon_png_cache.get(key)
    if cached is not None:
        return cached
    icon_path = _ICON_DIR / f"{name}.png"
    if not icon_path.exists():
        _icon_png_cache[key] = None
        return None
    try:
        raw = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            _icon_png_cache[key] = None
            return None
        resized = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            img = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        elif resized.shape[2] == 4:
            # Use alpha channel as icon shape — works with both
            # white-on-transparent and black-on-transparent icons
            img = cv2.cvtColor(resized[:, :, 3], cv2.COLOR_GRAY2BGR)
        else:
            img = resized
        _icon_png_cache[key] = img
        return img
    except Exception:
        _icon_png_cache[key] = None
        return None


def _tint_icon(icon_bgr, color_rgb):
    """Tint a white-on-black icon image with the given RGB color."""
    if icon_bgr is None:
        return None
    # icon is white-on-black; multiply by normalized color
    r, g, b = color_rgb
    tinted = icon_bgr.copy()
    tinted[:, :, 0] = (icon_bgr[:, :, 0].astype(np.uint16) * b // 255).astype(np.uint8)
    tinted[:, :, 1] = (icon_bgr[:, :, 1].astype(np.uint16) * g // 255).astype(np.uint8)
    tinted[:, :, 2] = (icon_bgr[:, :, 2].astype(np.uint16) * r // 255).astype(np.uint8)
    return tinted


_tinted_icon_cache = {}
_TINTED_ICON_CACHE_MAX = 64

def _place_png_icon(canvas, icon_name, color_rgb):
    """Load a PNG icon at reduced size, tint, and alpha-blend onto canvas center.

    Unlike np.copyto this preserves whatever is already drawn on the canvas
    (e.g. the circle ring from _render_icon_button) by using the icon
    brightness as an alpha mask so only the icon strokes are painted.
    Returns True on success, False if the icon could not be loaded.
    """
    h, w = canvas.shape[:2]
    icon_size = int(min(w, h) * 0.55)
    if icon_size < 4:
        return False
    cache_key = (icon_name, icon_size, color_rgb)
    cached = _tinted_icon_cache.get(cache_key)
    if cached is not None:
        tinted, alpha = cached
    else:
        png = _load_icon_png(icon_name, icon_size)
        if png is None:
            return False
        tinted = _tint_icon(png, color_rgb)
        if tinted is None:
            return False
        alpha = png[:, :, 0:1].astype(np.float32) / 255.0
        if len(_tinted_icon_cache) >= _TINTED_ICON_CACHE_MAX:
            _tinted_icon_cache.clear()
        _tinted_icon_cache[cache_key] = (tinted, alpha)
    y_off = (h - icon_size) // 2
    x_off = (w - icon_size) // 2
    roi = canvas[y_off:y_off + icon_size, x_off:x_off + icon_size]
    blended = (tinted.astype(np.float32) * alpha
               + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    canvas[y_off:y_off + icon_size, x_off:x_off + icon_size] = blended
    return True


# PLD button initialised lazily after gpiozero is imported (see _init_pld_button).
_pld_button = None


def _init_pld_button():
    """Initialise the PLD sense button (GPIO 6).  Called once after gpiozero is imported."""
    global _pld_button
    try:
        from gpiozero import Button as _Btn
        _pld_button = _Btn(6)
    except Exception:
        _pld_button = None


def _force_read_battery():
    global batt_percent_cached, batt_voltage_cached
    global batt_cell_voltage_cached, batt_pack_voltage_cached
    if not I2C_AVAILABLE:
        return None, None
    bus = None
    try:
        bus = smbus2.SMBus(1)
        cap = _read_capacity(bus)
        volt = _read_voltage(bus)
    except Exception:
        cap = None
        volt = None
    finally:
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass
    if cap is not None or volt is not None:
        _update_battery_cache(cap, volt)
    return volt, cap


def _read_vcgencmd_metric(args, strip_chars=""):
    try:
        output = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None
    if "=" not in output:
        return None
    metric = output.split("=", 1)[1].strip()
    if strip_chars:
        metric = metric.rstrip(strip_chars)
    try:
        return float(metric)
    except ValueError:
        return None


def _read_cpu_temp():
    return _read_vcgencmd_metric(["vcgencmd", "measure_temp"], "'C")


def _read_all_pmic_metrics():
    """Read all PMIC metrics in a single subprocess call.

    Returns dict with keys: cpu_volt, cpu_amp, input_volt, watts.
    Previously this required 4 separate subprocess calls.
    """
    result = {"cpu_volt": None, "cpu_amp": None, "input_volt": None, "watts": None}
    try:
        output = subprocess.check_output(
            ["vcgencmd", "pmic_read_adc"], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return result
    amperages = {}
    voltages = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        label = parts[0]
        value = parts[-1]
        if "=" not in value:
            continue
        try:
            val = float(value.split("=", 1)[1][:-1])
        except ValueError:
            continue
        # Extract individual metrics by label
        if label == "VDD_CORE_V":
            result["cpu_volt"] = val
        elif label == "VDD_CORE_A":
            result["cpu_amp"] = val
        elif label == "EXT5V_V":
            result["input_volt"] = val
        short_label = label[:-2]
        if label.endswith("A"):
            amperages[short_label] = val
        else:
            voltages[short_label] = val
    # Calculate total power
    total = 0.0
    for key, amps in amperages.items():
        volts = voltages.get(key)
        if volts is None:
            continue
        total += amps * volts
    result["watts"] = total if total > 0 else None
    return result


def _read_fan_rpm():
    try:
        base = Path("/sys/devices/platform/cooling_fan")
        for entry in base.rglob("fan1_input"):
            try:
                with open(entry, "r") as fh:
                    rpm = fh.read().strip()
                return f"{rpm} RPM"
            except (FileNotFoundError, PermissionError):
                continue
    except Exception:
        pass
    return None


_pld_state_cache = {"ts": 0.0, "state": None}
_PLD_CACHE_TTL = 2.0  # Refresh AC power state every 2 seconds, not every frame


def _read_pld_state(use_cache=True):
    if _pld_button is None:
        return None
    if use_cache:
        now = time.monotonic()
        if now - _pld_state_cache["ts"] < _PLD_CACHE_TTL:
            return _pld_state_cache["state"]
    try:
        state = 0 if _pld_button.is_pressed else 1
    except Exception:
        state = None
    _pld_state_cache["ts"] = time.monotonic()
    _pld_state_cache["state"] = state
    return state


def _collect_power_stats(force_battery=False):
    if force_battery:
        _force_read_battery()
    with _batt_lock:
        pct = batt_percent_cached
        cell_volt = batt_cell_voltage_cached
        pack_volt = batt_pack_voltage_cached

    cpu_temp = _read_cpu_temp()
    # Single subprocess call for all PMIC metrics (was 4 separate calls)
    pmic = _read_all_pmic_metrics()
    cpu_volt = pmic["cpu_volt"]
    cpu_amp = pmic["cpu_amp"]
    input_volt = pmic["input_volt"]
    watts = pmic["watts"]
    fan = _read_fan_rpm()
    pld_state = _read_pld_state(use_cache=False)

    lines = ["Charging mode"]
    if pct is not None or pack_volt is not None:
        parts = []
        if pct is not None:
            parts.append(f"{pct:.1f}%")
        if pack_volt is not None:
            parts.append(f"{pack_volt:.2f}V")
        elif cell_volt is not None:
            parts.append(f"{cell_volt:.2f}V")
        lines.append("Battery: " + "  ".join(parts))
    else:
        lines.append("Battery: unavailable")

    if input_volt is not None:
        lines.append(f"Input Voltage: {input_volt:.2f}V")
    if cpu_volt is not None or cpu_amp is not None:
        cpu_parts = []
        if cpu_volt is not None:
            cpu_parts.append(f"{cpu_volt:.3f}V")
        if cpu_amp is not None:
            cpu_parts.append(f"{cpu_amp:.3f}A")
        lines.append("CPU Rail: " + " / ".join(cpu_parts))
    if cpu_temp is not None:
        lines.append(f"CPU Temp: {cpu_temp:.1f}°C")
    if watts is not None:
        lines.append(f"Power Usage: {watts:.2f}W")
    if fan:
        lines.append(f"Fan: {fan}")
    if pld_state is not None:
        lines.append("AC Power: OK" if pld_state == 1 else "AC Power: LOST")

    try:
        usage = shutil.disk_usage(LOCAL_PICTURES_DIR)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_mb = usage.free / (1024 * 1024)
        label = f"Storage: {used_gb:.1f} / {total_gb:.1f} GB ({free_gb:.1f} GB free)"
        if free_mb < _DISK_CRITICAL_THRESHOLD_MB:
            label += " *** CRITICAL ***"
        elif free_mb < _DISK_WARN_THRESHOLD_MB:
            label += " (LOW)"
        lines.append(label)
    except Exception:
        pass

    if _charge_shutdown_triggered:
        lines.append("Shutting down – battery full")

    lines.append("")
    lines.append("Tap screen to exit charging mode.")
    lines.append("Unit will shut down automatically at 95%.")
    return lines, pct


def _update_charge_stats(force=False):
    now = time.monotonic()
    if not force and now - _charge_stats_cache["timestamp"] < _CHARGE_STATS_REFRESH:
        return _charge_stats_cache["lines"], _charge_stats_cache["percent"]
    lines, pct = _collect_power_stats(force_battery=force)
    _charge_stats_cache["timestamp"] = now
    _charge_stats_cache["lines"] = lines
    _charge_stats_cache["percent"] = pct
    return lines, pct


def _maybe_trigger_charge_shutdown(percent):
    global _charge_shutdown_triggered
    if _charge_shutdown_triggered:
        return
    if percent is None:
        return
    # Only auto-shutdown when AC power is plugged in during charge mode.
    # Treat None (sensor unavailable) as unknown and allow shutdown,
    # so installations without the PLD GPIO still auto-shutdown at 95%.
    ac_state = _read_pld_state(use_cache=False)
    if ac_state == 0:
        return
    if percent >= 95.0:
        _charge_shutdown_triggered = True
        try:
            subprocess.Popen(
                ["sudo", "shutdown", "-h", "now", "Battery charged"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print("Shutdown request failed:", exc)

# -------------------- UI helpers --------------------

_HIDDEN_CURSOR_PATHS = []
_XFIXES_DATA = None


def _restore_cursor():
    global _XFIXES_DATA
    if not _XFIXES_DATA:
        return
    libX11, libXfixes, display_ptr = _XFIXES_DATA
    try:
        libX11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        libX11.XDefaultRootWindow.restype = ctypes.c_ulong
        root = libX11.XDefaultRootWindow(display_ptr)
        libXfixes.XFixesShowCursor.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        libXfixes.XFixesShowCursor.restype = None
        libXfixes.XFixesShowCursor(display_ptr, root)
        libX11.XFlush.argtypes = [ctypes.c_void_p]
        libX11.XFlush(display_ptr)
        libX11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        libX11.XCloseDisplay.restype = ctypes.c_int
        libX11.XCloseDisplay(display_ptr)
    except Exception:
        pass
    finally:
        _XFIXES_DATA = None


def _cleanup_cursor_artifacts():
    while _HIDDEN_CURSOR_PATHS:
        path = _HIDDEN_CURSOR_PATHS.pop()
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        except Exception as exc:
            print("Cursor cleanup error:", exc)


def _ensure_cursor_hidden():
    display = os.environ.get("DISPLAY")
    if not display:
        # Nothing to do when we are not under X11; rely on the framebuffer stack.
        return

    global _XFIXES_DATA

    if _XFIXES_DATA:
        return

    try:
        libX11 = ctypes.CDLL("libX11.so.6")
        libXfixes = ctypes.CDLL("libXfixes.so.3")

        libX11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        libX11.XOpenDisplay.restype = ctypes.c_void_p
        disp = libX11.XOpenDisplay(display.encode())
        if disp:
            libX11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
            libX11.XDefaultRootWindow.restype = ctypes.c_ulong
            root = libX11.XDefaultRootWindow(disp)
            libXfixes.XFixesHideCursor.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            libXfixes.XFixesHideCursor.restype = None
            libXfixes.XFixesHideCursor(disp, root)
            libX11.XFlush.argtypes = [ctypes.c_void_p]
            libX11.XFlush(disp)
            _XFIXES_DATA = (libX11, libXfixes, disp)
            atexit.register(_restore_cursor)
            return
    except OSError:
        pass
    except Exception as exc:
        print("Cursor XFixes hide error:", exc)

    try:
        fd, cursor_path = tempfile.mkstemp(prefix="cursor", suffix=".xbm")
        with os.fdopen(fd, "w") as cursor_file:
            cursor_file.write(
                "#define blank_width 1\n"
                "#define blank_height 1\n"
                "static unsigned char blank_bits[] = { 0x00 };\n"
            )

        mask_fd, mask_path = tempfile.mkstemp(prefix="cursor", suffix="mask.xbm")
        with os.fdopen(mask_fd, "w") as mask_file:
            mask_file.write(
                "#define blankm_width 1\n"
                "#define blankm_height 1\n"
                "static unsigned char blankm_bits[] = { 0x00 };\n"
            )

        _HIDDEN_CURSOR_PATHS.extend([cursor_path, mask_path])
        atexit.register(_cleanup_cursor_artifacts)

        subprocess.run(
            [
                "xsetroot",
                "-display",
                display,
                "-cursor",
                cursor_path,
                mask_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        # xsetroot not installed; fall back to unclutter with aggressive settings.
        subprocess.Popen(
            [
                "unclutter",
                "-idle",
                "0.05",
                "-noevents",
                "-grab",
                "-display",
                display,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        print("Cursor hide error:", exc)


# Run cursor hiding in background – the subprocess calls can take ~200ms
# and the cursor is not visible on the Pi touchscreen display anyway.
threading.Thread(target=_ensure_cursor_hidden, daemon=True).start()

LOCAL_PICTURES_DIR = "/home/pi/Pictures"
USB_MOUNT_POINT = "/home/pi/usb1"

def _ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as exc:
        print(f"Directory create error for {path}: {exc}")

_ensure_dir(LOCAL_PICTURES_DIR)

_capture_dir_lock = threading.Lock()
_capture_dir = LOCAL_PICTURES_DIR
_usb_device = None
_usb_mounted_by_us = False
_is_usb_target = False

# USB file sync queue for committing files to USB drive
_usb_sync_queue = queue.Queue()

def _set_capture_dir(path):
    global _capture_dir, _is_usb_target
    _ensure_dir(path)
    with _capture_dir_lock:
        _capture_dir = path
        _is_usb_target = path.startswith(USB_MOUNT_POINT)

def _queue_usb_sync(file_path):
    """Queue a file for syncing to USB if capturing to USB drive."""
    if _is_usb_target and file_path and os.path.isfile(file_path):
        _usb_sync_queue.put(file_path)

def _usb_sync_worker():
    """Background thread that commits captured files to USB storage and
    copies them to the local Pictures directory as a backup.

    This ensures DNG and JPG files are properly flushed to the USB drive
    by calling fsync on each file after it's written. Without this,
    files may remain in the kernel buffer cache and be lost if the
    USB drive is removed before the OS flushes the data.

    After syncing to USB, each file is also copied to LOCAL_PICTURES_DIR
    so that a local backup exists on the Pi's SD card.
    """
    while True:
        file_path = _usb_sync_queue.get()
        if file_path is None:
            _usb_sync_queue.task_done()
            break
        try:
            # Open file and fsync to ensure data is committed to USB
            fd = os.open(file_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            # Also sync the parent directory to ensure directory entry is committed
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                dir_fd = os.open(parent_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            # Copy to local Pictures directory as backup
            try:
                _ensure_dir(LOCAL_PICTURES_DIR)
                local_copy = os.path.join(
                    LOCAL_PICTURES_DIR, os.path.basename(file_path)
                )
                shutil.copy2(file_path, local_copy)
            except Exception as e:
                print(f"Local backup error for {file_path}: {e}")
        except Exception as e:
            print(f"USB sync error for {file_path}: {e}")
        finally:
            _usb_sync_queue.task_done()

_set_capture_dir(LOCAL_PICTURES_DIR)

def _detect_usb_partition():
    try:
        output = subprocess.check_output(
            ["lsblk", "-J", "-o", "NAME,MOUNTPOINT,TRAN,TYPE"],
            text=True,
        )
        info = json.loads(output)
    except Exception:
        return None, None

    def _iter_nodes(node):
        yield node
        for child in node.get("children", []) or []:
            yield from _iter_nodes(child)

    for dev in info.get("blockdevices", []) or []:
        if dev.get("tran") != "usb":
            continue
        for node in _iter_nodes(dev):
            if node.get("type") == "part":
                name = node.get("name")
                if name:
                    return f"/dev/{name}", node.get("mountpoint")
    return None, None

def _usb_mount_thread():
    global _usb_device, _usb_mounted_by_us
    _ensure_dir(USB_MOUNT_POINT)
    while True:
        device, mountpoint = _detect_usb_partition()
        if device:
            target_mount = mountpoint or USB_MOUNT_POINT
            mounted_here = (
                _usb_mounted_by_us
                if mountpoint and target_mount == USB_MOUNT_POINT
                else False
            )
            if not mountpoint:
                if not os.path.ismount(target_mount):
                    result = subprocess.call(
                        ["mount", device, target_mount],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    if result != 0:
                        time.sleep(15)
                        continue
                mounted_here = True
                mountpoint = target_mount
            if os.path.isdir(target_mount):
                _set_capture_dir(target_mount)
                _usb_device = device
                _usb_mounted_by_us = mounted_here
        else:
            if _usb_device and _usb_mounted_by_us and os.path.ismount(USB_MOUNT_POINT):
                subprocess.call(
                    ["umount", USB_MOUNT_POINT],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            _set_capture_dir(LOCAL_PICTURES_DIR)
            _usb_device = None
            _usb_mounted_by_us = False
        time.sleep(15)

threading.Thread(target=_usb_mount_thread, daemon=True).start()

image_count = 0


_capture_save_queue = queue.Queue()
_capture_sequence_lock = threading.Lock()
_capture_sequence_timestamp = None
_capture_sequence_index = 0


def _start_capture_sequence():
    global _capture_sequence_timestamp, _capture_sequence_index
    with _capture_sequence_lock:
        _capture_sequence_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _capture_sequence_index = 0


def _next_capture_basename():
    global _capture_sequence_timestamp, _capture_sequence_index
    with _capture_sequence_lock:
        if not _capture_sequence_timestamp:
            _capture_sequence_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _capture_sequence_index = 0
        _capture_sequence_index += 1
        index = _capture_sequence_index
        timestamp = _capture_sequence_timestamp
    return f"{timestamp}_{index:03d}"


def _prepare_output_frame(frame, target_dims, target_ratio):
    if frame is None:
        return None
    frame_to_use = frame
    if frame_to_use.ndim == 3 and frame_to_use.shape[2] == 4:
        frame_to_use = cv2.cvtColor(frame_to_use, cv2.COLOR_BGRA2RGB)
    if target_dims and target_dims[0] and target_dims[1]:
        frame_to_use = _center_crop(frame_to_use, target_dims[0], target_dims[1])
    elif target_ratio > 0:
        crop_w, crop_h = _largest_ratio_crop_dims(
            frame_to_use.shape[1], frame_to_use.shape[0], target_ratio
        )
        frame_to_use = _center_crop(frame_to_use, crop_w, crop_h)
    return frame_to_use


def _rotate_dng_180(dng_path):
    orientation_tag = 0x0112
    new_subfile_tag = 0x00FE
    sub_ifd_tag = 0x014A
    type_sizes = {
        1: 1,   # BYTE
        2: 1,   # ASCII
        3: 2,   # SHORT
        4: 4,   # LONG
        5: 8,   # RATIONAL
        7: 1,   # UNDEFINED
        9: 4,   # SLONG
        10: 8,  # SRATIONAL
    }

    if not dng_path or not os.path.isfile(dng_path):
        return
    try:
        with open(dng_path, "r+b") as handle:
            header = handle.read(8)
            if len(header) < 8:
                return
            endian = header[:2]
            if endian == b"II":
                fmt_prefix = "<"
            elif endian == b"MM":
                fmt_prefix = ">"
            else:
                return
            if struct.unpack(fmt_prefix + "H", header[2:4])[0] != 42:
                return
            first_ifd = struct.unpack(fmt_prefix + "I", header[4:8])[0]
            visited = set()
            ifd_queue = [first_ifd]

            def _read_value(entry_type, count, value_offset):
                if entry_type not in type_sizes:
                    return None
                total = type_sizes[entry_type] * count
                if total <= 4:
                    data = value_offset.to_bytes(4, "little" if fmt_prefix == "<" else "big")
                else:
                    handle.seek(value_offset)
                    data = handle.read(total)
                if entry_type == 3 and count == 1:
                    return struct.unpack(fmt_prefix + "H", data[:2])[0]
                if entry_type == 4 and count == 1:
                    return struct.unpack(fmt_prefix + "I", data[:4])[0]
                return None

            def _write_orientation(entry_type, count, value_offset):
                if entry_type != 3 or count != 1:
                    return
                total = type_sizes[entry_type] * count
                if total <= 4:
                    handle.seek(value_offset)
                    handle.write(struct.pack(fmt_prefix + "H", 3))
                    handle.write(b"\x00\x00")
                else:
                    handle.seek(value_offset)
                    handle.write(struct.pack(fmt_prefix + "H", 3))

            while ifd_queue:
                ifd_offset = ifd_queue.pop(0)
                if not ifd_offset or ifd_offset in visited:
                    continue
                visited.add(ifd_offset)
                handle.seek(ifd_offset)
                count_data = handle.read(2)
                if len(count_data) < 2:
                    continue
                entry_count = struct.unpack(fmt_prefix + "H", count_data)[0]
                entries_start = handle.tell()
                new_subfile_type = 0
                orientation_entry = None
                sub_ifd_offsets = []
                for _ in range(entry_count):
                    entry = handle.read(12)
                    if len(entry) < 12:
                        break
                    tag, entry_type, count, value_offset = struct.unpack(
                        fmt_prefix + "HHII", entry
                    )
                    if tag == new_subfile_tag:
                        value = _read_value(entry_type, count, value_offset)
                        if value is not None:
                            new_subfile_type = value
                    elif tag == orientation_tag:
                        orientation_entry = (entry_type, count, value_offset)
                    elif tag == sub_ifd_tag and entry_type == 4:
                        if count == 1:
                            sub_ifd_offsets.append(value_offset)
                        else:
                            total = type_sizes[entry_type] * count
                            handle.seek(value_offset)
                            data = handle.read(total)
                            for idx in range(count):
                                offset = struct.unpack(
                                    fmt_prefix + "I", data[idx * 4:(idx + 1) * 4]
                                )[0]
                                sub_ifd_offsets.append(offset)
                if new_subfile_type == 0 and orientation_entry is not None:
                    entry_type, count, value_offset = orientation_entry
                    if type_sizes.get(entry_type):
                        entry_index = None
                        handle.seek(entries_start)
                        for idx in range(entry_count):
                            entry = handle.read(12)
                            if len(entry) < 12:
                                break
                            tag = struct.unpack(fmt_prefix + "H", entry[:2])[0]
                            if tag == orientation_tag:
                                entry_index = idx
                                break
                        if entry_index is not None:
                            entry_value_pos = entries_start + entry_index * 12 + 8
                            _write_orientation(entry_type, count, entry_value_pos if type_sizes[entry_type] * count <= 4 else value_offset)
                for sub_offset in sub_ifd_offsets:
                    ifd_queue.append(sub_offset)
                handle.seek(entries_start + entry_count * 12)
                next_ifd_data = handle.read(4)
                if len(next_ifd_data) == 4:
                    next_ifd = struct.unpack(fmt_prefix + "I", next_ifd_data)[0]
                    if next_ifd:
                        ifd_queue.append(next_ifd)
    except Exception as exc:
        print("DNG rotate error:", exc)


def _capture_save_worker():
    global image_count
    while True:
        job = _capture_save_queue.get()
        if job is None:
            _capture_save_queue.task_done()
            break
        (
            full_frame,
            target_dims,
            target_ratio,
            film_key,
            jpg_path,
            overlay_frame,
            dng_path,
        ) = job
        try:
            # Rotate DNG orientation tag in background (moved off main thread)
            if dng_path:
                try:
                    _rotate_dng_180(dng_path)
                    _queue_usb_sync(dng_path)
                except Exception as dng_err:
                    print(f"[Save] DNG rotate error: {dng_err}")
            frame_to_save = _prepare_output_frame(full_frame, target_dims, target_ratio)
            if overlay_frame is not None:
                overlay_to_use = _prepare_output_frame(overlay_frame, target_dims, target_ratio)
                if overlay_to_use is not None and frame_to_save is not None:
                    if overlay_to_use.shape[:2] != frame_to_save.shape[:2]:
                        overlay_to_use = cv2.resize(
                            overlay_to_use,
                            (frame_to_save.shape[1], frame_to_save.shape[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    frame_to_save = cv2.addWeighted(
                        frame_to_save.astype(np.float32),
                        0.5,
                        overlay_to_use.astype(np.float32),
                        0.5,
                        0,
                    ).astype(np.uint8)
            if film_key and film_key != "none":
                frame_to_save = apply_film_simulation_rgb(frame_to_save, film_key)
            if frame_to_save.ndim == 3 and frame_to_save.shape[2] == 3:
                frame_to_save = cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR)
            frame_to_save = cv2.rotate(frame_to_save, cv2.ROTATE_180)
            cv2.imwrite(jpg_path, frame_to_save, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            _queue_usb_sync(jpg_path)
            image_count += 1
        except Exception as e:
            print("Save error:", e)
        finally:
            _record_activity(wake=True)
            _capture_save_queue.task_done()


focus_peaking_enabled = _bool_value(_PERSISTED_SETTINGS.get("focus_peaking_enabled", True), True)

# Performance level: 0=LOW (fast), 1=MEDIUM (balanced), 2=HIGH (accurate)
FOCUS_PERF_LEVEL = 1  # Default: balanced

# Performance presets for focus peaking
# Format: (max_dim, blur_ksize, threshold_offset, skip_frames)
# No downsampling - process at full assist frame resolution to avoid blocky edges
_FOCUS_PERF_PRESETS = {
    0: (480, 5, 35, 1),    # LOW: smaller max dim, higher threshold, skip 1
    1: (640, 5, 30, 0),    # MEDIUM: balanced
    2: (800, 5, 25, 0),    # HIGH: larger processing size, more sensitive
}

# Performance presets for rangefinder
# Format: (roi_size, interval_ms, smoothing)
_RF_PERF_PRESETS = {
    0: (120, 66, 0.5),    # LOW: smaller ROI, ~15fps, faster smoothing
    1: (150, 33, 0.35),   # MEDIUM: balanced ~30fps
    2: (180, 0, 0.25),    # HIGH: full ROI, every frame, smoother
}

RF_ENABLED_DEFAULT = False
RF_MAX_SHIFT = 12
RF_SHIFT_DEADBAND = 1
RF_GHOST_ALPHA = 0.45
RF_GHOST_TINT = 0.25

# Dynamic values based on performance level
def _get_focus_preset():
    return _FOCUS_PERF_PRESETS.get(FOCUS_PERF_LEVEL, _FOCUS_PERF_PRESETS[1])

def _get_rf_preset():
    return _RF_PERF_PRESETS.get(FOCUS_PERF_LEVEL, _RF_PERF_PRESETS[1])

def set_focus_performance_level(level):
    """Set focus assist performance level: 0=LOW, 1=MEDIUM, 2=HIGH.

    LOW (0): Faster processing, less sensitive to avoid false positives.
             - 480px max dim, threshold +35, skip 1 frame
    MEDIUM (1): Balanced performance and sensitivity. Default setting.
             - 640px max dim, threshold +30, no skipping
    HIGH (2): Larger processing size, more sensitive to fine detail.
             - 800px max dim, threshold +25, no skipping
    """
    global FOCUS_PERF_LEVEL, _focus_last_mask, _focus_frame_counter
    if level not in (0, 1, 2):
        return False
    FOCUS_PERF_LEVEL = level
    # Clear cached mask when changing levels
    _focus_last_mask = None
    _focus_frame_counter = 0
    return True

def get_focus_performance_level():
    """Get current focus assist performance level (0=LOW, 1=MEDIUM, 2=HIGH)."""
    return FOCUS_PERF_LEVEL

def cycle_focus_performance_level():
    """Cycle through performance levels (0->1->2->0). Returns new level."""
    global FOCUS_PERF_LEVEL
    new_level = (FOCUS_PERF_LEVEL + 1) % 3
    set_focus_performance_level(new_level)
    return new_level

rangefinder_assist_enabled = _bool_value(_PERSISTED_SETTINGS.get("rangefinder_assist_enabled", RF_ENABLED_DEFAULT), RF_ENABLED_DEFAULT)
_rangefinder_prev_peaking = None
_rf_last_update = 0.0
_rf_shift_px = 0
_rf_shift_smoothed = 0.0

# Focus peaking frame skip state
_focus_frame_counter = 0
_focus_last_mask = None
_focus_last_gray = None

# Pre-computed kernels for different sizes
_FOCUS_KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_FOCUS_KERNEL_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
_FOCUS_BLEND_NUM = 35
_FOCUS_BLEND_DEN = 100
_FOCUS_COLOR = np.array([[0, 255, 0]], dtype=np.uint16)

# Rangefinder pre-allocated buffers
_rf_ghost_buffer = None
_rf_combined_buffer = None

# Shutter OSD composite cache — avoids per-frame GaussianBlur + text render
_osd_composite_cache_key = None
_osd_composite_text = None   # rendered text at full opacity
_osd_composite_shadow = None # pre-blurred shadow at full opacity

SCREEN_W, SCREEN_H = 800, 480
BAR_H = 64
TEXT_RENDER_SCALE = 2.0
ACCENT_COLOR_RGB = (80, 180, 255)   # soft blue — PIL uses RGB order

BACKLIGHT_PATH = "/sys/class/backlight/rpi_backlight/brightness"
BACKLIGHT_MAX_PATH = "/sys/class/backlight/rpi_backlight/max_brightness"
_BACKLIGHT_AVAILABLE = os.access(BACKLIGHT_PATH, os.W_OK)
try:
    with open(BACKLIGHT_MAX_PATH, "r") as _f:
        _BACKLIGHT_MAX = max(1, int(_f.read().strip() or "0"))
except Exception:
    _BACKLIGHT_MAX = 255

def _pct_to_level(pct):
    return max(0, min(_BACKLIGHT_MAX, int(round(_BACKLIGHT_MAX * (pct / 100.0)))))

_BACKLIGHT_NORMAL_LEVEL = _pct_to_level(50)
_BACKLIGHT_DIM_LEVEL = _pct_to_level(10)
_current_backlight_level = None
_last_activity_time = time.monotonic()
_backlight_dimmed = False

def _set_backlight_level(level):
    global _current_backlight_level, _BACKLIGHT_AVAILABLE
    if not _BACKLIGHT_AVAILABLE:
        return
    level = max(0, min(_BACKLIGHT_MAX, int(level)))
    if _current_backlight_level == level:
        return
    try:
        with open(BACKLIGHT_PATH, "w") as fh:
            fh.write(str(level))
        _current_backlight_level = level
    except Exception as exc:
        print("Backlight write error:", exc)
        _BACKLIGHT_AVAILABLE = False

def _record_activity(wake=False):
    global _last_activity_time, _backlight_dimmed, _current_backlight_level
    _last_activity_time = time.monotonic()
    if wake and _BACKLIGHT_AVAILABLE:
        # Force brightness write on UI interaction as a safeguard against
        # hardware state drift; invalidate cache to ensure write happens
        _current_backlight_level = None
        _set_backlight_level(_BACKLIGHT_NORMAL_LEVEL)
        _backlight_dimmed = False

def _update_backlight(now=None):
    global _backlight_dimmed
    if not _BACKLIGHT_AVAILABLE:
        return
    if now is None:
        now = time.monotonic()
    if now - _last_activity_time >= 120:
        if not _backlight_dimmed:
            _set_backlight_level(_BACKLIGHT_DIM_LEVEL)
            _backlight_dimmed = True
    elif _backlight_dimmed:
        _set_backlight_level(_BACKLIGHT_NORMAL_LEVEL)
        _backlight_dimmed = False

_record_activity(wake=True)

threading.Thread(target=_capture_save_worker, daemon=True).start()
threading.Thread(target=_usb_sync_worker, daemon=True).start()

def _even(val):
    return max(2, (int(val) + 1) // 2 * 2)

# True when the attached sensor is monochrome (e.g. IMX585 mono).  Set to
# its real value after the camera is initialised further down; the default
# is True so `_mono_to_gray` keeps its fast path until the sensor identity
# is known.  Color sensors (IMX492, IMX477, …) fall through to a proper
# RGB→luma conversion so histogram, focus peaking and exposure assist stay
# accurate.
IS_MONO_SENSOR = True

def _mono_to_gray(frame_rgb):
    """Extract grayscale from a camera frame efficiently.

    On a monochrome sensor R=G=B for every pixel, so extracting a single
    channel is equivalent to ``cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)``
    but avoids the weighted-sum computation.  On a Bayer color sensor the
    channels differ, so we do a proper RGB→luma conversion.
    """
    if frame_rgb.ndim == 2:
        return frame_rgb
    if IS_MONO_SENSOR:
        return np.ascontiguousarray(frame_rgb[:, :, 0])
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

wifi_enabled = False
_wifi_info_until = 0.0  # monotonic timestamp until which Wi-Fi info overlay is shown
WIFI_INFO_DURATION_S = 30.0
_wifi_button_rect = (0, 0, 0, 0)
_wifi_lock = threading.Lock()

FLIP_MODE_NORMAL = 0
FLIP_MODE_MIRROR = 1
FLIP_MODE_MIRROR_ROTATE = 2
flip_mode = _bounded_int(_PERSISTED_SETTINGS.get("flip_mode", FLIP_MODE_NORMAL), FLIP_MODE_NORMAL, FLIP_MODE_NORMAL, FLIP_MODE_MIRROR_ROTATE)
_flip_button_rect = (0, 0, 0, 0)

_exit_button_rect = (0, 0, 0, 0)

double_exposure_enabled = False  # never persisted — always starts off
double_exposure_first_frame = None
_double_exposure_button_rect = (0, 0, 0, 0)
_double_exposure_lock = threading.Lock()
_double_preview_cache = {"key": None, "image": None}
_DOUBLE_PREVIEW_ALPHA = 0.45

# --- Brenizer Mode (guided multi-shot capture for panoramic stitching) ---
# Tile definitions: each entry is {"id": capture_order, "col": grid_col, "row": grid_row, "label": direction_label}

# Brenizer 5 layout (simple cross — center + 4 cardinal directions):
#     [2]
# [4] [1] [5]
#     [3]
BRENIZER_5_TILES = [
    {"id": 1, "col": 1, "row": 1, "label": "Center"},
    {"id": 2, "col": 1, "row": 0, "label": "Top"},
    {"id": 3, "col": 1, "row": 2, "label": "Bottom"},
    {"id": 4, "col": 0, "row": 1, "label": "Left"},
    {"id": 5, "col": 2, "row": 1, "label": "Right"},
]
BRENIZER_5_GRID = (3, 3)  # cols, rows
BRENIZER_5_DIRECTIONS = [
    "Center — capture your subject",
    "Move down",   # tile 2: top of grid → pan down on flipped display
    "Move up",     # tile 3: bottom of grid → pan up on flipped display
    "Move right",  # tile 4: left of grid → pan right on flipped display
    "Move left",   # tile 5: right of grid → pan left on flipped display
]

# Brenizer 6 layout (optimised for 16:9 handheld portrait):
# Capture order: center, up, down, left, right, upper-right
# Cross pattern first (single-axis moves), then diagonal.
#      [6] [2]
#   [4] [1] [5]
#      [3]
BRENIZER_6_TILES = [
    {"id": 1, "col": 1, "row": 1, "label": "Center"},
    {"id": 2, "col": 1, "row": 0, "label": "Top"},
    {"id": 3, "col": 1, "row": 2, "label": "Bottom"},
    {"id": 4, "col": 0, "row": 1, "label": "Left"},
    {"id": 5, "col": 2, "row": 1, "label": "Right"},
    {"id": 6, "col": 0, "row": 0, "label": "Upper Left"},
]
BRENIZER_6_GRID = (3, 3)  # cols, rows
BRENIZER_6_DIRECTIONS = [
    "Center — capture your subject",
    "Move down",       # tile 2: top of grid → pan down on flipped display
    "Move up",         # tile 3: bottom of grid → pan up on flipped display
    "Move right",      # tile 4: left of grid → pan right on flipped display
    "Move left",       # tile 5: right of grid → pan left on flipped display
    "Move down-right", # tile 6: upper-left of grid → pan down-right on flipped display
]

# Brenizer 9 layout (standard 3x3):
# Capture order: center, up, down, left, right, then corners.
# [7] [2] [6]
# [4] [1] [5]
# [8] [3] [9]
BRENIZER_9_TILES = [
    {"id": 1, "col": 1, "row": 1, "label": "Center"},
    {"id": 2, "col": 1, "row": 0, "label": "Top"},
    {"id": 3, "col": 1, "row": 2, "label": "Bottom"},
    {"id": 4, "col": 0, "row": 1, "label": "Left"},
    {"id": 5, "col": 2, "row": 1, "label": "Right"},
    {"id": 6, "col": 2, "row": 0, "label": "Top Right"},
    {"id": 7, "col": 0, "row": 0, "label": "Top Left"},
    {"id": 8, "col": 0, "row": 2, "label": "Bottom Left"},
    {"id": 9, "col": 2, "row": 2, "label": "Bottom Right"},
]
BRENIZER_9_GRID = (3, 3)
BRENIZER_9_DIRECTIONS = [
    "Center — capture your subject",
    "Move down",       # tile 2: top → pan down on flipped display
    "Move up",         # tile 3: bottom → pan up on flipped display
    "Move right",      # tile 4: left → pan right on flipped display
    "Move left",       # tile 5: right → pan left on flipped display
    "Move down-left",  # tile 6: top-right → pan down-left on flipped display
    "Move down-right", # tile 7: top-left → pan down-right on flipped display
    "Move up-right",   # tile 8: bottom-left → pan up-right on flipped display
    "Move up-left",    # tile 9: bottom-right → pan up-left on flipped display
]

_brenizer_lock = threading.Lock()
_brenizer_active = False
_brenizer_variant = 5              # 5, 6, or 9
_brenizer_state = "idle"           # idle | awaiting_capture | complete
_brenizer_tile_idx = 0             # 0-based index into tile list
_brenizer_captured = []            # file paths captured so far
_brenizer_seq_dir = None           # subdirectory for this sequence
_brenizer_seq_id = None            # timestamp string for file naming
_brenizer_locked_controls = None   # dict of locked camera controls after frame 1
_brenizer_show_help = False        # instruction overlay visible
_brenizer_complete_rects = {}      # touch rects for completion screen buttons
_brenizer_pulse_t = 0.0            # animation timer for pulsing current tile
_brenizer_merge_toast_until = 0.0  # time to show "coming soon" toast
_brenizer_tile_thumbs = []         # list of per-tile thumbnail images (BGR, cell-sized)

# Multishot popup (replaces direct double-exposure toggle)
MULTISHOT_OPTIONS = [
    {"key": "double_exposure", "label": "Double Exp"},
    {"key": "brenizer_5", "label": "Brenizer 5"},
    {"key": "brenizer_6", "label": "Brenizer 6"},
    {"key": "brenizer_9", "label": "Brenizer 9"},
    {"key": "video_1080p", "label": "Video 1080p"},
    {"key": "video_4k", "label": "Video 4K"},
]
_multishot_popup_visible = False
_multishot_popup_rects = []
_multishot_popup_bounds = None
_multishot_popup_fade = 0.0

_rangefinder_button_rect = (0, 0, 0, 0)

# ---- Video recording mode ----
# Resolution presets for video recording (width, height, label)
VIDEO_RESOLUTION_OPTIONS = [
    {"key": "1080p", "label": "1080p", "size": (1920, 1080)},
    {"key": "4k",    "label": "4K",    "size": (3840, 2160)},
]
_video_resolution_idx = _bounded_int(_PERSISTED_SETTINGS.get("video_resolution_idx", 0), 0, 0, len(VIDEO_RESOLUTION_OPTIONS) - 1)
_VIDEO_FPS = 24                        # 24 fps cinematic
_VIDEO_MAX_DURATION_S = 120            # 2-minute clip limit
_VIDEO_WARN_REMAINING_S = 15           # Show warning at 15s remaining
_BATT_WARN_PCT = 10                    # Show low battery warning at this %
_BATT_CRITICAL_PCT = 5                 # Stop/block video recording at this %
_VIDEO_CODEC = "h264"                  # Software H.264 via libav (Pi 5 has no HW encoder)

# Video mode state
_video_mode_active = False             # True when camera is in video config
_video_recording = False               # True when actively recording a clip
_video_recording_start = 0.0           # monotonic time when recording started
_video_output_path = None              # Path to current .mp4 file
_video_encoder = None                  # picamera2 encoder object
_video_output = None                   # picamera2 output object
_video_lock = threading.Lock()
_video_help_shown = False              # Help overlay shown on first entry
_video_config = None                   # picamera2 video configuration (created on demand)

# Focus mode selection: 0=Off, 1=Focus Peaking, 2=Rangefinder Focus
FOCUS_MODE_LABELS = ["Peaking Off", "Focus Peaking", "Rangefinder"]
_focus_mode_idx = 0  # will be initialised from persisted rangefinder/peaking flags below
_focus_button_lock = threading.Lock()
_focus_toggle_rect = (0, 0, 0, 0)
_focus_popup_visible = False
_focus_popup_rects = []
_focus_popup_bounds = None
_focus_popup_fade = 0.0
_focus_popup_fade_target = 0.0

# Minimal UI mode: hide all buttons, show only preview, histogram, battery, settings text
_ui_minimal_mode = _bool_value(_PERSISTED_SETTINGS.get("ui_minimal_mode", False), False)
_minimal_mode_timer = None
_MINIMAL_MODE_HOLD_EXTRA_S = 5.0  # Additional seconds after initial 2s hold (7s total)

# Smooth fade for icon hide/show transition
_icons_fade = 0.0 if _ui_minimal_mode else 1.0   # Current opacity 0..1
_icons_fade_target = 0.0 if _ui_minimal_mode else 1.0
_ICONS_FADE_SPEED = 4.0  # Smooth but not sluggish

# Touch-press visual feedback: track where the finger is down (canvas coords)
_touch_down_pos = None
_touch_down_lock = threading.Lock()

# Swipe tracking variables
_swipe_start_x = None
_swipe_start_y = None
_swipe_start_time = None
_SWIPE_MIN_DISTANCE = 100   # Minimum horizontal distance (pixels) for left/right swipe
_SWIPE_MAX_TIME = 0.6       # Maximum duration (seconds) for a valid swipe
_SWIPE_MAX_VERTICAL = 100   # Maximum vertical deviation (pixels) for horizontal swipes

# Swipe-up-to-sleep thresholds (deliberately stricter to avoid accidental triggers)
_SWIPE_UP_MIN_DISTANCE = 150   # Minimum upward distance (pixels) — ~31 % of 480 px screen
_SWIPE_UP_MAX_HORIZONTAL = 80  # Maximum horizontal drift (pixels) allowed
_SWIPE_UP_MAX_TIME = 0.5       # Maximum duration (seconds) for a valid swipe-up

# Swipe-down-to-hide-icons thresholds (mirror swipe-up)
_SWIPE_DOWN_MIN_DISTANCE = 150
_SWIPE_DOWN_MAX_HORIZONTAL = 80
_SWIPE_DOWN_MAX_TIME = 0.5

_shutter_button_lock = threading.Lock()
_shutter_toggle_rect = (0, 0, 0, 0)
_shutter_popup_visible = False
_shutter_popup_rects = []
_shutter_popup_bounds = None
_shutter_popup_fade = 0.0
_shutter_popup_fade_target = 0.0

ASPECT_OPTIONS = [
    {"label": "16:9", "ratio": 16.0 / 9.0},
    {"label": "3:2", "ratio": 3.0 / 2.0},
    {"label": "3:1", "ratio": 3.0},
    {"label": "4:3", "ratio": 4.0 / 3.0},
    {"label": "5:4", "ratio": 5.0 / 4.0},
    {"label": "3:4", "ratio": 3.0 / 4.0},
    {"label": "4:5", "ratio": 4.0 / 5.0},
    {"label": "1:1", "ratio": 1.0},
    {"label": "9:16", "ratio": 9.0 / 16.0},
]

_aspect_idx = _bounded_int(_PERSISTED_SETTINGS.get("aspect_idx", 0), 0, 0, len(ASPECT_OPTIONS) - 1)
_aspect_button_lock = threading.Lock()
_aspect_toggle_rect = (0, 0, 0, 0)
_aspect_popup_visible = False
_aspect_popup_rects = []
_aspect_popup_bounds = None
_aspect_popup_fade = 0.0
_aspect_popup_fade_target = 0.0

# Smooth aspect-ratio transition state
_aspect_ratio_current = ASPECT_OPTIONS[_aspect_idx]["ratio"]  # Animated value
_aspect_ratio_target = ASPECT_OPTIONS[_aspect_idx]["ratio"]   # Desired value
_ASPECT_TRANSITION_SPEED = 8.0  # Exponential ease-out factor (higher = faster)

# --- Grid Overlay (submenu under Aspect Ratio) ---
GRID_OVERLAY_OPTIONS = [
    {"label": "Off",   "key": "off"},
    {"label": "3rds",  "key": "thirds"},
    {"label": "Golden", "key": "golden"},
]
_grid_overlay_idx = _bounded_int(_PERSISTED_SETTINGS.get("grid_overlay_idx", 0), 0, 0, len(GRID_OVERLAY_OPTIONS) - 1)
_grid_button_lock = threading.Lock()

FILM_OPTIONS = [
    {"label": "None", "short": "None", "key": "none"},
    {"label": "Ilford Delta 400", "short": "Delta 400", "key": "delta400"},
    {"label": "Ilford HP5", "short": "HP5", "key": "hp5"},
    {"label": "Kodak Tri-X 400", "short": "Tri-X", "key": "trix400"},
    {"label": "Blue Chrome", "short": "Blue", "key": "blue_chrome"},
]

_film_idx = _bounded_int(_PERSISTED_SETTINGS.get("film_idx", 0), 0, 0, len(FILM_OPTIONS) - 1)
_film_button_lock = threading.Lock()
_film_toggle_rect = (0, 0, 0, 0)
_film_popup_visible = False
_film_popup_rects = []
_film_popup_bounds = None
_film_popup_fade = 0.0
_film_popup_fade_target = 0.0

ISO_OPTIONS = [
    {"label": "AUTO", "short": "AUTO", "gain": None},
    {"label": "100", "short": "100", "gain": 1.0},
    {"label": "200", "short": "200", "gain": 2.0},
    {"label": "400", "short": "400", "gain": 4.0},
    {"label": "600", "short": "600", "gain": 6.0},
    {"label": "800", "short": "800", "gain": 8.0},
    {"label": "1600", "short": "1600", "gain": 16.0},
    {"label": "3200", "short": "3200", "gain": 32.0},
    {"label": "6400", "short": "6400", "gain": 64.0},
    {"label": "Exp Comp", "short": "EV", "gain": "exp_comp"},
]


_iso_idx = _bounded_int(_PERSISTED_SETTINGS.get("iso_idx", 0), 0, 0, len(ISO_OPTIONS) - 1)
_iso_button_lock = threading.Lock()
_iso_toggle_rect = (0, 0, 0, 0)
_iso_popup_visible = False
_iso_popup_rects = []
_iso_popup_bounds = None
_iso_popup_fade = 0.0
_iso_popup_fade_target = 0.0

# --- Exposure Compensation ---
EXP_COMP_OPTIONS = [
    {"label": "-2", "value": -2.0},
    {"label": "-1", "value": -1.0},
    {"label": "0", "value": 0.0},
    {"label": "+1", "value": 1.0},
    {"label": "+2", "value": 2.0},
]
_exp_comp_idx = _bounded_int(_PERSISTED_SETTINGS.get("exp_comp_idx", 2), 0, 0, len(EXP_COMP_OPTIONS) - 1)
_exp_comp_lock = threading.Lock()
_exp_comp_popup_visible = False
_exp_comp_popup_rects = []
_exp_comp_popup_bounds = None
_exp_comp_popup_fade = 0.0
_exp_comp_popup_fade_target = 0.0

# Popup fade animation settings
_POPUP_FADE_SPEED = 8.0  # Fade speed (higher = faster)
_POPUP_CORNER_RADIUS = 12  # Corner radius for rounded popup containers

# --- Shutter speed swipe OSD (on-screen display) ---
_shutter_osd_text = ""          # Label to display (e.g. "1/500")
_shutter_osd_fade = 0.0         # Current opacity 0..1
_shutter_osd_fade_target = 0.0  # Target opacity (1 = showing, 0 = hiding)
_shutter_osd_timer = 0.0        # Countdown (seconds) before starting fade-out
_SHUTTER_OSD_HOLD_S = 0.7       # How long the label stays fully visible
_SHUTTER_OSD_FADE_IN_SPEED = 12.0   # Fast pop-in
_SHUTTER_OSD_FADE_OUT_SPEED = 3.5   # Gentle fade-out

_FILM_CURVE_DATA = {
    "none": {
        "points": [(0, 0), (255, 255)],
        "grain": 0.0,
        "clarity": 0.0,
    },
    "delta400": {
        "points": [(0, 0), (45, 32), (120, 158), (195, 236), (255, 255)],
        "grain": 0.0,
        "clarity": 0.0,
    },
    "hp5": {
        "points": [(0, 4), (35, 28), (120, 160), (205, 240), (255, 255)],
        "grain": 0.0,
        "clarity": 0.0,
    },
    "trix400": {
        "points": [(0, 2), (30, 22), (115, 170), (210, 246), (255, 255)],
        "grain": 0.0,
        "clarity": 0.0,
    },
    "blue_chrome": {
        "points": [(0, 0), (35, 12), (95, 120), (170, 230), (255, 255)],
        "grain": 0.0,
        "clarity": 0.0,
    },
}

_film_rng = np.random.default_rng()
_film_noise_lock = threading.Lock()

# ---- Performance: Pre-allocated noise tile for film grain ----
# Generate noise at a smaller size and tile it for efficiency
_NOISE_TILE_SIZE = 256  # 256x256 tile, tiled as needed
_noise_tile_cache = {}  # keyed by (grain_level,) - cached tiles per grain setting
_NOISE_TILE_CACHE_MAX = 8  # Keep tiles for up to 8 different grain levels

def _get_noise_tile(grain):
    """Get or create a pre-generated noise tile for the given grain level."""
    grain_key = round(grain, 1)  # Round to 0.1 precision for caching
    if grain_key in _noise_tile_cache:
        return _noise_tile_cache[grain_key]
    # Generate new tile using grain_key to ensure cached tile matches the key
    with _film_noise_lock:
        tile = _film_rng.normal(0.0, grain_key, (_NOISE_TILE_SIZE, _NOISE_TILE_SIZE)).astype(np.float32)
    # Evict oldest if cache full
    if len(_noise_tile_cache) >= _NOISE_TILE_CACHE_MAX:
        oldest_key = next(iter(_noise_tile_cache))
        del _noise_tile_cache[oldest_key]
    _noise_tile_cache[grain_key] = tile
    return tile

def _apply_tiled_noise(image, grain):
    """Apply noise using tiled pre-generated noise pattern (faster than per-pixel generation)."""
    h, w = image.shape[:2]
    tile = _get_noise_tile(grain)
    tile_h, tile_w = tile.shape
    # Tile the noise pattern across the image dimensions
    reps_y = (h + tile_h - 1) // tile_h
    reps_x = (w + tile_w - 1) // tile_w
    noise = np.tile(tile, (reps_y, reps_x))[:h, :w]
    # Randomize phase by rolling (cheap operation)
    with _film_noise_lock:
        roll_y = _film_rng.integers(0, tile_h)
        roll_x = _film_rng.integers(0, tile_w)
    noise = np.roll(noise, (roll_y, roll_x), axis=(0, 1))
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# ---- Performance: Pre-allocated buffers for common operations ----
_display_flip_buffer = None  # Reused for flip operations
_scaled_frame_buffer = None  # Reused for frame scaling
# Battery indicator pre-allocated buffers (avoids np.zeros every frame)
_batt_buf_no_bolt = np.zeros((17, 33, 3), dtype=np.uint8)   # 13+4 x 30+3
_batt_buf_with_bolt = np.zeros((17, 49, 3), dtype=np.uint8)  # 13+4 x 30+3+14+2
_bolt_canvas = np.zeros((14, 14, 3), dtype=np.uint8)

ZOOM_LEVELS = [2.0, 4.0]
zoom_active = False
zoom_center_norm = (0.5, 0.5)
_zoom_level_idx = 0
TAP_ZOOM_BARRIER_PAD = 12
_preview_geom = {"x": 0, "y": 0, "w": SCREEN_W, "h": SCREEN_H}

def _choose_preview_size(full_w, full_h, oversample=1.5, max_long_edge=1280):
    """Pick a preview-stream size for the live view.

    The Pi 5's 4" touchscreen is 800x480, so oversampling much beyond ~1.5×
    wastes ISP bandwidth without improving what the user sees.  The
    previous defaults (oversample=2.0, max_long_edge=1920) requested a
    1600-wide lores stream which made the 47 MP IMX492 pipeline unusable.
    """
    if not (full_w and full_h):
        return (SCREEN_W, SCREEN_H)
    aspect = full_w / full_h
    desired_w = min(full_w, int(round(SCREEN_W * oversample)))
    desired_h = int(round(desired_w / aspect))
    if desired_h > full_h:
        desired_h = min(full_h, int(round(SCREEN_H * oversample)))
        desired_w = int(round(desired_h * aspect))
    if desired_w > max_long_edge:
        desired_w = max_long_edge
        desired_h = int(round(desired_w / aspect))
    return (_even(desired_w), _even(desired_h))

def _compute_display_geometry(src_w, src_h):
    if not (src_w and src_h):
        return SCREEN_W, SCREEN_H, 0, 0, max(0, SCREEN_H - BAR_H)
    scale = min(SCREEN_W / src_w, SCREEN_H / src_h)
    disp_w = max(1, int(round(src_w * scale)))
    disp_h = max(1, int(round(src_h * scale)))
    x_off = max((SCREEN_W - disp_w) // 2, 0)
    y_off = max((SCREEN_H - disp_h) // 2, 0)
    bar_y = min(SCREEN_H - BAR_H, max(y_off, y_off + disp_h - BAR_H))
    return disp_w, disp_h, x_off, y_off, bar_y

# ---- Performance: Track last geometry to enable selective canvas clearing ----
_last_canvas_geom = {"x": 0, "y": 0, "w": SCREEN_W, "h": SCREEN_H}

def _clear_canvas_margins(canvas, x_off, y_off, new_w, new_h):
    """Clear only the canvas margins (areas not covered by the frame).

    This is more efficient than clearing the entire canvas when the frame
    covers most of the display area.
    """
    global _last_canvas_geom
    ch, cw = canvas.shape[:2]

    # Calculate margin areas
    # Top margin
    if y_off > 0:
        canvas[0:y_off, :] = 0
    # Bottom margin
    bottom_start = y_off + new_h
    if bottom_start < ch:
        canvas[bottom_start:ch, :] = 0
    # Left margin (only in the frame area, top/bottom already cleared)
    if x_off > 0:
        canvas[y_off:bottom_start, 0:x_off] = 0
    # Right margin
    right_start = x_off + new_w
    if right_start < cw:
        canvas[y_off:bottom_start, right_start:cw] = 0

    _last_canvas_geom = {"x": x_off, "y": y_off, "w": new_w, "h": new_h}

def _largest_ratio_crop_dims(src_w, src_h, ratio):
    if not (src_w and src_h) or ratio <= 0:
        return src_w, src_h
    current = src_w / float(src_h)
    if current > ratio:
        target_w = int(round(src_h * ratio))
        target_h = src_h
    else:
        target_w = src_w
        target_h = int(round(src_w / ratio))
    target_w = max(2, min(src_w, _even(target_w)))
    target_h = max(2, min(src_h, _even(target_h)))
    return target_w, target_h

def _center_crop(frame, target_w, target_h):
    fh, fw = frame.shape[:2]
    target_w = max(1, min(fw, int(target_w)))
    target_h = max(1, min(fh, int(target_h)))
    if target_w == fw and target_h == fh:
        return frame
    x1 = max(0, (fw - target_w) // 2)
    y1 = max(0, (fh - target_h) // 2)
    x2 = x1 + target_w
    y2 = y1 + target_h
    return frame[y1:y2, x1:x2]

def ensure_channels(img, target_channels):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if target_channels == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == target_channels: return img
    if img.shape[2] == 3 and target_channels == 4: return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.shape[2] == 4 and target_channels == 3: return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def blit_add(dst, src, x, y):
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]
    if sw <= 0 or sh <= 0: return
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(dw, x + sw); y2 = min(dh, y + sh)
    if x1 >= x2 or y1 >= y2: return
    sx1 = x1 - x; sy1 = y1 - y
    sx2 = sx1 + (x2 - x1); sy2 = sy1 + (y2 - y1)
    roi = dst[y1:y2, x1:x2]
    src_crop = src[sy1:sy2, sx1:sx2]
    tgt_ch = roi.shape[2] if roi.ndim == 3 else 3
    if roi.ndim == 2 or roi.shape[2] != tgt_ch:
        roi = ensure_channels(roi, tgt_ch)
    if src_crop.ndim == 2 or src_crop.shape[2] != tgt_ch:
        src_crop = ensure_channels(src_crop, tgt_ch)
    cv2.add(roi, src_crop, dst=roi)
    dst[y1:y2, x1:x2] = roi

_shadow_cache = {}  # id(src) -> (shadow_img, src_id) — avoids per-frame GaussianBlur
_SHADOW_CACHE_MAX = 32

def blit_add_with_shadow(dst, src, x, y, shadow_strength=0.7, blur_ksize=7):
    """Additive-blit *src* onto *dst* with a soft dark shadow for readability
    on bright backgrounds.  Uses only fast integer cv2 ops on small ROIs.
    Caches the shadow of *src* keyed on id(src) — safe because cached text
    blocks return the same object on cache hits."""
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]
    if sw <= 0 or sh <= 0: return
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(dw, x + sw); y2 = min(dh, y + sh)
    if x1 >= x2 or y1 >= y2: return
    sx1 = x1 - x; sy1 = y1 - y
    roi = dst[y1:y2, x1:x2]
    src_crop = src[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
    tgt_ch = roi.shape[2] if roi.ndim == 3 else 3
    if roi.ndim == 2 or roi.shape[2] != tgt_ch:
        roi = ensure_channels(roi, tgt_ch)
    if src_crop.ndim == 2 or src_crop.shape[2] != tgt_ch:
        src_crop = ensure_channels(src_crop, tgt_ch)
    # Cache the blurred shadow — GaussianBlur is expensive per-frame
    # Use max-channel luminance so colored text gets a uniform (channel-
    # independent) dark backdrop.  Without this, the colored shadow
    # subtraction + colored text addition nearly cancel out on bright
    # backgrounds, washing out the accent hue.
    _cache_key = (src.shape, src.dtype.str, int(src.sum()))
    _cached = _shadow_cache.get(_cache_key)
    if _cached is not None:
        full_shadow = _cached
    else:
        if src.ndim == 3 and src.shape[2] >= 3:
            _gray = np.max(src[:, :, :3], axis=2)
            _shadow_src = cv2.merge([_gray, _gray, _gray])
        else:
            _shadow_src = src
        full_shadow = cv2.convertScaleAbs(
            cv2.GaussianBlur(_shadow_src, (blur_ksize, blur_ksize), 0),
            alpha=shadow_strength, beta=0)
        if len(_shadow_cache) >= _SHADOW_CACHE_MAX:
            try:
                _shadow_cache.pop(next(iter(_shadow_cache)))
            except Exception:
                _shadow_cache.clear()
        _shadow_cache[_cache_key] = full_shadow
    shadow_crop = full_shadow[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
    if shadow_crop.ndim == 2 or shadow_crop.shape[2] != tgt_ch:
        shadow_crop = ensure_channels(shadow_crop, tgt_ch)
    cv2.subtract(roi, shadow_crop, dst=roi)
    cv2.add(roi, src_crop, dst=roi)
    dst[y1:y2, x1:x2] = roi

def _film_lut_for(key):
    preset = _FILM_CURVE_DATA.get(key)
    if preset is None:
        preset = _FILM_CURVE_DATA["none"]
    lut = preset.get("lut")
    if lut is None:
        points = preset.get("points", [(0, 0), (255, 255)])
        if not points:
            points = [(0, 0), (255, 255)]
        xs = np.array([max(0, min(255, p[0])) for p in points], dtype=np.float32)
        ys = np.array([max(0, min(255, p[1])) for p in points], dtype=np.float32)
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        if xs[0] > 0:
            xs = np.insert(xs, 0, 0.0)
            ys = np.insert(ys, 0, ys[0])
        if xs[-1] < 255:
            xs = np.append(xs, 255.0)
            ys = np.append(ys, ys[-1])
        lut = np.clip(np.interp(np.arange(256), xs, ys), 0, 255).astype(np.uint8)
        preset["lut"] = lut
    return lut

def apply_film_simulation_rgb(frame_rgb, key):
    if key == "none":
        return frame_rgb
    preset = _FILM_CURVE_DATA.get(key)
    if preset is None:
        preset = _FILM_CURVE_DATA["none"]
    lut = _film_lut_for(key)
    gray = _mono_to_gray(frame_rgb)
    toned = cv2.LUT(gray, lut)
    clarity = float(preset.get("clarity", 0.0) or 0.0)
    if clarity > 0:
        sigma = 1.2 + clarity * 1.5
        blur = cv2.GaussianBlur(toned, (0, 0), sigma)
        # Optimized: use cv2.addWeighted with dst parameter to reduce allocations
        alpha = 1.0 + clarity
        beta = -clarity
        cv2.addWeighted(toned, alpha, blur, beta, 0, dst=toned)
    grain = float(preset.get("grain", 0.0) or 0.0)
    if grain > 0:
        # Use optimized tiled noise instead of per-pixel generation
        toned = _apply_tiled_noise(toned, grain)
    rgb = cv2.cvtColor(toned, cv2.COLOR_GRAY2RGB)
    if key == "blue_chrome":
        # Optimized blue_chrome: use integer math with np.int16 to avoid float conversion
        toned_i16 = toned.astype(np.int16)
        # shadow factor: (120 - toned) / 120, scaled to 0-255 range for integer math
        shadow_scaled = np.clip(120 - toned_i16, 0, 120)  # 0-120 range
        # Pre-compute color adjustments scaled by 120 (to divide once at end)
        # blue_boost = 40 * shadow / 120 = shadow * 40 / 120 = shadow / 3
        # red_cut = 12 * shadow / 120 = shadow / 10
        # green_cut = 6 * shadow / 120 = shadow / 20
        rgb_i16 = rgb.astype(np.int16)
        rgb_i16[..., 2] = np.clip(rgb_i16[..., 2] + shadow_scaled // 3, 0, 255)
        rgb_i16[..., 1] = np.clip(rgb_i16[..., 1] - shadow_scaled // 20, 0, 255)
        rgb_i16[..., 0] = np.clip(rgb_i16[..., 0] - shadow_scaled // 10, 0, 255)
        rgb = rgb_i16.astype(np.uint8)
    return rgb

def format_shutter_speed(us):
    return f"1/{int(round(1e6/us))}s" if us and us > 0 else "AUTO"

def _prepare_focus_assist_frame(frame_rgb):
    if frame_rgb is None:
        return None
    preset = _get_focus_preset()
    max_dim_limit = preset[0]  # max_dim from preset
    h, w = frame_rgb.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_dim_limit:
        return frame_rgb
    scale = max_dim_limit / float(max_dim)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

def apply_focus_peaking(frame_rgb, assist_frame=None):
    """Optimized focus peaking using Laplacian edge detection.

    Laplacian (second-order derivative) is used because it responds to the
    rate of change of intensity gradients - sharp in-focus edges produce
    strong responses while blurry out-of-focus edges produce weak responses.
    """
    global _focus_frame_counter, _focus_last_mask, _focus_last_gray

    # Get performance preset: (max_dim, blur_ksize, threshold_offset, skip_frames)
    preset = _get_focus_preset()
    blur_ksize = preset[1]
    threshold_offset = preset[2]
    skip_frames = preset[3]

    # Extract grayscale (monochrome sensor: R=G=B, skip weighted conversion)
    gray = _mono_to_gray(frame_rgb)

    # Frame skipping: reuse cached mask on skipped frames
    _focus_frame_counter += 1
    if skip_frames > 0 and _focus_frame_counter % (skip_frames + 1) != 0:
        if _focus_last_mask is not None:
            # Resize cached mask if needed
            if _focus_last_mask.shape[:2] != frame_rgb.shape[:2]:
                mask = cv2.resize(_focus_last_mask, (frame_rgb.shape[1], frame_rgb.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
                _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
            else:
                mask = _focus_last_mask
            mask_idx = mask > 0
            if np.any(mask_idx):
                view = frame_rgb[mask_idx]
                blended = (
                    (view.astype(np.uint16) * (_FOCUS_BLEND_DEN - _FOCUS_BLEND_NUM)
                     + _FOCUS_COLOR * _FOCUS_BLEND_NUM)
                    // _FOCUS_BLEND_DEN
                ).astype(np.uint8)
                frame_rgb[mask_idx] = blended
        return frame_rgb, gray

    # Use assist frame (already downscaled by _prepare_focus_assist_frame)
    assist_frame = assist_frame if assist_frame is not None else frame_rgb
    assist_gray = _mono_to_gray(assist_frame)

    # Apply Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(assist_gray, (blur_ksize, blur_ksize), 0)

    # Laplacian edge detection - responds strongly to sharp transitions (in-focus)
    # and weakly to gradual transitions (out-of-focus)
    lap = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
    edges = cv2.convertScaleAbs(lap)

    # Adaptive threshold: mean + offset ensures only strong edges are highlighted
    # Higher offset = less sensitive (fewer false positives on soft edges)
    mean_edge = cv2.mean(edges)[0]
    thr = max(30, int(mean_edge + threshold_offset))
    _, mask = cv2.threshold(edges, thr, 255, cv2.THRESH_BINARY)

    # Morphological cleanup: open removes small noise specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _FOCUS_KERNEL_3, iterations=1)

    # Scale mask to frame size using linear interpolation for smooth edges
    if mask.shape[:2] != frame_rgb.shape[:2]:
        mask = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Re-threshold after interpolation to get clean binary mask
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Cache the mask for frame skipping
    _focus_last_mask = mask.copy()

    # Apply green overlay on sharp areas
    mask_idx = mask > 0
    if np.any(mask_idx):
        view = frame_rgb[mask_idx]
        blended = (
            (view.astype(np.uint16) * (_FOCUS_BLEND_DEN - _FOCUS_BLEND_NUM)
             + _FOCUS_COLOR * _FOCUS_BLEND_NUM)
            // _FOCUS_BLEND_DEN
        ).astype(np.uint8)
        frame_rgb[mask_idx] = blended

    return frame_rgb, gray

def estimate_shift(roi_gray):
    """Estimate focus shift using Laplacian variance with optimized thresholds."""
    if roi_gray is None or roi_gray.size == 0:
        return RF_MAX_SHIFT
    # Skip blur for small ROIs (already low resolution)
    if roi_gray.shape[0] < 100:
        lap = cv2.Laplacian(roi_gray, cv2.CV_16S, ksize=3)
    else:
        blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        lap = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
    # Use numpy variance directly (faster than float conversion)
    sharpness = lap.var()
    low_thr = 35.0
    high_thr = 220.0
    norm = (sharpness - low_thr) / max(1.0, high_thr - low_thr)
    norm = max(0.0, min(1.0, norm))
    return int(round((1.0 - norm) * RF_MAX_SHIFT))

def _rf_roi_bounds(frame_shape, roi_size):
    """Calculate ROI bounds for rangefinder."""
    if frame_shape is None or len(frame_shape) < 2:
        return None
    h, w = frame_shape[:2]
    size = int(roi_size)
    size = max(20, min(size, w - 2, h - 2))
    if size <= 0:
        return None
    cx = w // 2
    cy = h // 2
    x1 = max(0, min(w - size, cx - size // 2))
    y1 = max(0, min(h - size, cy - size // 2 + 8))
    return int(x1), int(y1), int(size)

def _update_rangefinder_shift(frame_rgb):
    """Update rangefinder shift with throttling based on performance preset."""
    global _rf_last_update, _rf_shift_px, _rf_shift_smoothed
    if frame_rgb is None:
        return

    # Get preset: (roi_size, interval_ms, smoothing)
    preset = _get_rf_preset()
    roi_size = preset[0]
    interval_ms = preset[1]
    smoothing = preset[2]

    now = time.monotonic()
    if interval_ms > 0:
        if now - _rf_last_update < (interval_ms / 1000.0):
            return

    roi_bounds = _rf_roi_bounds(frame_rgb.shape, roi_size)
    if roi_bounds is None:
        return
    x1, y1, size = roi_bounds
    roi = frame_rgb[y1:y1 + size, x1:x1 + size]
    if roi.size == 0:
        return

    # Monochrome sensor: extract single channel directly
    roi_gray = _mono_to_gray(roi)
    target_shift = estimate_shift(roi_gray)

    # EMA smoothing with preset factor
    _rf_shift_smoothed = (1.0 - smoothing) * _rf_shift_smoothed + smoothing * float(target_shift)
    next_shift = int(round(_rf_shift_smoothed))
    if abs(next_shift - _rf_shift_px) >= RF_SHIFT_DEADBAND:
        _rf_shift_px = next_shift
    _rf_last_update = now

def _draw_rangefinder_overlay(frame_rgb):
    """Draw rangefinder ghost overlay with optimized blending."""
    global _rf_ghost_buffer, _rf_combined_buffer
    if frame_rgb is None:
        return

    # Get ROI size from preset
    preset = _get_rf_preset()
    roi_size = preset[0]

    roi_bounds = _rf_roi_bounds(frame_rgb.shape, roi_size)
    if roi_bounds is None:
        return
    x1, y1, size = roi_bounds
    roi = frame_rgb[y1:y1 + size, x1:x1 + size]
    if roi.size == 0:
        return

    shift = int(_rf_shift_px)
    expected_shape = (size, size, 3)

    # Reuse ghost buffer if size matches
    if _rf_ghost_buffer is None or _rf_ghost_buffer.shape != expected_shape:
        _rf_ghost_buffer = np.zeros(expected_shape, dtype=np.uint8)

    ghost = _rf_ghost_buffer
    if shift == 0:
        np.copyto(ghost, roi)
    else:
        ghost.fill(0)
        if shift > 0:
            ghost[:, shift:] = roi[:, :size - shift]
        else:
            ghost[:, :size + shift] = roi[:, -shift:]

    # Use cv2.addWeighted for efficiency (hardware accelerated on some platforms)
    # First apply white tint to ghost, then blend with original ROI
    ghost_tinted = cv2.addWeighted(ghost, 1.0 - RF_GHOST_TINT,
                                   np.full_like(ghost, 255), RF_GHOST_TINT, 0)
    combined = cv2.addWeighted(roi, 1.0 - RF_GHOST_ALPHA, ghost_tinted, RF_GHOST_ALPHA, 0)

    frame_rgb[y1:y1 + size, x1:x1 + size] = combined
    cv2.rectangle(frame_rgb, (x1, y1), (x1 + size, y1 + size), (230, 230, 230), 1)

def _set_rangefinder_assist(enabled):
    global rangefinder_assist_enabled, focus_peaking_enabled, _rangefinder_prev_peaking
    if enabled == rangefinder_assist_enabled:
        return
    if enabled:
        # Save focus peaking state, then disable it while Rangefinder Assist is active.
        _rangefinder_prev_peaking = focus_peaking_enabled
        focus_peaking_enabled = False
        rangefinder_assist_enabled = True
        _debounced_save_settings()
    else:
        rangefinder_assist_enabled = False
        _debounced_save_settings()
        # Restore focus peaking to the saved state (if any).
        if _rangefinder_prev_peaking is not None:
            focus_peaking_enabled = _rangefinder_prev_peaking
        _rangefinder_prev_peaking = None

if rangefinder_assist_enabled:
    _set_rangefinder_assist(True)

# Derive initial _focus_mode_idx: prefer persisted value, fall back to flags
_focus_mode_idx = _bounded_int(
    _PERSISTED_SETTINGS.get("focus_mode_idx",
        2 if rangefinder_assist_enabled else (1 if focus_peaking_enabled else 0)),
    0, 0, len(FOCUS_MODE_LABELS) - 1
)


def _set_focus_mode(idx):
    """Set focus mode: 0=Off, 1=Focus Peaking, 2=Rangefinder Focus."""
    global _focus_mode_idx, _focus_popup_visible, _focus_popup_rects, _focus_popup_bounds
    global focus_peaking_enabled, rangefinder_assist_enabled
    with _focus_button_lock:
        idx = max(0, min(idx, len(FOCUS_MODE_LABELS) - 1))
        if idx == _focus_mode_idx:
            _focus_popup_visible = False
            _focus_popup_rects = []
            _focus_popup_bounds = None
        else:
            _focus_mode_idx = idx
            _focus_popup_visible = False
            _focus_popup_rects = []
            _focus_popup_bounds = None
    # Apply the corresponding focus state
    if idx == 2:
        # Rangefinder Focus – enable rangefinder, disable peaking
        _set_rangefinder_assist(True)
    elif idx == 1:
        # Focus Peaking – disable rangefinder, enable peaking
        if rangefinder_assist_enabled:
            _set_rangefinder_assist(False)
        focus_peaking_enabled = True
    else:
        # Off – disable both
        if rangefinder_assist_enabled:
            _set_rangefinder_assist(False)
        focus_peaking_enabled = False
    _record_activity(wake=True)
    _debounced_save_settings()


def draw_histogram(gray, height=BAR_H, width=256, rotated=False):
    # Faster than cv2.calcHist for our small UI histogram
    # rotated=True draws histogram upside-down (equivalent to cv2.ROTATE_180)
    if gray is None or gray.size == 0:
        return np.zeros((height, max(1, width), 3), dtype=np.uint8)
    # Downsample for speed; histogram is for UI only
    g = gray[::2, ::2] if gray.shape[0] > 1 and gray.shape[1] > 1 else gray
    hist = np.bincount(g.reshape(-1), minlength=256).astype(np.float32)
    cw = max(1, width)
    comp = cv2.resize(hist.reshape(1, -1), (cw, 1), interpolation=cv2.INTER_AREA).reshape(-1)
    p99 = float(np.percentile(comp, 99))
    disp_max = max(50.0, min(p99, 5000.0))
    hh = height
    img = np.zeros((hh, cw, 3), dtype=np.uint8)
    if disp_max <= 0:
        return img
    scale = (hh - 6) / disp_max
    heights = np.minimum((comp * scale).astype(np.int32), hh - 6)
    valid = heights > 0
    if np.any(valid):
        xs = np.nonzero(valid)[0]
        ys = heights[valid]
        # Vectorized: fill columns using numpy indexing instead of per-bar cv2.line
        if rotated:
            # Bars grow from top, X mirrored
            mxs = cw - 1 - xs
            for mx, bar in zip(mxs, ys):
                img[0:int(bar), int(mx)] = 255
        else:
            for x, bar in zip(xs, ys):
                img[hh - 1 - int(bar):hh, int(x)] = 255
    return img

def draw_battery(dst, percent, bar_y, right_margin=12, left_margin=None, align_top=False):
    h, w = dst.shape[:2]
    iw, ih = 30, 13
    nub_w, nub_h = 3, 5

    if left_margin is not None:
        max_x1 = w - iw - nub_w - max(0, right_margin)
        if max_x1 < 0:
            return None
        x1 = int(round(left_margin))
        x1 = max(0, min(max_x1, x1))
        x2 = x1 + iw
    else:
        x2 = w - right_margin
        x1 = x2 - iw

    if align_top:
        y1 = max(0, int(round(bar_y)))
    else:
        y1 = bar_y + BAR_H - ih
    if percent is None:
        pct = 0
    else:
        pct = int(round(percent))
    cv2.rectangle(dst, (x1,y1), (x1+iw,y1+ih), (255,255,255), 2)
    cv2.rectangle(dst, (x1+iw, y1+(ih-nub_h)//2), (x1+iw+nub_w, y1+(ih+nub_h)//2), (255,255,255), -1)
    ix1, iy1 = x1 + 3, y1 + 3
    ix2 = x1 + iw - 3
    iy2 = y1 + ih - 3
    if pct > 0 and ix2 > ix1:
        interior_w = ix2 - ix1 + 1
        fill_w = max(1, int(round(interior_w * pct / 100.0)))
        cv2.rectangle(dst, (ix1, iy1), (ix1 + fill_w - 1, iy2), (255,255,255), -1)
    return (x1, y1, x1 + iw + nub_w, y1 + ih)


def _icon_button_bounds(dst_shape, top, right_edge, size, margin):
    dst_h, dst_w = dst_shape[:2]
    x2 = min(dst_w - margin, right_edge)
    x1 = max(margin, x2 - size)
    if x1 < margin:
        x1 = margin
        x2 = x1 + size
    y1 = max(margin, top)
    y2 = y1 + size
    if y2 > dst_h - margin:
        y2 = dst_h - margin
        y1 = max(margin, y2 - size)
    return int(x1), int(y1), int(x2), int(y2)


def _draw_rounded_rect(img, pt1, pt2, color, corner_radius, thickness=-1):
    """Draw a rounded rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(corner_radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness == -1:
        # Filled rounded rectangle
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        # Outlined rounded rectangle
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


_rounded_rect_mask_cache = {}
_ROUNDED_RECT_MASK_CACHE_MAX = 32

def _create_rounded_rect_mask(width, height, corner_radius):
    """Create an alpha mask for a rounded rectangle with clean anti-aliased edges."""
    key = (width, height, corner_radius)
    cached = _rounded_rect_mask_cache.get(key)
    if cached is not None:
        return cached
    mask = np.zeros((height, width), dtype=np.uint8)
    r = min(corner_radius, width // 2, height // 2)
    # Fill the main areas
    cv2.rectangle(mask, (r, 0), (width - r, height), 255, -1)
    cv2.rectangle(mask, (0, r), (width, height - r), 255, -1)
    # Fill the corner circles with anti-aliasing
    cv2.circle(mask, (r, r), r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (width - r - 1, r), r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (r, height - r - 1), r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (width - r - 1, height - r - 1), r, 255, -1, cv2.LINE_AA)
    if len(_rounded_rect_mask_cache) >= _ROUNDED_RECT_MASK_CACHE_MAX:
        _rounded_rect_mask_cache.clear()
    _rounded_rect_mask_cache[key] = mask
    return mask


def _update_popup_fades(dt):
    """Update all popup fade animations. Call once per frame with delta time."""
    global _shutter_popup_fade, _shutter_popup_fade_target
    global _aspect_popup_fade, _aspect_popup_fade_target
    global _film_popup_fade, _film_popup_fade_target
    global _iso_popup_fade, _iso_popup_fade_target
    global _focus_popup_fade, _focus_popup_fade_target
    global _exp_comp_popup_fade, _exp_comp_popup_fade_target
    global _multishot_popup_fade
    fade_step = _POPUP_FADE_SPEED * dt

    # Shutter popup fade
    _shutter_popup_fade_target = 1.0 if _shutter_popup_visible else 0.0
    if _shutter_popup_fade < _shutter_popup_fade_target:
        _shutter_popup_fade = min(_shutter_popup_fade + fade_step, 1.0)
    elif _shutter_popup_fade > _shutter_popup_fade_target:
        _shutter_popup_fade = max(_shutter_popup_fade - fade_step, 0.0)

    # Aspect popup fade
    _aspect_popup_fade_target = 1.0 if _aspect_popup_visible else 0.0
    if _aspect_popup_fade < _aspect_popup_fade_target:
        _aspect_popup_fade = min(_aspect_popup_fade + fade_step, 1.0)
    elif _aspect_popup_fade > _aspect_popup_fade_target:
        _aspect_popup_fade = max(_aspect_popup_fade - fade_step, 0.0)

    # Film popup fade
    _film_popup_fade_target = 1.0 if _film_popup_visible else 0.0
    if _film_popup_fade < _film_popup_fade_target:
        _film_popup_fade = min(_film_popup_fade + fade_step, 1.0)
    elif _film_popup_fade > _film_popup_fade_target:
        _film_popup_fade = max(_film_popup_fade - fade_step, 0.0)

    # ISO popup fade
    _iso_popup_fade_target = 1.0 if _iso_popup_visible else 0.0
    if _iso_popup_fade < _iso_popup_fade_target:
        _iso_popup_fade = min(_iso_popup_fade + fade_step, 1.0)
    elif _iso_popup_fade > _iso_popup_fade_target:
        _iso_popup_fade = max(_iso_popup_fade - fade_step, 0.0)

    # Focus mode popup fade
    _focus_popup_fade_target = 1.0 if _focus_popup_visible else 0.0
    if _focus_popup_fade < _focus_popup_fade_target:
        _focus_popup_fade = min(_focus_popup_fade + fade_step, 1.0)
    elif _focus_popup_fade > _focus_popup_fade_target:
        _focus_popup_fade = max(_focus_popup_fade - fade_step, 0.0)

    # Exp Comp popup fade
    _exp_comp_popup_fade_target = 1.0 if _exp_comp_popup_visible else 0.0
    if _exp_comp_popup_fade < _exp_comp_popup_fade_target:
        _exp_comp_popup_fade = min(_exp_comp_popup_fade + fade_step, 1.0)
    elif _exp_comp_popup_fade > _exp_comp_popup_fade_target:
        _exp_comp_popup_fade = max(_exp_comp_popup_fade - fade_step, 0.0)

    # Multishot popup fade (Double Exp / Brenizer)
    _ms_target = 1.0 if _multishot_popup_visible else 0.0
    if _multishot_popup_fade < _ms_target:
        _multishot_popup_fade = min(_multishot_popup_fade + fade_step, 1.0)
    elif _multishot_popup_fade > _ms_target:
        _multishot_popup_fade = max(_multishot_popup_fade - fade_step, 0.0)

    # Smooth aspect-ratio crop transition (exponential ease-out)
    global _aspect_ratio_current
    diff = _aspect_ratio_target - _aspect_ratio_current
    if abs(diff) > 1e-4:
        _aspect_ratio_current += diff * min(1.0, _ASPECT_TRANSITION_SPEED * dt)
    else:
        _aspect_ratio_current = _aspect_ratio_target

    # Icons hide/show fade (minimal mode transition)
    global _icons_fade, _icons_fade_target
    _icons_fade_target = 0.0 if _ui_minimal_mode else 1.0
    icons_step = _ICONS_FADE_SPEED * dt
    if _icons_fade < _icons_fade_target:
        _icons_fade = min(_icons_fade + icons_step, 1.0)
    elif _icons_fade > _icons_fade_target:
        _icons_fade = max(_icons_fade - icons_step, 0.0)

    # Shutter speed swipe OSD fade
    global _shutter_osd_fade, _shutter_osd_fade_target, _shutter_osd_timer
    if _shutter_osd_timer > 0.0:
        _shutter_osd_timer -= dt
        if _shutter_osd_timer <= 0.0:
            _shutter_osd_timer = 0.0
            _shutter_osd_fade_target = 0.0   # start fading out
    if _shutter_osd_fade < _shutter_osd_fade_target:
        osd_step = _SHUTTER_OSD_FADE_IN_SPEED * dt
        _shutter_osd_fade = min(_shutter_osd_fade + osd_step, 1.0)
    elif _shutter_osd_fade > _shutter_osd_fade_target:
        osd_step = _SHUTTER_OSD_FADE_OUT_SPEED * dt
        _shutter_osd_fade = max(_shutter_osd_fade - osd_step, 0.0)


def _blend_popup_to_dst(dst, popup_img, x1, y1, x2, y2, opacity, mask=None):
    """Blend a popup image onto the destination with given opacity and optional alpha mask."""
    if opacity <= 0.0:
        return
    dst_h, dst_w = dst.shape[:2]
    bx1 = max(0, x1)
    by1 = max(0, y1)
    bx2 = min(dst_w, x2)
    by2 = min(dst_h, y2)
    if bx1 >= bx2 or by1 >= by2:
        return
    sx1 = bx1 - x1
    sy1 = by1 - y1
    sx2 = sx1 + (bx2 - bx1)
    sy2 = sy1 + (by2 - by1)

    if mask is not None:
        # Use alpha mask for transparent corners
        mask_roi = mask[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
        # Apply overall opacity to the mask
        mask_roi = mask_roi * opacity
        # Expand mask to 3 channels for broadcasting
        mask_3ch = mask_roi[:, :, np.newaxis]
        roi = dst[by1:by2, bx1:bx2].astype(np.float32)
        popup_roi = popup_img[sy1:sy2, sx1:sx2].astype(np.float32)
        blended = (popup_roi * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
        dst[by1:by2, bx1:bx2] = blended
    elif opacity >= 1.0:
        dst[by1:by2, bx1:bx2] = popup_img[sy1:sy2, sx1:sx2]
    else:
        roi = dst[by1:by2, bx1:bx2].astype(np.float32)
        popup_roi = popup_img[sy1:sy2, sx1:sx2].astype(np.float32)
        blended = (popup_roi * opacity + roi * (1.0 - opacity)).astype(np.uint8)
        dst[by1:by2, bx1:bx2] = blended


def _render_icon_button(dst, bounds, active=False, accent_color=(180, 180, 180), inner_opacity=0.9):
    """Render a circular icon button with clean anti-aliased edges."""
    x1, y1, x2, y2 = bounds
    size = max(1, x2 - x1)
    key = (size, bool(active), tuple(int(c) for c in accent_color), round(float(inner_opacity), 3))
    cached = _ICON_BTN_CACHE.get(key)
    if cached is not None:
        btn, mask = cached
        return btn.copy(), mask.copy()
    # Render at a higher resolution and downscale for smoother edges
    scale = 2
    pad = 4 * scale  # extra padding so anti-aliased fringe isn't clipped
    render_size = size * scale + pad * 2
    btn = np.zeros((render_size, render_size, 3), dtype=np.uint8)
    center = (render_size // 2, render_size // 2)
    radius = size * scale // 2 - 2 * scale
    # Create clean anti-aliased circular mask
    mask = np.zeros((render_size, render_size), dtype=np.uint8)
    cv2.circle(mask, center, radius + 1 * scale, 255, -1, lineType=cv2.LINE_AA)
    # Draw button layers – solid outline style
    cv2.circle(btn, center, radius, (30, 30, 30), -1, cv2.LINE_AA)
    ring_color = accent_color if active else (110, 110, 110)
    cv2.circle(btn, center, radius, ring_color, 2 * scale, cv2.LINE_AA)
    cv2.circle(btn, center, radius - 3 * scale, (16, 16, 16), -1, cv2.LINE_AA)
    if inner_opacity < 1.0:
        overlay = np.full_like(btn, 255)
        cv2.addWeighted(overlay, 1.0 - inner_opacity, btn, inner_opacity, 0, dst=btn)
    # Downscale for final output with clean edges
    btn = cv2.resize(btn, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    # Convert mask to normalized float for alpha blending
    mask = mask.astype(np.float32) / 255.0
    # Cache the base button so we don't rebuild every frame
    if len(_ICON_BTN_CACHE) >= _ICON_BTN_CACHE_MAX:
        try:
            _ICON_BTN_CACHE.pop(next(iter(_ICON_BTN_CACHE)))
        except Exception:
            _ICON_BTN_CACHE.clear()
    _ICON_BTN_CACHE[key] = (btn.copy(), mask.copy())
    return btn, mask


def _draw_aspect_icon(canvas, label, ratio, active):
    """Refined aspect ratio icon with clean frame and corner brackets."""
    # Try PNG icon first
    color = UI_TEXT_PRIMARY if active else UI_TEXT_DIM
    if _place_png_icon(canvas, "aspect", color):
        return
    h, w = canvas.shape[:2]
    thickness = max(2, w // 26)
    color = UI_TEXT_PRIMARY if active else UI_TEXT_DIM
    inset = int(min(w, h) * 0.20)
    avail_w = w - 2 * inset
    avail_h = h - 2 * inset
    target_w, target_h = avail_w, avail_h
    if ratio > 0:
        target_w = avail_w
        target_h = int(round(target_w / ratio))
        if target_h > avail_h:
            target_h = avail_h
            target_w = int(round(target_h * ratio))
    x1 = (w - target_w) // 2
    y1 = (h - target_h) // 2
    x2 = x1 + target_w
    y2 = y1 + target_h
    # Clean frame outline
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    # Corner accent marks (subtle ticks)
    tick = max(4, int(min(target_w, target_h) * 0.22))
    tick_color = color
    # Top-left
    cv2.line(canvas, (x1, y1 + tick), (x1, y1), tick_color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1), (x1 + tick, y1), tick_color, thickness, cv2.LINE_AA)
    # Top-right
    cv2.line(canvas, (x2 - tick, y1), (x2, y1), tick_color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2, y1 + tick), tick_color, thickness, cv2.LINE_AA)
    # Bottom-left
    cv2.line(canvas, (x1, y2 - tick), (x1, y2), tick_color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1 + tick, y2), tick_color, thickness, cv2.LINE_AA)
    # Bottom-right
    cv2.line(canvas, (x2 - tick, y2), (x2, y2), tick_color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2, y2 - tick), tick_color, thickness, cv2.LINE_AA)


def _draw_film_icon(canvas, short_label, active):
    # Try PNG icon first
    color = UI_TEXT_PRIMARY if active else UI_TEXT_DIM
    if _place_png_icon(canvas, "film", color):
        return
    h, w = canvas.shape[:2]
    edge_color = UI_TEXT_PRIMARY if active else UI_TEXT_DIM
    fill_color = (50, 50, 50)
    strip_height = int(h * 0.55)
    y1 = (h - strip_height) // 2
    y2 = y1 + strip_height
    edge_w = max(6, w // 7)
    # draw sprocket edges
    cv2.rectangle(canvas, (w // 2 - edge_w - (w // 5), y1 - 6), (w // 2 - (w // 5), y2 + 6), edge_color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (w // 2 + (w // 5), y1 - 6), (w // 2 + edge_w + (w // 5), y2 + 6), edge_color, -1, cv2.LINE_AA)
    inner_x1 = w // 2 - (w // 5)
    inner_x2 = w // 2 + (w // 5)
    cv2.rectangle(canvas, (inner_x1, y1), (inner_x2, y2), fill_color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (inner_x1, y1), (inner_x2, y2), edge_color, 2, cv2.LINE_AA)
    hole_count = 5
    hole_h = max(4, int((strip_height - 12) / (hole_count * 1.1)))
    hole_w = max(4, edge_w - 4)
    gap = 0 if hole_count <= 1 else max(0, int((strip_height - hole_h * hole_count) / (hole_count - 1)))
    for idx in range(hole_count):
        top = y1 + 6 + idx * (hole_h + gap)
        bottom = top + hole_h
        cv2.rectangle(canvas, (inner_x1 - hole_w - 4, top), (inner_x1 - 4, bottom), (30, 30, 30), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (inner_x2 + 4, top), (inner_x2 + hole_w + 4, bottom), (30, 30, 30), -1, cv2.LINE_AA)
    # add perforation dividers
    divider_color = (120, 120, 120)
    for offset in (-1, 1):
        cx = inner_x1 if offset < 0 else inner_x2
        cv2.line(canvas, (cx + offset * 2, y1 - 6), (cx + offset * 2, y2 + 6), divider_color, 1, cv2.LINE_AA)


def _draw_wifi_icon(canvas, active):
    """Refined WiFi icon with clean signal arcs and balanced proportions."""
    color = UI_TEXT_PRIMARY if active else UI_INACTIVE
    if _place_png_icon(canvas, "wifi", color):
        return
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2 + 4
    max_r = int(min(w, h) * 0.38)
    color = UI_TEXT_PRIMARY if active else UI_INACTIVE
    thickness = max(2, w // 28)
    # Draw three concentric arcs with consistent spacing
    for frac in (1.0, 0.65, 0.32):
        radius = int(max_r * frac)
        if radius > thickness:
            cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, -135, -45, color, thickness, cv2.LINE_AA)
    # Center dot
    dot_r = max(3, w // 18)
    cv2.circle(canvas, (cx, cy), dot_r, color, -1, lineType=cv2.LINE_AA)


def _draw_flip_icon(canvas, active):
    color = UI_ACCENT_SYSTEM if active else UI_INACTIVE
    if _place_png_icon(canvas, "flip", color):
        return
    h, w = canvas.shape[:2]
    color = UI_ACCENT_SYSTEM if active else UI_INACTIVE
    thickness = max(3, w // 16)
    mid_y = h // 2
    inset = max(10, w // 6)
    bar_len = w - 2 * inset
    cv2.line(canvas, (inset, mid_y), (inset + bar_len, mid_y), color, thickness, cv2.LINE_AA)
    tip = max(6, thickness * 2)
    left_tip = np.array(
        [[inset, mid_y], [inset + tip, mid_y - tip], [inset + tip, mid_y + tip]], dtype=np.int32
    )
    right_tip = np.array(
        [[inset + bar_len, mid_y], [inset + bar_len - tip, mid_y - tip], [inset + bar_len - tip, mid_y + tip]],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [left_tip, right_tip], color, lineType=cv2.LINE_AA)
    stem = max(2, thickness - 1)
    span = max(12, h // 3)
    cv2.line(canvas, (w // 2, mid_y - span // 2), (w // 2, mid_y + span // 2), color, stem, cv2.LINE_AA)


def _draw_charge_icon(canvas, active):
    """Refined lightning bolt icon with clean geometric shape."""
    color = UI_ACCENT_EXPOSURE if active else UI_INACTIVE
    if _place_png_icon(canvas, "charge", color):
        return
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    color = UI_ACCENT_EXPOSURE if active else UI_INACTIVE
    outline = (245, 180, 90) if active else (170, 170, 170)
    # Scale factors based on canvas size
    sx = w / 68.0
    sy = h / 68.0
    # Clean, balanced lightning bolt shape
    pts = np.array([
        (cx + int(-2 * sx), cy + int(-20 * sy)),
        (cx + int(6 * sx), cy + int(-20 * sy)),
        (cx + int(2 * sx), cy + int(-4 * sy)),
        (cx + int(10 * sx), cy + int(-4 * sy)),
        (cx + int(-4 * sx), cy + int(20 * sy)),
        (cx + int(0 * sx), cy + int(2 * sy)),
        (cx + int(-8 * sx), cy + int(2 * sy)),
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [pts], True, outline, max(1, int(1.5 * sx)), cv2.LINE_AA)


def _draw_shutter_icon(canvas, manual_selected, popup_visible):
    """Shutter speed text icon centered within the button."""
    # Try PNG icon first
    if manual_selected:
        color = UI_ACCENT_EXPOSURE
    elif popup_visible:
        color = UI_TEXT_PRIMARY
    else:
        color = UI_TEXT_DIM
    if _place_png_icon(canvas, "shutter", color):
        return
    h, w = canvas.shape[:2]
    text = "S/S"
    font_size = max(14, int(w * 0.34))
    tw, th = _ui_measure_text(text, font_size)
    x = (w - tw) // 2
    y = (h - th) // 2
    if manual_selected:
        _ui_draw_text(canvas, text, x, y, font_size, (240, 225, 150), outline_bgr=(0, 0, 0))
    else:
        color = (215, 215, 215) if popup_visible else (165, 165, 165)
        _ui_draw_text(canvas, text, x, y, font_size, color, outline_bgr=(0, 0, 0))


def _draw_iso_icon(canvas, manual_selected, popup_visible):
    """ISO text icon centered within the button."""
    # Try PNG icon first
    if manual_selected:
        color = UI_ACCENT_EXPOSURE
    elif popup_visible:
        color = UI_TEXT_PRIMARY
    else:
        color = UI_TEXT_DIM
    if _place_png_icon(canvas, "iso", color):
        return
    h, w = canvas.shape[:2]
    text = "ISO"
    font_size = max(14, int(w * 0.34))
    tw, th = _ui_measure_text(text, font_size)
    x = (w - tw) // 2
    y = (h - th) // 2
    if manual_selected:
        _ui_draw_text(canvas, text, x, y, font_size, (235, 215, 135), outline_bgr=(0, 0, 0))
    else:
        color = (215, 215, 215) if popup_visible else (165, 165, 165)
        _ui_draw_text(canvas, text, x, y, font_size, color, outline_bgr=(0, 0, 0))


def _draw_icon_to_dst(dst, img, mask, bounds):
    x1, y1, x2, y2 = bounds
    # Check if a finger is currently pressing this button
    pressed = False
    with _touch_down_lock:
        pos = _touch_down_pos
    if pos is not None:
        px, py = pos
        if x1 <= px < x2 and y1 <= py < y2:
            pressed = True
    img = cv2.rotate(img, cv2.ROTATE_180)
    mask = cv2.rotate(mask, cv2.ROTATE_180)
    clip_x1 = max(0, x1)
    clip_y1 = max(0, y1)
    clip_x2 = min(dst.shape[1], x2)
    clip_y2 = min(dst.shape[0], y2)
    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return None
    sx1 = clip_x1 - x1
    sy1 = clip_y1 - y1
    sx2 = sx1 + (clip_x2 - clip_x1)
    sy2 = sy1 + (clip_y2 - clip_y1)
    roi = dst[clip_y1:clip_y2, clip_x1:clip_x2]
    mask_roi = mask[sy1:sy2, sx1:sx2]
    if mask_roi.ndim == 2:
        alpha = mask_roi[..., None]
    else:
        alpha = mask_roi
    icon_roi = img[sy1:sy2, sx1:sx2].astype(np.float32)
    if pressed:
        icon_roi = np.minimum(icon_roi * ICON_PRESSED_BRIGHTEN, 255.0)
    dst_roi = roi.astype(np.float32)
    opacity = ICON_PRESSED_OPACITY if pressed else ICON_GLOBAL_OPACITY
    # Apply icons fade for smooth minimal-mode transition
    opacity *= _icons_fade
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0) * opacity
    blended = (icon_roi * alpha + dst_roi * (1.0 - alpha)).astype(np.uint8)
    dst[clip_y1:clip_y2, clip_x1:clip_x2] = blended
    return (clip_x1, clip_y1, clip_x2, clip_y2)


def draw_flip_button(dst, active, top, right_edge, size=72, margin=16):
    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    icon, mask = _render_icon_button(dst, bounds, active=active, accent_color=UI_ACCENT_SYSTEM)
    _draw_flip_icon(icon, active)
    return _draw_icon_to_dst(dst, icon, mask, bounds)


def draw_charge_button(dst, active, top, right_edge, size=72, margin=16):
    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    icon, mask = _render_icon_button(dst, bounds, active=active, accent_color=UI_ACCENT_EXPOSURE)
    _draw_charge_icon(icon, active)
    return _draw_icon_to_dst(dst, icon, mask, bounds)


def _draw_sleep_icon(canvas, active):
    """Refined crescent moon icon with elegant proportions."""
    color = UI_ACCENT_SYSTEM if active else UI_INACTIVE
    if _place_png_icon(canvas, "sleep", color):
        return
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(w, h) * 0.26)
    color = UI_ACCENT_SYSTEM if active else UI_INACTIVE
    # Main moon circle
    cv2.circle(canvas, (cx - 2, cy), r, color, -1, cv2.LINE_AA)
    # Refined cutout for elegant crescent shape
    cut_r = int(r * 0.82)
    cut_offset_x = int(r * 0.55)
    cut_offset_y = int(r * -0.15)
    cv2.circle(canvas, (cx - 2 + cut_offset_x, cy + cut_offset_y), cut_r, (0, 0, 0), -1, cv2.LINE_AA)

def draw_sleep_button(dst, active, top, right_edge, size=72, margin=16):
    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    icon, mask = _render_icon_button(dst, bounds, active=active, accent_color=UI_ACCENT_SYSTEM, inner_opacity=0.85)
    _draw_sleep_icon(icon, active)
    return _draw_icon_to_dst(dst, icon, mask, bounds)


def _draw_exit_icon(canvas):
    """Refined exit icon with clean door and arrow design."""
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    thickness = max(2, w // 24)
    color = (220, 220, 220)
    # Compact door frame
    door_w = int(min(w, h) * 0.32)
    door_h = int(min(w, h) * 0.48)
    door_x1 = cx - door_w // 2 - int(w * 0.08)
    door_y1 = cy - door_h // 2
    door_x2 = door_x1 + door_w
    door_y2 = door_y1 + door_h
    # Clean door frame (three sides)
    cv2.line(canvas, (door_x1, door_y1), (door_x2, door_y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (door_x1, door_y1), (door_x1, door_y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (door_x1, door_y2), (door_x2, door_y2), color, thickness, cv2.LINE_AA)
    # Right exit arrow
    arrow_start = cx - int(w * 0.02)
    arrow_end = cx + int(w * 0.22)
    cv2.line(canvas, (arrow_start, cy), (arrow_end, cy), color, thickness, cv2.LINE_AA)
    # Arrowhead
    head_size = max(4, int(w * 0.10))
    cv2.line(canvas, (arrow_end, cy), (arrow_end - head_size, cy - head_size), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (arrow_end, cy), (arrow_end - head_size, cy + head_size), color, thickness, cv2.LINE_AA)


def _draw_x_icon(canvas):
    """Refined X close icon with balanced proportions."""
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    thickness = max(2, w // 18)
    color = (225, 225, 225)
    # Balanced X size
    x_size = int(min(w, h) * 0.24)
    # Draw clean X with consistent stroke
    cv2.line(canvas, (cx - x_size, cy - x_size), (cx + x_size, cy + x_size), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (cx - x_size, cy + x_size), (cx + x_size, cy - x_size), color, thickness, cv2.LINE_AA)


def draw_x_exit_button(dst, top, right_edge, size=72, margin=16):
    """Draw an X exit button for charging mode."""
    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    icon, mask = _render_icon_button(dst, bounds, active=True, accent_color=(200, 200, 200), inner_opacity=0.85)
    _draw_x_icon(icon)
    return _draw_icon_to_dst(dst, icon, mask, bounds)


def _draw_trash_icon(canvas):
    """Draw a trash can icon centered on the canvas."""
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    sx = w / 68.0
    sy = h / 68.0
    color = (180, 120, 120)
    thickness = max(2, int(1.8 * sx))
    # Lid
    lid_y = cy + int(-16 * sy)
    lid_left = cx + int(-12 * sx)
    lid_right = cx + int(12 * sx)
    cv2.line(canvas, (lid_left, lid_y), (lid_right, lid_y), color, thickness, cv2.LINE_AA)
    # Handle on lid
    handle_left = cx + int(-5 * sx)
    handle_right = cx + int(5 * sx)
    handle_top = cy + int(-20 * sy)
    cv2.line(canvas, (handle_left, lid_y), (handle_left, handle_top), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (handle_left, handle_top), (handle_right, handle_top), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (handle_right, handle_top), (handle_right, lid_y), color, thickness, cv2.LINE_AA)
    # Body
    body_top = cy + int(-13 * sy)
    body_bot = cy + int(18 * sy)
    body_left = cx + int(-10 * sx)
    body_right = cx + int(10 * sx)
    taper = int(2 * sx)
    pts = np.array([
        (body_left, body_top),
        (body_right, body_top),
        (body_right - taper, body_bot),
        (body_left + taper, body_bot),
    ], dtype=np.int32)
    cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)
    # Vertical lines inside body
    line_thickness = max(1, int(1.2 * sx))
    for offset in [-4, 0, 4]:
        lx = cx + int(offset * sx)
        cv2.line(canvas, (lx, body_top + int(4 * sy)), (lx, body_bot - int(3 * sy)), color, line_thickness, cv2.LINE_AA)


def draw_trash_button(dst, top, right_edge, size=72, margin=16):
    """Draw a trash/delete button for charging mode."""
    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    icon, mask = _render_icon_button(dst, bounds, active=True, accent_color=UI_ACCENT_DANGER, inner_opacity=0.85)
    _draw_trash_icon(icon)
    return _draw_icon_to_dst(dst, icon, mask, bounds)


def _delete_pictures_files():
    """Delete all files in the Pictures folder."""
    deleted = 0
    errors = 0
    try:
        for entry in os.scandir(LOCAL_PICTURES_DIR):
            if entry.is_file():
                try:
                    os.remove(entry.path)
                    deleted += 1
                except Exception:
                    errors += 1
    except Exception as exc:
        print(f"Error scanning {LOCAL_PICTURES_DIR}: {exc}")
    return deleted, errors


def _draw_delete_confirm_overlay(canvas):
    """Draw a confirmation overlay for deleting all pictures, rotated 180° to
    match the rest of the charging-mode UI."""
    global _delete_confirm_yes_rect, _delete_confirm_no_rect
    h, w = canvas.shape[:2]
    # Semi-transparent dark overlay
    cv2.addWeighted(canvas, 0.3, np.zeros_like(canvas), 0.7, 0, dst=canvas)

    # Count files to show in message
    file_count = 0
    try:
        for entry in os.scandir(LOCAL_PICTURES_DIR):
            if entry.is_file():
                file_count += 1
    except Exception:
        pass

    # Draw dialog into a temporary buffer so we can rotate it 180°
    dialog = np.zeros((h, w, 3), dtype=np.uint8)

    # Message box dimensions
    box_w, box_h = 500, 200
    bx1 = (w - box_w) // 2
    by1 = (h - box_h) // 2
    bx2 = bx1 + box_w
    by2 = by1 + box_h

    # Draw message box background
    _draw_rounded_rect(dialog, (bx1, by1), (bx2, by2), (40, 40, 40), 16, -1)
    _draw_rounded_rect(dialog, (bx1, by1), (bx2, by2), (180, 130, 130), 16, 2)

    # Title
    title = "Delete All Pictures?"
    title_font = 22
    tw, th = _ui_measure_text(title, title_font)
    tx = bx1 + (box_w - tw) // 2
    ty = by1 + 30

    # Subtitle with file count
    sub = f"{file_count} file{'s' if file_count != 1 else ''} will be permanently deleted."
    sub_font = 16
    sw, sh = _ui_measure_text(sub, sub_font)
    sx = bx1 + (box_w - sw) // 2
    sy = by1 + 70

    # Buttons
    btn_w, btn_h = 140, 50
    gap = 40
    total_btn_w = btn_w * 2 + gap
    btn_y1 = by2 - 70
    btn_y2 = btn_y1 + btn_h

    # Cancel button (left)
    cancel_x1 = bx1 + (box_w - total_btn_w) // 2
    cancel_x2 = cancel_x1 + btn_w
    _draw_rounded_rect(dialog, (cancel_x1, btn_y1), (cancel_x2, btn_y2), (80, 80, 80), 10, -1)
    _draw_rounded_rect(dialog, (cancel_x1, btn_y1), (cancel_x2, btn_y2), (160, 160, 160), 10, 2)
    c_text = "Cancel"
    btn_font = 18
    cw, ch = _ui_measure_text(c_text, btn_font)
    c_tx = cancel_x1 + (btn_w - cw) // 2
    c_ty = btn_y1 + (btn_h - ch) // 2

    # Delete button (right)
    del_x1 = cancel_x2 + gap
    del_x2 = del_x1 + btn_w
    _draw_rounded_rect(dialog, (del_x1, btn_y1), (del_x2, btn_y2), (140, 50, 50), 10, -1)
    _draw_rounded_rect(dialog, (del_x1, btn_y1), (del_x2, btn_y2), (200, 100, 100), 10, 2)
    d_text = "Delete"
    dw, dh = _ui_measure_text(d_text, btn_font)
    d_tx = del_x1 + (btn_w - dw) // 2
    d_ty = btn_y1 + (btn_h - dh) // 2

    # Draw all text using TrueType UI font
    _ui_draw_text(dialog, title, tx, ty, title_font, (160, 160, 220))
    _ui_draw_text(dialog, sub, sx, sy, sub_font, (200, 200, 200))
    _ui_draw_text(dialog, c_text, c_tx, c_ty, btn_font, (220, 220, 220))
    _ui_draw_text(dialog, d_text, d_tx, d_ty, btn_font, (220, 220, 255))

    # Rotate the dialog 180° to match the inverted display
    dialog = cv2.rotate(dialog, cv2.ROTATE_180)

    # Composite rotated dialog onto the dimmed canvas (non-black pixels)
    mask = np.any(dialog > 0, axis=2)
    canvas[mask] = dialog[mask]

    # Transform button rects to their 180°-rotated canvas positions
    _delete_confirm_no_rect = (w - cancel_x2, h - btn_y2, w - cancel_x1, h - btn_y1)
    _delete_confirm_yes_rect = (w - del_x2, h - btn_y2, w - del_x1, h - btn_y1)


def draw_exit_button(dst, top, right_edge, size=72, margin=16):
    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    icon, mask = _render_icon_button(dst, bounds, active=True, accent_color=(200, 200, 200), inner_opacity=0.85)
    _draw_exit_icon(icon)
    return _draw_icon_to_dst(dst, icon, mask, bounds)


def _terminate_application():
    try:
        subprocess.run(["pkill", "-f", "wlf8.py"])
    except Exception as exc:
        print("Exit button error:", exc)
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def _point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x < x2 and y1 <= y < y2

def _point_in_rect_padded(x, y, rect, pad=0):
    if rect is None:
        return False
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return False
    pad = max(0, int(pad))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(SCREEN_W, x2 + pad)
    y2 = min(SCREEN_H, y2 + pad)
    return x1 <= x < x2 and y1 <= y < y2

def _apply_wifi_state(desired_on):
    state = "on" if desired_on else "off"
    try:
        subprocess.call(
            ["nmcli", "radio", "wifi", state],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print("Wi-Fi toggle error:", exc)

_wifi_applying = False

def _toggle_wifi():
    global wifi_enabled, _wifi_applying
    with _wifi_lock:
        wifi_enabled = not wifi_enabled
        desired = wifi_enabled
    _debounced_save_settings()
    if not _wifi_applying:
        _wifi_applying = True
        def _do_apply(state):
            global _wifi_applying
            try:
                _apply_wifi_state(state)
            finally:
                _wifi_applying = False
        threading.Thread(target=_do_apply, args=(desired,), daemon=True).start()

def draw_wifi_button(dst, enabled, top, right_edge, size=72, margin=16):
    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    accent = UI_ACCENT_CREATIVE if enabled else UI_INACTIVE
    icon, mask = _render_icon_button(dst, bounds, active=enabled, accent_color=accent)
    _draw_wifi_icon(icon, enabled)
    return _draw_icon_to_dst(dst, icon, mask, bounds)


def _draw_double_exposure_icon(canvas, active, waiting):
    """Refined double exposure icon with clean overlapping frames."""
    # Try PNG icon first
    color = UI_ACCENT_CREATIVE if active else UI_INACTIVE
    if _place_png_icon(canvas, "double_exposure", color):
        return
    h, w = canvas.shape[:2]
    back_color = (210, 175, 140) if active else (160, 160, 170)
    front_color = (235, 190, 140) if waiting else (185, 185, 200)
    thickness = max(2, w // 24)
    # Back frame (slightly offset)
    pad = int(min(w, h) * 0.20)
    offset = int(min(w, h) * 0.10)
    bx1, by1 = pad, pad + offset
    bx2, by2 = w - pad - offset, h - pad
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), back_color, thickness, cv2.LINE_AA)
    # Front frame (overlapping)
    fx1, fy1 = pad + offset, pad
    fx2, fy2 = w - pad, h - pad - offset
    cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), front_color, thickness, cv2.LINE_AA)


def draw_double_exposure_button(dst, enabled, waiting, top, right_edge, size=72, margin=16):
    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    accent = UI_ACCENT_CREATIVE if enabled or waiting else UI_INACTIVE
    icon, mask = _render_icon_button(dst, bounds, active=enabled or waiting, accent_color=accent)
    _draw_double_exposure_icon(icon, enabled or waiting, waiting)
    return _draw_icon_to_dst(dst, icon, mask, bounds)


def _draw_rangefinder_icon(canvas, active):
    """Refined rangefinder icon with focus bracket corners."""
    color = UI_ACCENT_FOCUS if active else UI_INACTIVE
    if _place_png_icon(canvas, "rangefinder", color):
        return
    h, w = canvas.shape[:2]
    color = UI_ACCENT_FOCUS if active else UI_INACTIVE
    thickness = max(2, w // 24)
    pad = int(min(w, h) * 0.22)
    corner_len = int(min(w, h) * 0.18)
    x1, y1 = pad, pad
    x2, y2 = w - pad, h - pad
    # Top-left corner bracket
    cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
    # Top-right corner bracket
    cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
    # Bottom-left corner bracket
    cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
    # Bottom-right corner bracket
    cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)
    # Center focus point
    cx, cy = w // 2, h // 2
    dot_r = max(2, w // 22)
    cv2.circle(canvas, (cx, cy), dot_r, color, -1, lineType=cv2.LINE_AA)


def draw_rangefinder_button(dst, active, top, right_edge, size=72, margin=16):
    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    accent = UI_ACCENT_FOCUS if active else UI_INACTIVE
    icon, mask = _render_icon_button(dst, bounds, active=active, accent_color=accent)
    _draw_rangefinder_icon(icon, active)
    return _draw_icon_to_dst(dst, icon, mask, bounds)


def _render_popup_grid(dst, bounds, labels, active_idx, fade_opacity, popup_visible,
                       cols=2, font_size=18, active_color=(235, 165, 75),
                       position="left", margin=16):
    """Generic popup grid renderer used by all control popups.

    Args:
        dst: Canvas to render onto.
        bounds: (x1, y1, x2, y2) of the toggle button.
        labels: List of strings (option labels to display).
        active_idx: Currently selected index.
        fade_opacity: Current fade level (0.0-1.0).
        popup_visible: Whether popup is logically open.
        cols: Number of grid columns.
        font_size: Font size for option text.
        active_color: BGR accent color for the selected option.
        position: "left", "right", or "center" relative to the button.
        margin: Screen margin in pixels.

    Returns:
        (popup_rects, popup_bounds) where popup_rects is [(idx, rect), ...].
    """
    if not (popup_visible or fade_opacity > 0.01):
        return [], None

    dst_h, dst_w = dst.shape[:2]
    option_pad_x, option_pad_y = 14, 10
    panel_pad_x, panel_pad_y = 14, 14
    gap_x, gap_y = 8, 8
    corner_r = _POPUP_CORNER_RADIUS
    btn_corner_r = 6

    metrics = [_ui_measure_text(lbl, font_size) for lbl in labels]
    max_txt_w = max(m[0] for m in metrics)
    max_txt_h = max(m[1] for m in metrics)
    btn_w = max_txt_w + option_pad_x * 2
    btn_h = max_txt_h + option_pad_y * 2
    rows = (len(labels) + cols - 1) // cols
    panel_w = cols * btn_w + (cols - 1) * gap_x + 2 * panel_pad_x
    panel_h = rows * btn_h + (rows - 1) * gap_y + 2 * panel_pad_y

    # Vertical: prefer above the button, fall back to below
    gap = 12
    space_above = bounds[1]
    space_below = dst_h - bounds[3]
    if space_above >= panel_h + gap and space_above >= space_below:
        panel_y = max(margin, bounds[1] - gap - panel_h)
    else:
        panel_y = min(dst_h - panel_h - margin, bounds[3] + gap)

    # Horizontal positioning
    if position == "right":
        panel_x = min(bounds[2] + gap, dst_w - panel_w - margin)
    elif position == "center":
        panel_x = max(margin, (dst_w - panel_w) // 2)
    else:  # "left"
        panel_x = max(margin, min(bounds[0] - gap - panel_w, dst_w - panel_w - margin))
    panel_x = max(margin, panel_x)
    panel_x2 = panel_x + panel_w
    panel_y2 = panel_y + panel_h

    popup_img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    popup_mask = _create_rounded_rect_mask(panel_w, panel_h, corner_r)
    _draw_rounded_rect(popup_img, (0, 0), (panel_w, panel_h), (22, 22, 22), corner_r)
    _draw_rounded_rect(popup_img, (0, 0), (panel_w - 1, panel_h - 1), (180, 180, 180), corner_r, 2)

    popup_rects = []
    text_items = []
    for idx, label in enumerate(labels):
        col = idx % cols
        row = idx // cols
        local_x1 = panel_pad_x + col * (btn_w + gap_x)
        local_y1 = panel_pad_y + row * (btn_h + gap_y)
        local_x2 = local_x1 + btn_w
        local_y2 = local_y1 + btn_h

        btn_color = active_color if (idx == active_idx) else (55, 55, 55)
        _draw_rounded_rect(popup_img, (local_x1, local_y1), (local_x2, local_y2), btn_color, btn_corner_r)
        _draw_rounded_rect(popup_img, (local_x1, local_y1), (local_x2 - 1, local_y2 - 1), (140, 140, 140), btn_corner_r, 1)
        txt_w, txt_h = _ui_measure_text(label, font_size)
        tx = local_x1 + (btn_w - txt_w) // 2
        ty = local_y1 + (btn_h - txt_h) // 2
        text_items.append((label, tx, ty, (255, 255, 255)))

        rot_x1 = panel_x + (panel_w - local_x2)
        rot_y1 = panel_y + (panel_h - local_y2)
        rot_x2 = panel_x + (panel_w - local_x1)
        rot_y2 = panel_y + (panel_h - local_y1)
        popup_rects.append((idx, (rot_x1, rot_y1, rot_x2, rot_y2)))

    _ui_draw_text_batch(popup_img, text_items, font_size, outline_bgr=(0, 0, 0))
    popup_img = cv2.rotate(popup_img, cv2.ROTATE_180)
    popup_mask = cv2.rotate(popup_mask, cv2.ROTATE_180)
    _blend_popup_to_dst(dst, popup_img, panel_x, panel_y, panel_x2, panel_y2, fade_opacity, popup_mask)
    return popup_rects, (panel_x, panel_y, panel_x2, panel_y2)


def _draw_focus_controls(dst, top, right_edge, size=72, margin=16):
    """Draw focus mode button with popup menu (Peaking Off / Focus Peaking / Rangefinder)."""
    global _focus_toggle_rect, _focus_popup_rects, _focus_popup_bounds
    with _focus_button_lock:
        active_idx = _focus_mode_idx
        popup_visible = _focus_popup_visible

    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    is_active = active_idx > 0
    accent = UI_ACCENT_FOCUS if is_active else UI_INACTIVE
    icon, mask = _render_icon_button(dst, bounds, active=is_active or popup_visible, accent_color=accent)
    _draw_rangefinder_icon(icon, is_active)
    drawn_rect = _draw_icon_to_dst(dst, icon, mask, bounds)

    popup_rects, popup_bounds = _render_popup_grid(
        dst, bounds, FOCUS_MODE_LABELS, active_idx, _focus_popup_fade, popup_visible,
        cols=1, font_size=18, active_color=UI_ACCENT_FOCUS, position="right", margin=margin,
    )

    with _focus_button_lock:
        _focus_toggle_rect = drawn_rect if drawn_rect else bounds
        if popup_visible:
            _focus_popup_rects = popup_rects
            _focus_popup_bounds = popup_bounds
        else:
            _focus_popup_rects = []
            _focus_popup_bounds = None

    return drawn_rect if drawn_rect else bounds


with _wifi_lock:
    wifi_enabled = _bool_value(_PERSISTED_SETTINGS.get("wifi_enabled", False), False)
try:
    _apply_wifi_state(wifi_enabled)
except Exception:
    pass


def _get_ip_address(force=False, refresh_s=5.0):
    """Return cached IP address (refresh every few seconds to avoid shelling out each frame)."""
    global _IP_CACHE
    now = time.time()
    if (not force) and (now - float(_IP_CACHE.get('ts', 0.0)) < refresh_s):
        return _IP_CACHE.get('ip', 'Unavailable')
    ip = 'Unavailable'
    try:
        output = os.popen('hostname -I 2>/dev/null').read().strip().split()
        if output:
            ip = output[0]
    except Exception:
        pass
    _IP_CACHE = {'ts': now, 'ip': ip}
    return ip


def _clear_double_exposure_state():
    global double_exposure_first_frame, _double_preview_cache
    with _double_exposure_lock:
        double_exposure_first_frame = None
        _double_preview_cache = {"key": None, "image": None}


# ==================== Brenizer Mode: Controller Functions ====================

def _brenizer_tiles():
    """Return the tile list for the current Brenizer variant."""
    if _brenizer_variant == 9:
        return BRENIZER_9_TILES
    if _brenizer_variant == 6:
        return BRENIZER_6_TILES
    return BRENIZER_5_TILES


def _brenizer_directions():
    """Return the direction list for the current Brenizer variant."""
    if _brenizer_variant == 9:
        return BRENIZER_9_DIRECTIONS
    if _brenizer_variant == 6:
        return BRENIZER_6_DIRECTIONS
    return BRENIZER_5_DIRECTIONS


def _brenizer_grid():
    """Return (cols, rows) for the current Brenizer variant."""
    if _brenizer_variant == 9:
        return BRENIZER_9_GRID
    if _brenizer_variant == 6:
        return BRENIZER_6_GRID
    return BRENIZER_5_GRID


def _enter_brenizer_mode(variant=6):
    """Enter Brenizer guided capture mode."""
    global _brenizer_active, _brenizer_variant, _brenizer_state
    global _brenizer_tile_idx, _brenizer_captured, _brenizer_seq_dir
    global _brenizer_seq_id, _brenizer_locked_controls, _brenizer_show_help
    global _brenizer_complete_rects, _brenizer_pulse_t, _brenizer_tile_thumbs
    global double_exposure_enabled

    with _brenizer_lock:
        _brenizer_variant = variant
        _brenizer_active = True
        _brenizer_state = "awaiting_capture"
        _brenizer_tile_idx = 0
        _brenizer_captured = []
        _brenizer_locked_controls = None
        _brenizer_show_help = True
        _brenizer_complete_rects = {}
        _brenizer_pulse_t = 0.0
        _brenizer_tile_thumbs = []

        # Create sequence directory
        _brenizer_seq_id = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        with _capture_dir_lock:
            base_dir = _capture_dir
        _brenizer_seq_dir = os.path.join(base_dir, f"brenizer_{_brenizer_seq_id}")
        _ensure_dir(_brenizer_seq_dir)

    # Disable double exposure when entering Brenizer
    with _double_exposure_lock:
        double_exposure_enabled = False
    _clear_double_exposure_state()

    print(f"[Brenizer] Entered Brenizer {variant} mode — sequence dir: {_brenizer_seq_dir}")


def _exit_brenizer_mode():
    """Exit Brenizer mode and restore camera settings."""
    global _brenizer_active, _brenizer_state, _brenizer_locked_controls
    global _brenizer_complete_rects, _brenizer_show_help, _brenizer_tile_thumbs

    with _brenizer_lock:
        was_locked = _brenizer_locked_controls is not None
        _brenizer_active = False
        _brenizer_state = "idle"
        _brenizer_locked_controls = None
        _brenizer_complete_rects = {}
        _brenizer_show_help = False
        _brenizer_tile_thumbs = []

    # Restore camera AE controls
    if was_locked:
        try:
            _apply_shutter_controls()
        except Exception as e:
            print(f"[Brenizer] Error restoring camera controls: {e}")

    print("[Brenizer] Exited Brenizer mode")


# ==================== Video Recording Mode ====================

def _enter_video_mode(resolution_idx=0):
    """Switch camera to video configuration and enter video standby mode.

    The camera is stopped, reconfigured for video, and restarted.
    The user then presses the shutter button to start/stop recording.
    """
    global _video_mode_active, _video_recording, _video_resolution_idx
    global _video_config, _video_help_shown
    global double_exposure_enabled

    resolution_idx = max(0, min(resolution_idx, len(VIDEO_RESOLUTION_OPTIONS) - 1))

    with _video_lock:
        if _video_mode_active:
            return  # Already in video mode
        _video_resolution_idx = resolution_idx

    # Disable other creative modes
    with _double_exposure_lock:
        double_exposure_enabled = False
    _clear_double_exposure_state()
    if _brenizer_active:
        _exit_brenizer_mode()

    res = VIDEO_RESOLUTION_OPTIONS[resolution_idx]
    vid_w, vid_h = res["size"]
    # Clamp to sensor resolution
    vid_w = min(vid_w, FULL_W)
    vid_h = min(vid_h, FULL_H)

    print(f"[Video] Entering video mode: {res['label']} ({vid_w}x{vid_h}) @ {_VIDEO_FPS}fps H.264")

    # Boost CPU for 4K to keep the live preview smooth
    if vid_w >= 3840:
        _set_cpu_4k_boost(True)

    try:
        picam2.stop()

        # Create video configuration with lores stream for preview
        _lores_w = min(800, vid_w)
        _lores_h = int(_lores_w * vid_h / vid_w)
        # Ensure even dimensions
        _lores_w = _lores_w & ~1
        _lores_h = _lores_h & ~1

        # The sensor is physically mounted upside-down. The ISP transform
        # cannot be used here because it applies to all streams including the
        # lores preview, which the display code already flips via cv2.flip.
        # Instead, rotation is handled by embedding display-rotation metadata
        # in the MP4 container (see _start_video_recording).
        #
        # Explicitly suppress the raw stream: picamera2 otherwise allocates
        # one matching the chosen sensor mode, which on large sensors (e.g.
        # IMX492 at 47 MP) can push memory use past the Pi 5's CMA limit
        # and prevent the camera from starting.  H.264 encoding only needs
        # the main YUV420 stream, so there's nothing to lose by dropping
        # the raw buffer here.
        _video_buffer_count = 2 if (FULL_W * FULL_H) > 20_000_000 else 4
        _video_config = picam2.create_video_configuration(
            main={"size": (vid_w, vid_h), "format": "YUV420"},
            lores={"size": (_lores_w, _lores_h), "format": "RGB888"},
            raw=None,
            controls={
                "FrameDurationLimits": (int(1e6 / _VIDEO_FPS), int(1e6 / _VIDEO_FPS)),
                "AeMeteringMode": 2,
                "NoiseReductionMode": 0,
            },
            buffer_count=_video_buffer_count,
        )
        picam2.configure(_video_config)
        picam2.start()

        with _video_lock:
            _video_mode_active = True
            _video_recording = False
            _video_help_shown = False

        _debounced_save_settings()
        print("[Video] Camera reconfigured for video — ready to record")

    except Exception as e:
        print(f"[Video] Error entering video mode: {e}")
        traceback.print_exc()
        # Attempt to restore still mode
        try:
            picam2.configure(_preview_running_config)
            picam2.start()
            picam2.set_controls({"FrameDurationLimits": _PREVIEW_FRAME_LIMITS})
        except Exception:
            pass


def _exit_video_mode():
    """Stop any active recording and restore camera to still configuration."""
    global _video_mode_active, _video_recording, _video_config

    with _video_lock:
        was_recording = _video_recording
        was_active = _video_mode_active

    if not was_active:
        return

    if was_recording:
        _stop_video_recording()

    # Restore CPU to normal thermal cap if we boosted for 4K
    with _video_lock:
        res_idx = _video_resolution_idx
    res = VIDEO_RESOLUTION_OPTIONS[res_idx]
    if res["size"][0] >= 3840:
        _set_cpu_4k_boost(False)

    print("[Video] Exiting video mode — restoring still configuration")

    try:
        picam2.stop()
        picam2.configure(_preview_running_config)
        picam2.start()
        picam2.set_controls({"FrameDurationLimits": _PREVIEW_FRAME_LIMITS})
        _apply_shutter_controls()
    except Exception as e:
        print(f"[Video] Error restoring still config: {e}")
        traceback.print_exc()

    with _video_lock:
        _video_mode_active = False
        _video_recording = False
        _video_config = None

    print("[Video] Restored to still photography mode")


def _start_video_recording():
    """Begin recording an H.264 video clip."""
    global _video_recording, _video_recording_start, _video_output_path
    global _video_encoder, _video_output

    with _video_lock:
        if not _video_mode_active or _video_recording:
            return

    if not _disk_can_capture():
        print("[Video] Blocked: disk space critical — clear internal memory to continue")
        return

    if not _batt_can_record():
        print("[Video] Blocked: battery critically low — charge battery to record")
        return

    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FfmpegOutput

    res = VIDEO_RESOLUTION_OPTIONS[_video_resolution_idx]

    # Build output path
    with _capture_dir_lock:
        target_dir = _capture_dir
    _ensure_dir(target_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_video_{res['key']}.mp4"
    output_path = os.path.join(target_dir, filename)

    try:
        # H264Encoder uses software H.264 encoding via libav on Pi 5 (no HW encoder).
        # FfmpegOutput wraps the stream into an MP4 container.
        vid_w, vid_h = res["size"]
        vid_w = min(vid_w, FULL_W)
        vid_h = min(vid_h, FULL_H)

        # Bitrate: 1080p ~8Mbps, 4K ~20Mbps — good quality without huge files
        if vid_w >= 3840:
            bitrate = 20_000_000
        else:
            bitrate = 8_000_000

        encoder = H264Encoder(bitrate=bitrate)
        # Embed 180° rotation metadata in the MP4 container so players rotate
        # on playback. The sensor is mounted upside-down but we cannot use an
        # ISP transform (it would double-flip the preview). FfmpegOutput splits
        # its filename arg and appends it to the ffmpeg command line.
        output = FfmpegOutput(f"-metadata:s:v rotate=180 {output_path}", audio=False)

        # Apply the real manual shutter speed for recording.  During preview,
        # slow shutters (>1/30s) are capped at 1/30s for smooth live-view; the
        # actual exposure is stored in _slow_shutter_capture_us.  We must set it
        # on the sensor before the encoder starts so every video frame uses the
        # intended exposure time.
        _vid_slow_us = _slow_shutter_capture_us
        if _vid_slow_us is not None:
            picam2.set_controls({
                "ExposureTime": int(_vid_slow_us),
                "FrameDurationLimits": (int(_vid_slow_us), int(_vid_slow_us)),
            })
            # Wait for sensor to apply the new exposure before encoding starts
            time.sleep(_vid_slow_us / 1e6 * 2 + 0.05)
            print(f"[Video] Slow shutter applied: {_vid_slow_us}µs")

        picam2.start_encoder(encoder, output, name="main")

        with _video_lock:
            _video_recording = True
            _video_recording_start = time.monotonic()
            _video_output_path = output_path
            _video_encoder = encoder
            _video_output = output

        print(f"[Video] Recording started: {output_path}")

    except Exception as e:
        print(f"[Video] Error starting recording: {e}")
        traceback.print_exc()


def _stop_video_recording():
    """Stop the current video recording and finalize the file."""
    global _video_recording, _video_encoder, _video_output, _video_output_path

    with _video_lock:
        if not _video_recording:
            return
        encoder = _video_encoder
        output_path = _video_output_path

    try:
        picam2.stop_encoder()
        print(f"[Video] Recording stopped: {output_path}")
        _queue_usb_sync(output_path)
    except Exception as e:
        print(f"[Video] Error stopping recording: {e}")
        traceback.print_exc()
    finally:
        with _video_lock:
            _video_recording = False
            _video_encoder = None
            _video_output = None
            _video_output_path = None

    # Restore preview shutter cap so live-view returns to smooth frame rate
    _apply_shutter_controls()


def _video_elapsed_s():
    """Return seconds elapsed since recording started, or 0 if not recording."""
    with _video_lock:
        if not _video_recording:
            return 0.0
        return time.monotonic() - _video_recording_start


def _video_remaining_s():
    """Return seconds remaining before auto-stop, or full duration if not recording."""
    elapsed = _video_elapsed_s()
    return max(0.0, _VIDEO_MAX_DURATION_S - elapsed)


def _brenizer_on_capture(jpg_path):
    """Called after each successful Brenizer frame capture.

    Locks exposure/gain after the first (center) shot and advances
    to the next tile. When all tiles are captured, transitions to
    the completion state.
    """
    global _brenizer_tile_idx, _brenizer_state, _brenizer_locked_controls

    with _brenizer_lock:
        if not _brenizer_active:
            return

        # Lock exposure after center shot
        if _brenizer_tile_idx == 0:
            try:
                meta = _meta  # latest metadata from main loop
                locked_exp = int(meta.get("ExposureTime", 33333))
                locked_gain = float(meta.get("AnalogueGain", 1.0))
                _brenizer_locked_controls = {
                    "ExposureTime": locked_exp,
                    "AnalogueGain": locked_gain,
                }
                picam2.set_controls({
                    "AeEnable": False,
                    "ExposureTime": locked_exp,
                    "AnalogueGain": locked_gain,
                })
                print(f"[Brenizer] Locked exposure: {locked_exp}µs, gain: {locked_gain:.2f}")
            except Exception as e:
                print(f"[Brenizer] Warning: could not lock exposure: {e}")

        _brenizer_captured.append(jpg_path)
        _brenizer_tile_idx += 1

        total = len(_brenizer_tiles())
        if _brenizer_tile_idx >= total:
            _brenizer_state = "complete"
            print(f"[Brenizer] Capture complete — {total} images saved to {_brenizer_seq_dir}")
        else:
            directions = _brenizer_directions()
            next_dir = directions[_brenizer_tile_idx] if _brenizer_tile_idx < len(directions) else ""
            print(f"[Brenizer] Shot {_brenizer_tile_idx}/{total} captured — next: {next_dir}")


def _brenizer_retake_last():
    """Retake the last captured frame by stepping back one tile."""
    global _brenizer_tile_idx, _brenizer_state

    with _brenizer_lock:
        if not _brenizer_active or _brenizer_tile_idx <= 0:
            return

        _brenizer_tile_idx -= 1
        _brenizer_state = "awaiting_capture"

        # Remove the last tile thumbnail
        if _brenizer_tile_thumbs:
            _brenizer_tile_thumbs.pop()

        # Remove the last captured file path
        if _brenizer_captured:
            last_path = _brenizer_captured.pop()
            try:
                if os.path.exists(last_path):
                    os.remove(last_path)
                # Also remove DNG if it exists
                dng_path = last_path.replace(".jpg", ".dng")
                if os.path.exists(dng_path):
                    os.remove(dng_path)
            except Exception as e:
                print(f"[Brenizer] Error removing file: {e}")

        tile_num = _brenizer_tile_idx + 1
        print(f"[Brenizer] Retaking shot {tile_num}")


# ==================== Brenizer Mode: UI Rendering Functions ====================

def _render_brenizer_minimap(canvas, x_pos, y_pos, live_frame=None):
    """Draw a mini-map showing capture progress with image thumbnails.

    Completed tiles show the actual captured image thumbnail.
    The current/active tile shows the live preview feed.
    Remaining tiles show a dim outline.

    Args:
        canvas: The display canvas to draw onto.
        x_pos: Left edge of the mini-map area.
        y_pos: Top edge of the mini-map area.
        live_frame: Current preview frame (BGR) for showing in the active tile.
    """
    global _brenizer_pulse_t

    tiles = _brenizer_tiles()
    cols, rows = _brenizer_grid()
    tile_idx = _brenizer_tile_idx

    # Cell size — large enough to show meaningful image content
    cell_w, cell_h = 56, 36
    gap = 2
    pad = 4
    map_w = cols * cell_w + (cols - 1) * gap + 2 * pad
    map_h = rows * cell_h + (rows - 1) * gap + 2 * pad

    # Build mini-map in a small buffer
    minimap = np.zeros((map_h, map_w, 3), dtype=np.uint8)

    # Dark background
    cv2.rectangle(minimap, (0, 0), (map_w - 1, map_h - 1), (20, 20, 20), -1)

    # Pulse animation for current tile border
    _brenizer_pulse_t += _FRAME_DT * 3.0
    pulse = 0.5 + 0.5 * np.sin(_brenizer_pulse_t)

    # Prepare live preview thumbnail for active tile (resize once, reuse)
    live_thumb = None
    if live_frame is not None and tile_idx < len(tiles):
        try:
            live_thumb = cv2.resize(live_frame, (cell_w, cell_h),
                                    interpolation=cv2.INTER_AREA)
            if live_thumb.ndim == 3 and live_thumb.shape[2] == 4:
                live_thumb = cv2.cvtColor(live_thumb, cv2.COLOR_BGRA2BGR)
        except Exception:
            live_thumb = None

    for seq_idx, tile in enumerate(tiles):
        col, row = tile["col"], tile["row"]
        cx = pad + col * (cell_w + gap)
        cy = pad + row * (cell_h + gap)

        if seq_idx < tile_idx and seq_idx < len(_brenizer_tile_thumbs):
            # Completed tile — show captured image thumbnail
            thumb = _brenizer_tile_thumbs[seq_idx]
            if thumb is not None:
                minimap[cy:cy + cell_h, cx:cx + cell_w] = thumb
            else:
                cv2.rectangle(minimap, (cx, cy), (cx + cell_w - 1, cy + cell_h - 1),
                              UI_ACCENT_CREATIVE, -1)
            # Thin accent border to indicate completion
            cv2.rectangle(minimap, (cx - 1, cy - 1),
                          (cx + cell_w, cy + cell_h), UI_ACCENT_CREATIVE, 1)
        elif seq_idx == tile_idx:
            # Current tile — show live preview
            if live_thumb is not None:
                minimap[cy:cy + cell_h, cx:cx + cell_w] = live_thumb
            # Pulsing bright border on the active tile
            brightness = int(140 + 115 * pulse)
            highlight = (brightness, int(brightness * 0.7), int(brightness * 0.3))
            cv2.rectangle(minimap, (cx - 1, cy - 1),
                          (cx + cell_w, cy + cell_h), highlight, 2)
        else:
            # Remaining tile — dim outline
            cv2.rectangle(minimap, (cx, cy), (cx + cell_w - 1, cy + cell_h - 1),
                          (50, 50, 50), 1)

    # Outer border
    cv2.rectangle(minimap, (0, 0), (map_w - 1, map_h - 1), (70, 70, 70), 1)

    # Blend onto canvas (no rotation — matches preview orientation)
    blit_add(canvas, minimap, x_pos, y_pos)


def _render_brenizer_status(canvas, x_off, y_off, new_w, new_h):
    """Draw Brenizer mode status text (mode name, shot counter, direction).

    Renders at the top of the preview area, replacing the normal SS/ISO display
    when Brenizer mode is active.
    """
    tiles = _brenizer_tiles()
    tile_idx = _brenizer_tile_idx
    total = len(tiles)
    variant = _brenizer_variant
    directions = _brenizer_directions()
    direction = directions[tile_idx] if tile_idx < len(directions) else ""

    # Build status lines
    mode_text = f"BRENIZER {variant}"
    shot_text = f"Shot {tile_idx + 1} / {total}"

    # Render mode name + shot counter as a single rotated block
    status_block = make_rotated_text_block(
        [[{"text": mode_text, "font_scale": 1.2}, f"   {shot_text}"]],
        font_scale=1.4,
        thickness=2,
        rotate_180=True,
        max_h=50,
        pad_x=4,
        pad_y=2,
    )
    # Center horizontally at top of preview
    sx = SCREEN_W // 2 - status_block.shape[1] // 2
    sy = max(12, y_off + 12)
    blit_add_with_shadow(canvas, status_block, sx, sy)

    return sy + status_block.shape[0]


def _render_brenizer_direction_prompt(canvas, y_offset):
    """Draw directional guidance text below the status line."""
    tile_idx = _brenizer_tile_idx
    directions = _brenizer_directions()
    direction = directions[tile_idx] if tile_idx < len(directions) else ""

    if tile_idx == 0:
        # First shot — guide to center
        line1 = "Capture your subject"
        line2 = ""
    else:
        line1 = direction
        line2 = "Keep ~30% overlap"

    lines = [[line1]]
    if line2:
        lines.append([{"text": line2, "font_scale": 0.8}])

    prompt_block = make_rotated_text_block(
        lines,
        font_scale=1.6,
        thickness=2,
        rotate_180=True,
        max_h=80,
        pad_x=8,
        pad_y=4,
        line_gap=6,
    )

    px = SCREEN_W // 2 - prompt_block.shape[1] // 2
    py = y_offset + 8
    blit_add_with_shadow(canvas, prompt_block, px, py)


def _render_brenizer_help_overlay(canvas):
    """Draw semi-transparent instruction overlay for Brenizer mode."""
    overlay_h, overlay_w = canvas.shape[:2]

    # Darken the background
    roi = canvas[0:overlay_h, 0:overlay_w]
    cv2.convertScaleAbs(roi, alpha=0.3, beta=0, dst=roi)

    help_lines = [
        ["BRENIZER MODE"],
        [""],
        ["Start with your subject in the center"],
        ["Keep your feet planted"],
        ["Rotate the camera, don't step"],
        ["Keep about 30% overlap"],
        ["Follow the highlighted tile"],
        [""],
        [{"text": "Tap anywhere to start", "font_scale": 0.9}],
    ]

    help_block = make_rotated_text_block(
        help_lines,
        font_scale=1.2,
        thickness=1,
        rotate_180=True,
        max_h=int(overlay_h * 0.75),
        pad_x=16,
        pad_y=8,
        line_gap=10,
    )

    hx = (overlay_w - help_block.shape[1]) // 2
    hy = (overlay_h - help_block.shape[0]) // 2
    blit_add_with_shadow(canvas, help_block, hx, hy)


def _render_brenizer_complete_screen(canvas):
    """Draw the completion screen after all Brenizer frames are captured.

    Shows summary text and touch buttons for Done / Retake Last / Preview Merge.
    Button rects are stored in _brenizer_complete_rects for touch handling.
    """
    global _brenizer_complete_rects

    h, w = canvas.shape[:2]

    # Darken background
    cv2.convertScaleAbs(canvas, alpha=0.25, beta=0, dst=canvas)

    total = len(_brenizer_tiles())
    variant = _brenizer_variant

    # Title block
    title_lines = [
        [{"text": "Capture Complete", "font_scale": 1.4}],
        [f"{total} images saved"],
        [f"BRENIZER {variant}"],
    ]

    title_block = make_rotated_text_block(
        title_lines,
        font_scale=1.3,
        thickness=2,
        rotate_180=True,
        max_h=int(h * 0.35),
        pad_x=16,
        pad_y=8,
        line_gap=12,
    )

    tx = (w - title_block.shape[1]) // 2
    ty = int(h * 0.12)
    blit_add(canvas, title_block, tx, ty)

    # Draw buttons — three horizontally arranged touch targets
    btn_labels = [("Done", "done"), ("Retake Last", "retake"), ("Preview Merge", "merge")]
    btn_w, btn_h = 160, 52
    btn_gap = 20
    total_btn_w = len(btn_labels) * btn_w + (len(btn_labels) - 1) * btn_gap
    btn_start_x = (w - total_btn_w) // 2
    btn_y = int(h * 0.55)

    _brenizer_complete_rects = {}
    for i, (label, key) in enumerate(btn_labels):
        bx = btn_start_x + i * (btn_w + btn_gap)
        by = btn_y

        # Button background (rotated coordinates — draw in normal space, rects in rotated)
        if key == "done":
            btn_color = UI_ACCENT_CREATIVE
        elif key == "retake":
            btn_color = (100, 100, 100)
        else:
            btn_color = (70, 70, 70)

        # Draw rounded button
        cv2.rectangle(canvas, (bx, by), (bx + btn_w, by + btn_h), btn_color, -1)
        cv2.rectangle(canvas, (bx, by), (bx + btn_w, by + btn_h), (180, 180, 180), 1)

        # Draw label text (rotated 180°)
        font_sz = 16
        tw_val, th_val = _ui_measure_text(label, font_sz)
        # Create small text buffer, rotate, then blit
        txt_buf = np.zeros((th_val + 4, tw_val + 4, 3), dtype=np.uint8)
        _ui_draw_text(txt_buf, label, 2, 2, font_sz, (255, 255, 255), outline_bgr=(0, 0, 0))
        txt_buf = cv2.rotate(txt_buf, cv2.ROTATE_180)
        text_x = bx + (btn_w - txt_buf.shape[1]) // 2
        text_y = by + (btn_h - txt_buf.shape[0]) // 2
        blit_add(canvas, txt_buf, text_x, text_y)

        # Store rect in canvas coordinates (same space as all other button rects)
        _brenizer_complete_rects[key] = (bx, by, bx + btn_w, by + btn_h)

    # Show "Coming soon" note under Preview Merge if applicable
    if time.monotonic() < _brenizer_merge_toast_until:
        toast_block = make_rotated_text_block(
            [["Preview merge coming soon"]],
            font_scale=1.0,
            thickness=1,
            rotate_180=True,
            max_h=30,
        )
        toast_x = (w - toast_block.shape[1]) // 2
        toast_y = btn_y + btn_h + 20
        blit_add_with_shadow(canvas, toast_block, toast_x, toast_y)


def _draw_brenizer_icon(canvas, active):
    """Draw a grid icon representing Brenizer mode."""
    h, w = canvas.shape[:2]
    color = UI_ACCENT_CREATIVE if active else UI_INACTIVE
    thickness = max(1, w // 30)
    pad = int(min(w, h) * 0.22)

    # Draw a 3x3 mini grid
    grid_x1, grid_y1 = pad, pad
    grid_x2, grid_y2 = w - pad, h - pad
    gw = grid_x2 - grid_x1
    gh = grid_y2 - grid_y1
    cell_gap = max(2, w // 20)

    cell_w = (gw - 2 * cell_gap) // 3
    cell_h = (gh - 2 * cell_gap) // 3

    for row in range(3):
        for col in range(3):
            cx = grid_x1 + col * (cell_w + cell_gap)
            cy = grid_y1 + row * (cell_h + cell_gap)
            if row == 1 and col == 1:
                # Center cell is filled
                cv2.rectangle(canvas, (cx, cy), (cx + cell_w, cy + cell_h), color, -1)
            else:
                cv2.rectangle(canvas, (cx, cy), (cx + cell_w, cy + cell_h), color, thickness)


# ==================== Video Recording Overlay ====================

def _draw_video_overlay(dst):
    """Draw the video recording overlay: mode indicator, timer, and resolution badge.

    When in video standby (not recording): shows "VIDEO" label and resolution.
    When recording: shows pulsing red dot, elapsed/remaining time, resolution.
    All text is rendered via make_rotated_text_block (180° rotated) to match
    the flipped display orientation used by the rest of the UI.
    """
    with _video_lock:
        recording = _video_recording
        mode_active = _video_mode_active
        rec_start = _video_recording_start
    if not mode_active:
        return

    dst_h, dst_w = dst.shape[:2]
    res = VIDEO_RESOLUTION_OPTIONS[_video_resolution_idx]

    if recording:
        elapsed = time.monotonic() - rec_start
        remaining = max(0.0, _VIDEO_MAX_DURATION_S - elapsed)
        elapsed_min = int(elapsed) // 60
        elapsed_sec = int(elapsed) % 60

        # Pulsing red recording dot (blinks every ~1s)
        # Positioned near top of the visual display (bottom of canvas, since 180° rotated)
        pulse = (time.monotonic() % 1.0) < 0.7
        dot_radius = 10
        # In the rotated display, visual top-center is canvas bottom-center
        dot_cx = dst_w // 2 + 60
        dot_cy = 28
        if pulse:
            cv2.circle(dst, (dot_cx, dot_cy), dot_radius, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.circle(dst, (dot_cx, dot_cy), dot_radius, (0, 0, 180), 1, cv2.LINE_AA)

        # Timer + resolution badge as rotated text block (positioned right of dot)
        timer_text = f"REC {elapsed_min:01d}:{elapsed_sec:02d}"
        badge_text = f"{res['label']} H.264 24fps"
        rec_block = make_rotated_text_block(
            [[{"text": timer_text, "font_scale": 1.4}, f"   {badge_text}"]],
            font_scale=1.2,
            thickness=2,
            rotate_180=True,
            max_h=40,
            pad_x=4,
            pad_y=2,
        )
        # Place to the left of the dot (visually right, since rotated)
        bx = dot_cx - dot_radius - 8 - rec_block.shape[1]
        by = dot_cy - rec_block.shape[0] // 2
        blit_add_with_shadow(dst, rec_block, bx, by)

        # Remaining time warning (last 15 seconds)
        if remaining <= _VIDEO_WARN_REMAINING_S:
            warn_text = f"{int(remaining)}s left"
            warn_block = make_rotated_text_block(
                [[warn_text]],
                font_scale=1.0,
                thickness=2,
                rotate_180=True,
                max_h=30,
                pad_x=4,
                pad_y=2,
            )
            wx = dst_w // 2 - warn_block.shape[1] // 2
            wy = dot_cy + dot_radius + 6
            # Blink the warning
            if (time.monotonic() % 0.5) < 0.3:
                blit_add_with_shadow(dst, warn_block, wx, wy)

    else:
        # Standby mode — show "VIDEO" label and resolution
        standby_block = make_rotated_text_block(
            [[{"text": f"VIDEO  {res['label']}", "font_scale": 1.4}]],
            font_scale=1.4,
            thickness=2,
            rotate_180=True,
            max_h=40,
            pad_x=4,
            pad_y=2,
        )
        sx = dst_w // 2 - standby_block.shape[1] // 2
        sy = 14
        blit_add_with_shadow(dst, standby_block, sx, sy)

        # "Press shutter to record" hint
        hint_block = make_rotated_text_block(
            [["Press shutter to record | Hold to exit"]],
            font_scale=0.9,
            thickness=1,
            rotate_180=True,
            max_h=28,
            pad_x=4,
            pad_y=2,
        )
        hx = dst_w // 2 - hint_block.shape[1] // 2
        hy = sy + standby_block.shape[0] + 4
        blit_add_with_shadow(dst, hint_block, hx, hy)


def _draw_video_icon(canvas, active=False):
    """Draw a simple video camera icon on a button canvas."""
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    color = (0, 0, 200) if active else (200, 200, 200)
    # Camera body rectangle
    bw, bh = 22, 16
    x1 = cx - bw // 2 - 3
    y1 = cy - bh // 2
    x2 = x1 + bw
    y2 = y1 + bh
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    # Lens/play triangle
    pts = np.array([
        [x2 + 3, cy - 6],
        [x2 + 12, cy],
        [x2 + 3, cy + 6],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA)


# ==================== Brenizer Mode: Multishot Popup Rendering ====================

def _draw_multishot_controls(dst, top, right_edge, size=72, margin=16):
    """Draw the multishot button (Double Exp / Brenizer / Video) with popup menu.

    When Brenizer mode is active, shows the grid icon.
    When Video mode is active, shows the video camera icon.
    Otherwise shows the double-exposure overlapping-frames icon.
    """
    global _multishot_popup_rects, _multishot_popup_bounds

    with _double_exposure_lock:
        double_on = double_exposure_enabled
        double_waiting = double_exposure_first_frame is not None
    with _brenizer_lock:
        bren_active = _brenizer_active
    with _video_lock:
        video_active = _video_mode_active
        video_recording = _video_recording
    popup_visible = _multishot_popup_visible

    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    is_active = double_on or double_waiting or bren_active or video_active
    if video_active:
        accent = (0, 0, 200) if video_recording else UI_ACCENT_CREATIVE
    else:
        accent = UI_ACCENT_CREATIVE if is_active else UI_INACTIVE
    icon, mask = _render_icon_button(dst, bounds, active=is_active, accent_color=accent)

    if video_active:
        _draw_video_icon(icon, video_recording)
    elif bren_active:
        _draw_brenizer_icon(icon, True)
    else:
        _draw_double_exposure_icon(icon, double_on or double_waiting, double_waiting)

    drawn_rect = _draw_icon_to_dst(dst, icon, mask, bounds)

    # Render popup if visible
    labels = [opt["label"] for opt in MULTISHOT_OPTIONS]
    # Determine active index
    active_idx = -1
    if double_on:
        active_idx = 0
    elif bren_active and _brenizer_variant == 5:
        active_idx = 1
    elif bren_active and _brenizer_variant == 6:
        active_idx = 2
    elif bren_active and _brenizer_variant == 9:
        active_idx = 3
    elif video_active and _video_resolution_idx == 0:
        active_idx = 4  # Video 1080p
    elif video_active and _video_resolution_idx == 1:
        active_idx = 5  # Video 4K

    popup_rects, popup_bounds = _render_popup_grid(
        dst, bounds, labels, active_idx, _multishot_popup_fade, popup_visible,
        cols=1, font_size=18, active_color=UI_ACCENT_CREATIVE,
        position="right", margin=margin,
    )

    _multishot_popup_rects = popup_rects if popup_visible else []
    _multishot_popup_bounds = popup_bounds

    return drawn_rect if drawn_rect else bounds


def _draw_shutter_controls(dst, top, right_edge, size=72, margin=16, gap=18):
    global _shutter_toggle_rect, _shutter_popup_rects, _shutter_popup_bounds

    with _shutter_button_lock:
        active_idx = current_shutter_idx
        popup_visible = _shutter_popup_visible

    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    manual_selected = SHUTTER_OPTIONS_US[active_idx] is not None
    icon, mask = _render_icon_button(dst, bounds, active=manual_selected, accent_color=UI_ACCENT_EXPOSURE)
    _draw_shutter_icon(icon, manual_selected, popup_visible)
    drawn_rect = _draw_icon_to_dst(dst, icon, mask, bounds)

    popup_rects, popup_bounds = _render_popup_grid(
        dst, bounds, SHUTTER_LABELS, active_idx, _shutter_popup_fade, popup_visible,
        cols=min(2, len(SHUTTER_LABELS)) or 1, font_size=18, position="right", margin=margin,
    )

    with _shutter_button_lock:
        _shutter_toggle_rect = drawn_rect if drawn_rect else bounds
        if popup_visible:
            _shutter_popup_rects = popup_rects
            _shutter_popup_bounds = popup_bounds
        else:
            _shutter_popup_rects = []
            _shutter_popup_bounds = None

    return drawn_rect if drawn_rect else bounds


def _draw_film_controls(dst, top, right_edge, size=72, margin=16, gap=18):
    global _film_toggle_rect, _film_popup_rects, _film_popup_bounds
    with _film_button_lock:
        active_idx = _film_idx
        popup_visible = _film_popup_visible

    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    short_label = FILM_OPTIONS[active_idx]["short"]
    is_alt = active_idx != 0
    accent = UI_ACCENT_CREATIVE if is_alt else UI_INACTIVE
    icon, mask = _render_icon_button(dst, bounds, active=popup_visible or is_alt, accent_color=accent)
    _draw_film_icon(icon, short_label, popup_visible or is_alt)
    drawn_rect = _draw_icon_to_dst(dst, icon, mask, bounds)

    film_labels = [opt["label"] for opt in FILM_OPTIONS]
    popup_rects, popup_bounds = _render_popup_grid(
        dst, bounds, film_labels, active_idx, _film_popup_fade, popup_visible,
        cols=min(3, len(FILM_OPTIONS)) or 1, font_size=17, position="left", margin=margin,
    )

    with _film_button_lock:
        _film_toggle_rect = drawn_rect if drawn_rect else bounds
        if popup_visible:
            _film_popup_rects = popup_rects
            _film_popup_bounds = popup_bounds
        else:
            _film_popup_rects = []
            _film_popup_bounds = None

    return drawn_rect if drawn_rect else bounds

_GRID_POPUP_IDX_OFFSET = 1000  # offset for grid items in combined aspect popup rects

def _draw_aspect_controls(dst, top, right_edge, size=72, margin=16):
    global _aspect_toggle_rect, _aspect_popup_rects, _aspect_popup_bounds
    with _aspect_button_lock:
        active_idx = _aspect_idx
        popup_visible = _aspect_popup_visible
    with _grid_button_lock:
        grid_idx = _grid_overlay_idx

    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    label = ASPECT_OPTIONS[active_idx]["label"]
    ratio = ASPECT_OPTIONS[active_idx]["ratio"]
    icon, mask = _render_icon_button(dst, bounds, active=popup_visible, accent_color=UI_ACCENT_CREATIVE)
    _draw_aspect_icon(icon, label, ratio, popup_visible)
    drawn_rect = _draw_icon_to_dst(dst, icon, mask, bounds)

    # --- Combined aspect + grid popup ---
    popup_rects, popup_bounds = _render_aspect_grid_popup(
        dst, bounds, _aspect_popup_fade, popup_visible,
        active_idx, grid_idx, margin=margin,
    )

    with _aspect_button_lock:
        _aspect_toggle_rect = drawn_rect if drawn_rect else bounds
        if popup_visible:
            _aspect_popup_rects = popup_rects
            _aspect_popup_bounds = popup_bounds
        else:
            _aspect_popup_rects = []
            _aspect_popup_bounds = None

    return drawn_rect if drawn_rect else bounds


def _render_aspect_grid_popup(dst, bounds, fade_opacity, popup_visible,
                               aspect_idx, grid_idx, margin=16):
    """Render combined popup with aspect ratio options + grid overlay subsection."""
    if not (popup_visible or fade_opacity > 0.01):
        return [], None

    dst_h, dst_w = dst.shape[:2]
    font_size = 18
    section_font_size = 14
    option_pad_x, option_pad_y = 14, 10
    panel_pad_x, panel_pad_y = 14, 14
    gap_x, gap_y = 8, 8
    corner_r = _POPUP_CORNER_RADIUS
    btn_corner_r = 6
    active_color = UI_ACCENT_CREATIVE

    # --- Aspect ratio section layout ---
    aspect_labels = [opt["label"] for opt in ASPECT_OPTIONS]
    aspect_cols = min(3, len(aspect_labels)) or 1
    a_metrics = [_ui_measure_text(lbl, font_size) for lbl in aspect_labels]
    a_max_w = max(m[0] for m in a_metrics)
    a_max_h = max(m[1] for m in a_metrics)
    a_btn_w = a_max_w + option_pad_x * 2
    a_btn_h = a_max_h + option_pad_y * 2
    a_rows = (len(aspect_labels) + aspect_cols - 1) // aspect_cols
    aspect_section_w = aspect_cols * a_btn_w + (aspect_cols - 1) * gap_x
    aspect_section_h = a_rows * a_btn_h + (a_rows - 1) * gap_y

    # --- Grid overlay section layout ---
    grid_labels = [opt["label"] for opt in GRID_OVERLAY_OPTIONS]
    grid_cols = len(grid_labels)
    g_metrics = [_ui_measure_text(lbl, font_size) for lbl in grid_labels]
    g_max_w = max(m[0] for m in g_metrics)
    g_max_h = max(m[1] for m in g_metrics)
    g_btn_w = g_max_w + option_pad_x * 2
    g_btn_h = g_max_h + option_pad_y * 2
    grid_section_w = grid_cols * g_btn_w + (grid_cols - 1) * gap_x
    grid_section_h = g_btn_h  # Single row

    # --- Section header ---
    section_label = "Grid"
    sec_tw, sec_th = _ui_measure_text(section_label, section_font_size)
    divider_gap = 12
    section_header_h = sec_th + 6  # text + small padding below

    # --- Panel dimensions ---
    content_w = max(aspect_section_w, grid_section_w)
    panel_w = content_w + 2 * panel_pad_x
    panel_h = (panel_pad_y + aspect_section_h +
               divider_gap + section_header_h + 4 +
               grid_section_h + panel_pad_y)

    # Vertical: prefer above the button, fall back to below
    gap = 12
    space_above = bounds[1]
    space_below = dst_h - bounds[3]
    if space_above >= panel_h + gap and space_above >= space_below:
        panel_y = max(margin, bounds[1] - gap - panel_h)
    else:
        panel_y = min(dst_h - panel_h - margin, bounds[3] + gap)

    # Horizontal: left of button
    panel_x = max(margin, min(bounds[0] - gap - panel_w, dst_w - panel_w - margin))
    panel_x = max(margin, panel_x)
    panel_x2 = panel_x + panel_w
    panel_y2 = panel_y + panel_h

    popup_img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    popup_mask = _create_rounded_rect_mask(panel_w, panel_h, corner_r)
    _draw_rounded_rect(popup_img, (0, 0), (panel_w, panel_h), (22, 22, 22), corner_r)
    _draw_rounded_rect(popup_img, (0, 0), (panel_w - 1, panel_h - 1), (180, 180, 180), corner_r, 2)

    popup_rects = []
    text_items = []

    # --- Draw aspect ratio options ---
    aspect_x_off = panel_pad_x + (content_w - aspect_section_w) // 2
    aspect_y_off = panel_pad_y
    for idx, lbl in enumerate(aspect_labels):
        col = idx % aspect_cols
        row = idx // aspect_cols
        lx1 = aspect_x_off + col * (a_btn_w + gap_x)
        ly1 = aspect_y_off + row * (a_btn_h + gap_y)
        lx2 = lx1 + a_btn_w
        ly2 = ly1 + a_btn_h
        btn_color = active_color if (idx == aspect_idx) else (55, 55, 55)
        _draw_rounded_rect(popup_img, (lx1, ly1), (lx2, ly2), btn_color, btn_corner_r)
        _draw_rounded_rect(popup_img, (lx1, ly1), (lx2 - 1, ly2 - 1), (140, 140, 140), btn_corner_r, 1)
        tw, th = _ui_measure_text(lbl, font_size)
        tx = lx1 + (a_btn_w - tw) // 2
        ty = ly1 + (a_btn_h - th) // 2
        text_items.append((lbl, tx, ty, (255, 255, 255)))
        # Rotated touch rect (popup is rotated 180°)
        rot_x1 = panel_x + (panel_w - lx2)
        rot_y1 = panel_y + (panel_h - ly2)
        rot_x2 = panel_x + (panel_w - lx1)
        rot_y2 = panel_y + (panel_h - ly1)
        popup_rects.append((idx, (rot_x1, rot_y1, rot_x2, rot_y2)))

    # --- Divider line + section label ---
    div_y = aspect_y_off + aspect_section_h + divider_gap // 2
    cv2.line(popup_img, (panel_pad_x, div_y), (panel_w - panel_pad_x, div_y),
             (80, 80, 80), 1, cv2.LINE_AA)
    sec_label_y = div_y + divider_gap // 2
    sec_label_x = panel_pad_x + 2

    # --- Draw grid overlay options ---
    grid_y_off = sec_label_y + section_header_h + 4
    grid_x_off = panel_pad_x + (content_w - grid_section_w) // 2
    for idx, lbl in enumerate(grid_labels):
        lx1 = grid_x_off + idx * (g_btn_w + gap_x)
        ly1 = grid_y_off
        lx2 = lx1 + g_btn_w
        ly2 = ly1 + g_btn_h
        btn_color = active_color if (idx == grid_idx) else (55, 55, 55)
        _draw_rounded_rect(popup_img, (lx1, ly1), (lx2, ly2), btn_color, btn_corner_r)
        _draw_rounded_rect(popup_img, (lx1, ly1), (lx2 - 1, ly2 - 1), (140, 140, 140), btn_corner_r, 1)
        tw, th = _ui_measure_text(lbl, font_size)
        tx = lx1 + (g_btn_w - tw) // 2
        ty = ly1 + (g_btn_h - th) // 2
        text_items.append((lbl, tx, ty, (255, 255, 255)))
        # Rotated touch rect with offset to distinguish from aspect items
        rot_x1 = panel_x + (panel_w - lx2)
        rot_y1 = panel_y + (panel_h - ly2)
        rot_x2 = panel_x + (panel_w - lx1)
        rot_y2 = panel_y + (panel_h - ly1)
        popup_rects.append((_GRID_POPUP_IDX_OFFSET + idx, (rot_x1, rot_y1, rot_x2, rot_y2)))

    _ui_draw_text_batch(popup_img, text_items, font_size, outline_bgr=(0, 0, 0))
    # Draw section label at smaller font size
    _ui_draw_text_batch(popup_img,
                        [(section_label, sec_label_x, sec_label_y, (160, 160, 160))],
                        section_font_size, outline_bgr=(0, 0, 0))
    popup_img = cv2.rotate(popup_img, cv2.ROTATE_180)
    popup_mask = cv2.rotate(popup_mask, cv2.ROTATE_180)
    _blend_popup_to_dst(dst, popup_img, panel_x, panel_y, panel_x2, panel_y2, fade_opacity, popup_mask)
    return popup_rects, (panel_x, panel_y, panel_x2, panel_y2)


def _set_grid_overlay(idx):
    """Set the active grid overlay option (grid is a subsection of the aspect popup)."""
    global _grid_overlay_idx
    global _aspect_popup_visible, _aspect_popup_rects, _aspect_popup_bounds
    with _grid_button_lock:
        idx = max(0, min(idx, len(GRID_OVERLAY_OPTIONS) - 1))
        _grid_overlay_idx = idx
    with _aspect_button_lock:
        _aspect_popup_visible = False
        _aspect_popup_rects = []
        _aspect_popup_bounds = None
    _record_activity(wake=True)
    _debounced_save_settings()


def _current_grid_key():
    with _grid_button_lock:
        return GRID_OVERLAY_OPTIONS[_grid_overlay_idx]["key"]


_grid_overlay_cache = {}  # (w, h, grid_key) -> pre-scaled uint8 overlay


def _build_grid_overlay(w, h, grid_key):
    """Build the overlay image for a given grid mode (called once, then cached)."""
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    line_color = (255, 255, 255)
    thickness = max(1, min(w, h) // 400)

    if grid_key == "thirds":
        for i in (1, 2):
            vx = w * i // 3
            cv2.line(overlay, (vx, 0), (vx, h - 1), line_color, thickness, cv2.LINE_AA)
            hy = h * i // 3
            cv2.line(overlay, (0, hy), (w - 1, hy), line_color, thickness, cv2.LINE_AA)
    elif grid_key == "golden":
        rect_color = (200, 200, 200)
        spiral_color = (80, 80, 255)  # red in BGR
        spiral_thick = max(2, min(w, h) // 200)
        phi = 0.6180339887
        rx, ry, rw, rh = 0, 0, w, h
        for i in range(10):
            orient = (i + 2) % 4
            if orient == 0:
                sq = int(round(rw * phi)); rem = rw - sq
                cv2.line(overlay, (rx + sq, ry), (rx + sq, ry + rh - 1),
                         rect_color, thickness, cv2.LINE_AA)
                cv2.ellipse(overlay, (rx + sq, ry + rh), (sq, rh), 0, 180, 270,
                            spiral_color, spiral_thick, cv2.LINE_AA)
                rx, rw = rx + sq, rem
            elif orient == 1:
                sq = int(round(rh * phi)); rem = rh - sq
                cv2.line(overlay, (rx, ry + sq), (rx + rw - 1, ry + sq),
                         rect_color, thickness, cv2.LINE_AA)
                cv2.ellipse(overlay, (rx, ry + sq), (rw, sq), 0, 270, 360,
                            spiral_color, spiral_thick, cv2.LINE_AA)
                ry, rh = ry + sq, rem
            elif orient == 2:
                sq = int(round(rw * phi)); rem = rw - sq
                split_x = rx + rw - sq
                cv2.line(overlay, (split_x, ry), (split_x, ry + rh - 1),
                         rect_color, thickness, cv2.LINE_AA)
                cv2.ellipse(overlay, (split_x, ry), (sq, rh), 0, 0, 90,
                            spiral_color, spiral_thick, cv2.LINE_AA)
                rw = rem
            else:
                sq = int(round(rh * phi)); rem = rh - sq
                split_y = ry + rh - sq
                cv2.line(overlay, (rx, split_y), (rx + rw - 1, split_y),
                         rect_color, thickness, cv2.LINE_AA)
                cv2.ellipse(overlay, (rx + rw, split_y), (rw, sq), 0, 90, 180,
                            spiral_color, spiral_thick, cv2.LINE_AA)
                rh = rem
            if rw < 2 or rh < 2:
                break

    # Pre-scale by alpha so per-frame work is a single cv2.add
    alpha = 0.35
    return (overlay.astype(np.float32) * alpha).astype(np.uint8)


def _draw_grid_overlay(disp, grid_key):
    """Draw translucent grid overlay on the camera preview.

    The overlay is built once and cached; per-frame cost is a single
    saturating add (cv2.add) with no allocations or masking.
    """
    if grid_key == "off":
        return
    h, w = disp.shape[:2]
    if h < 10 or w < 10:
        return

    global _grid_overlay_cache
    cache_key = (w, h, grid_key)
    if cache_key not in _grid_overlay_cache:
        # Replace cache (single entry) to avoid unbounded memory
        _grid_overlay_cache = {cache_key: _build_grid_overlay(w, h, grid_key)}

    cv2.add(disp, _grid_overlay_cache[cache_key], disp)


def _draw_iso_controls(dst, top, right_edge, size=72, margin=16):
    global _iso_toggle_rect, _iso_popup_rects, _iso_popup_bounds
    with _iso_button_lock:
        active_idx = _iso_idx
        popup_visible = _iso_popup_visible

    bounds = _icon_button_bounds(dst.shape, top, right_edge, size, margin)
    option = ISO_OPTIONS[active_idx]
    manual_selected = option["gain"] is not None
    icon, mask = _render_icon_button(
        dst,
        bounds,
        active=popup_visible or manual_selected,
        accent_color=UI_ACCENT_EXPOSURE if manual_selected else UI_ACCENT_CREATIVE,
    )
    _draw_iso_icon(icon, manual_selected, popup_visible)
    drawn_rect = _draw_icon_to_dst(dst, icon, mask, bounds)

    iso_labels = [opt["label"] for opt in ISO_OPTIONS]
    popup_rects, popup_bounds = _render_popup_grid(
        dst, bounds, iso_labels, active_idx, _iso_popup_fade, popup_visible,
        cols=min(2, len(ISO_OPTIONS)) or 1, font_size=18, position="left", margin=margin,
    )

    with _iso_button_lock:
        _iso_toggle_rect = drawn_rect if drawn_rect else bounds
        if popup_visible:
            _iso_popup_rects = popup_rects
            _iso_popup_bounds = popup_bounds
        else:
            _iso_popup_rects = []
            _iso_popup_bounds = None

    return drawn_rect if drawn_rect else bounds


def _set_aspect_ratio(idx):
    global _aspect_idx, _aspect_popup_visible, _aspect_popup_rects, _aspect_popup_bounds
    global _aspect_ratio_target
    with _aspect_button_lock:
        idx = max(0, min(idx, len(ASPECT_OPTIONS) - 1))
        if idx == _aspect_idx:
            _aspect_popup_visible = False
            _aspect_popup_rects = []
            _aspect_popup_bounds = None
        else:
            _aspect_idx = idx
            _aspect_ratio_target = ASPECT_OPTIONS[idx]["ratio"]
            _aspect_popup_visible = False
            _aspect_popup_rects = []
            _aspect_popup_bounds = None
    _record_activity(wake=True)
    _debounced_save_settings()

def _current_aspect_ratio():
    with _aspect_button_lock:
        return ASPECT_OPTIONS[_aspect_idx]["ratio"]

def _set_film_profile(idx):
    global _film_idx, _film_popup_visible, _film_popup_rects, _film_popup_bounds
    with _film_button_lock:
        idx = max(0, min(idx, len(FILM_OPTIONS) - 1))
        if idx == _film_idx:
            _film_popup_visible = False
            _film_popup_rects = []
            _film_popup_bounds = None
        else:
            _film_idx = idx
            _film_popup_visible = False
            _film_popup_rects = []
            _film_popup_bounds = None
    _record_activity(wake=True)
    _debounced_save_settings()

def _current_film_key():
    with _film_button_lock:
        return FILM_OPTIONS[_film_idx]["key"]

def _current_film_label():
    with _film_button_lock:
        return FILM_OPTIONS[_film_idx]["label"]


def _set_shutter_option(idx):
    global current_shutter_idx, _shutter_popup_visible, _shutter_popup_rects, _shutter_popup_bounds
    with _shutter_button_lock:
        idx = max(0, min(idx, len(SHUTTER_OPTIONS_US) - 1))
        if idx == current_shutter_idx:
            _shutter_popup_visible = False
            _shutter_popup_rects = []
            _shutter_popup_bounds = None
        else:
            current_shutter_idx = idx
            _shutter_popup_visible = False
            _shutter_popup_rects = []
            _shutter_popup_bounds = None
    _record_activity(wake=True)
    _debounced_save_settings()
    _apply_shutter_controls()


def _set_iso_option(idx):
    global _iso_idx, _iso_popup_visible, _iso_popup_rects, _iso_popup_bounds
    global _exp_comp_popup_visible, _exp_comp_popup_rects, _exp_comp_popup_bounds
    with _iso_button_lock:
        idx = max(0, min(idx, len(ISO_OPTIONS) - 1))
        # Check if user selected the "Exp Comp" submenu entry
        if ISO_OPTIONS[idx].get("gain") == "exp_comp":
            _iso_popup_visible = False
            _iso_popup_rects = []
            _iso_popup_bounds = None
    if ISO_OPTIONS[idx].get("gain") == "exp_comp":
        with _exp_comp_lock:
            _exp_comp_popup_visible = True
        _record_activity(wake=True)
        return
    with _iso_button_lock:
        if idx == _iso_idx:
            _iso_popup_visible = False
            _iso_popup_rects = []
            _iso_popup_bounds = None
        else:
            _iso_idx = idx
            _iso_popup_visible = False
            _iso_popup_rects = []
            _iso_popup_bounds = None
    _record_activity(wake=True)
    _debounced_save_settings()
    _apply_shutter_controls()


def _current_iso_gain():
    with _iso_button_lock:
        gain = ISO_OPTIONS[_iso_idx]["gain"]
        # "exp_comp" is a sentinel for the Exp Comp submenu, not a real gain
        if gain == "exp_comp":
            return None
        return gain


def _current_iso_label():
    with _iso_button_lock:
        return ISO_OPTIONS[_iso_idx]["label"]


def _current_exp_comp_value():
    with _exp_comp_lock:
        return EXP_COMP_OPTIONS[_exp_comp_idx]["value"]


def _current_exp_comp_label():
    with _exp_comp_lock:
        return EXP_COMP_OPTIONS[_exp_comp_idx]["label"]


def _set_exp_comp_option(idx):
    global _exp_comp_idx, _exp_comp_popup_visible, _exp_comp_popup_rects, _exp_comp_popup_bounds
    with _exp_comp_lock:
        idx = max(0, min(idx, len(EXP_COMP_OPTIONS) - 1))
        _exp_comp_idx = idx
        _exp_comp_popup_visible = False
        _exp_comp_popup_rects = []
        _exp_comp_popup_bounds = None
    _record_activity(wake=True)
    _debounced_save_settings()
    _apply_shutter_controls()


def _draw_exp_comp_popup(dst):
    """Draw the Exposure Compensation popup (shown after selecting Exp Comp from ISO menu)."""
    global _exp_comp_popup_rects, _exp_comp_popup_bounds
    with _exp_comp_lock:
        active_idx = _exp_comp_idx
        popup_visible = _exp_comp_popup_visible

    dst_h, dst_w = dst.shape[:2]
    margin = 16
    # Use a centered virtual bounds for the popup grid
    center_bounds = (dst_w // 2 - 36, dst_h // 2 - 36, dst_w // 2 + 36, dst_h // 2 + 36)
    exp_labels = [opt["label"] for opt in EXP_COMP_OPTIONS]
    popup_rects, popup_bounds = _render_popup_grid(
        dst, center_bounds, exp_labels, active_idx, _exp_comp_popup_fade, popup_visible,
        cols=len(EXP_COMP_OPTIONS), font_size=20, position="center", margin=margin,
    )

    with _exp_comp_lock:
        if popup_visible:
            _exp_comp_popup_rects = popup_rects
            _exp_comp_popup_bounds = popup_bounds
        else:
            _exp_comp_popup_rects = []
            _exp_comp_popup_bounds = None


def _apply_zoom(frame, zoom_factor, center_norm):
    if zoom_factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    crop_w = max(1, int(round(w / zoom_factor)))
    crop_h = max(1, int(round(h / zoom_factor)))
    cx = int(round(center_norm[0] * w))
    cy = int(round(center_norm[1] * h))
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    x1 = max(0, min(w - crop_w, cx - crop_w // 2))
    y1 = max(0, min(h - crop_h, cy - crop_h // 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return frame[y1:y2, x1:x2]


def _get_double_preview_overlay(target_w, target_h, aspect_ratio, film_key, zoom_factor, zoom_center):
    global _double_preview_cache
    with _double_exposure_lock:
        if not double_exposure_enabled:
            return None
        first_frame = double_exposure_first_frame
        cache_key = (target_w, target_h, aspect_ratio, film_key, zoom_factor, zoom_center)
        cached = _double_preview_cache.get("image") if _double_preview_cache.get("key") == cache_key else None
    if first_frame is None:
        return None
    if cached is not None:
        return cached

    overlay = first_frame
    if overlay.ndim == 3 and overlay.shape[2] == 4:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGB)

    if zoom_factor > 1.0 and zoom_center is not None:
        overlay = _apply_zoom(overlay, zoom_factor, zoom_center)

    if aspect_ratio > 0:
        crop_w, crop_h = _largest_ratio_crop_dims(overlay.shape[1], overlay.shape[0], aspect_ratio)
        overlay = _center_crop(overlay, crop_w, crop_h)

    if target_w > 0 and target_h > 0:
        overlay = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if film_key and film_key != "none":
        overlay = apply_film_simulation_rgb(overlay, film_key)

    with _double_exposure_lock:
        _double_preview_cache = {"key": cache_key, "image": overlay}
    return overlay

def _dismiss_all_popups(except_lock=None):
    """Close all popup menus. Optionally skip one lock (the popup being opened)."""
    _popup_locks = [
        (_focus_button_lock, "_focus_popup_visible", "_focus_popup_rects", "_focus_popup_bounds"),
        (_film_button_lock, "_film_popup_visible", "_film_popup_rects", "_film_popup_bounds"),
        (_aspect_button_lock, "_aspect_popup_visible", "_aspect_popup_rects", "_aspect_popup_bounds"),
        (_iso_button_lock, "_iso_popup_visible", "_iso_popup_rects", "_iso_popup_bounds"),
        (_shutter_button_lock, "_shutter_popup_visible", "_shutter_popup_rects", "_shutter_popup_bounds"),
        (_exp_comp_lock, "_exp_comp_popup_visible", "_exp_comp_popup_rects", "_exp_comp_popup_bounds"),
        (_brenizer_lock, "_multishot_popup_visible", "_multishot_popup_rects", "_multishot_popup_bounds"),
    ]
    for lock, vis_name, rects_name, bounds_name in _popup_locks:
        if lock is except_lock:
            continue
        with lock:
            globals()[vis_name] = False
            globals()[rects_name] = []
            globals()[bounds_name] = None


def _on_touch(event, x, y, flags, param):
    global zoom_active, zoom_center_norm, _zoom_level_idx
    global _aspect_popup_visible, _aspect_popup_rects, _aspect_popup_bounds
    global _film_popup_visible, _film_popup_rects, _film_popup_bounds
    global _iso_popup_visible, _iso_popup_rects, _iso_popup_bounds
    global _shutter_popup_visible, _shutter_popup_rects, _shutter_popup_bounds
    global _focus_popup_visible, _focus_popup_rects, _focus_popup_bounds
    global _exp_comp_popup_visible, _exp_comp_popup_rects, _exp_comp_popup_bounds
    global _multishot_popup_visible
    global charge_mode_active, flip_mode, _delete_confirm_active
    global double_exposure_enabled, double_exposure_first_frame, _double_preview_cache
    global _brenizer_show_help, _brenizer_merge_toast_until
    global _swipe_start_x, _swipe_start_y, _swipe_start_time
    global _touch_down_pos

    # Track swipe start on button-down
    if event == cv2.EVENT_LBUTTONDOWN:
        _swipe_start_x = x
        _swipe_start_y = y
        _swipe_start_time = time.time()
        # Store press position in canvas coords for visual feedback
        tx, ty = x, y
        if flip_mode == FLIP_MODE_MIRROR:
            tx = SCREEN_W - 1 - tx
        elif flip_mode == FLIP_MODE_MIRROR_ROTATE:
            ty = SCREEN_H - 1 - ty
        with _touch_down_lock:
            _touch_down_pos = (tx, ty)
        return

    if event != cv2.EVENT_LBUTTONUP:
        return

    # Clear press highlight on finger lift
    with _touch_down_lock:
        _touch_down_pos = None

    # Detect swipe gestures (blocked during Brenizer capture)
    if _swipe_start_x is not None and _swipe_start_time is not None:
        dx = _swipe_start_x - x  # positive = leftward swipe
        dy = abs(y - _swipe_start_y) if _swipe_start_y is not None else 0
        dy_signed = (y - _swipe_start_y) if _swipe_start_y is not None else 0  # positive = downward
        dy_up = -dy_signed  # positive = upward
        elapsed = time.time() - _swipe_start_time
        _swipe_start_x = None
        _swipe_start_y = None
        _swipe_start_time = None

        # Block all swipe gestures during Brenizer capture to prevent
        # accidental shutter speed changes or sleep mode activation
        if not _brenizer_active:
            # LEFT SWIPE: increase shutter speed (faster)
            if dx >= _SWIPE_MIN_DISTANCE and dy <= _SWIPE_MAX_VERTICAL and elapsed <= _SWIPE_MAX_TIME:
                print(f"[UI] Left swipe detected – increasing shutter speed")
                _cycle_shutter()
                return

            # RIGHT SWIPE: decrease shutter speed (slower / toward Auto)
            if -dx >= _SWIPE_MIN_DISTANCE and dy <= _SWIPE_MAX_VERTICAL and elapsed <= _SWIPE_MAX_TIME:
                print(f"[UI] Right swipe detected – decreasing shutter speed")
                _cycle_shutter_reverse()
                return

            # UPWARD SWIPE: enter sleep mode (strict thresholds to avoid
            # accidental activation during tap-to-zoom)
            if (dy_up >= _SWIPE_UP_MIN_DISTANCE
                    and abs(dx) <= _SWIPE_UP_MAX_HORIZONTAL
                    and elapsed <= _SWIPE_UP_MAX_TIME):
                print("[UI] Swipe-up detected – entering sleep mode")
                _enter_sleep_mode()
                return

            # DOWNWARD SWIPE: toggle on-screen icons (minimal mode)
            if (dy_signed >= _SWIPE_DOWN_MIN_DISTANCE
                    and abs(dx) <= _SWIPE_DOWN_MAX_HORIZONTAL
                    and elapsed <= _SWIPE_DOWN_MAX_TIME):
                print("[UI] Swipe-down detected – toggling on-screen icons")
                _toggle_minimal_mode()
                return

    if flip_mode == FLIP_MODE_MIRROR:
        x = SCREEN_W - 1 - x
    elif flip_mode == FLIP_MODE_MIRROR_ROTATE:
        y = SCREEN_H - 1 - y

    # --- Brenizer mode touch handling ---
    with _brenizer_lock:
        bren_active = _brenizer_active
        bren_state = _brenizer_state
        bren_help = _brenizer_show_help

    if bren_active:
        if bren_help:
            # Tap anywhere to dismiss help overlay
            _brenizer_show_help = False
            return
        if bren_state == "complete":
            # Handle completion screen buttons
            for key, rect in _brenizer_complete_rects.items():
                if _point_in_rect(x, y, rect):
                    if key == "done":
                        _exit_brenizer_mode()
                    elif key == "retake":
                        _brenizer_retake_last()
                    elif key == "merge":
                        _brenizer_merge_toast_until = time.monotonic() + 3.0
                    return
            return  # Block other touches on completion screen
        # During active capture, block all touch except shutter (GPIO)
        return

    # --- Video mode touch handling: block UI changes while recording ---
    with _video_lock:
        _vid_touch_recording = _video_recording
    if _vid_touch_recording:
        # While actively recording, block all touch — shutter button (GPIO) controls stop
        return

    if charge_mode_active:
        # Handle delete confirmation overlay first if active
        if _delete_confirm_active:
            if _point_in_rect(x, y, _delete_confirm_yes_rect):
                _delete_pictures_files()
                _delete_confirm_active = False
                _charge_stats_cache["timestamp"] = 0.0
                return
            # Cancel or tap anywhere else dismisses the confirmation
            _delete_confirm_active = False
            return
        # In Charge mode, allow tapping the Sleep button (and Exit) instead of always closing.
        if _point_in_rect(x, y, _sleep_button_rect):
            _exit_charge_mode()
            _enter_sleep_mode()
            return
        # Trash button opens delete confirmation
        if _point_in_rect(x, y, _trash_button_rect):
            _delete_confirm_active = True
            return
        # Exit button (X) terminates the application and returns to desktop
        if _point_in_rect(x, y, _exit_button_rect):
            _terminate_application()
            return
        # Default behaviour: tap anywhere else to exit charge mode
        _exit_charge_mode()
        return
    _record_activity(wake=True)
    with _film_button_lock:
        film_toggle_rect = _film_toggle_rect
        film_popup_visible = _film_popup_visible
        film_popup_rects = list(_film_popup_rects)
        film_popup_bounds = _film_popup_bounds
    with _aspect_button_lock:
        toggle_rect = _aspect_toggle_rect
        popup_visible = _aspect_popup_visible
        popup_rects = list(_aspect_popup_rects)
        popup_bounds = _aspect_popup_bounds
    with _shutter_button_lock:
        shutter_toggle_rect = _shutter_toggle_rect
        shutter_popup_visible = _shutter_popup_visible
        shutter_popup_rects = list(_shutter_popup_rects)
        shutter_popup_bounds = _shutter_popup_bounds
    with _iso_button_lock:
        iso_toggle_rect = _iso_toggle_rect
        iso_popup_visible = _iso_popup_visible
        iso_popup_rects = list(_iso_popup_rects)
        iso_popup_bounds = _iso_popup_bounds
    with _focus_button_lock:
        focus_toggle_rect = _focus_toggle_rect
        focus_popup_visible = _focus_popup_visible
        focus_popup_rects = list(_focus_popup_rects)
        focus_popup_bounds = _focus_popup_bounds
    with _exp_comp_lock:
        exp_comp_popup_visible = _exp_comp_popup_visible
        exp_comp_popup_rects = list(_exp_comp_popup_rects)
        exp_comp_popup_bounds = _exp_comp_popup_bounds
    if exp_comp_popup_visible:
        for idx, rect in exp_comp_popup_rects:
            if _point_in_rect(x, y, rect):
                _set_exp_comp_option(idx)
                return
        if exp_comp_popup_bounds is not None and _point_in_rect(x, y, exp_comp_popup_bounds):
            return
        with _exp_comp_lock:
            _exp_comp_popup_visible = False
            _exp_comp_popup_rects = []
            _exp_comp_popup_bounds = None
        return
    if focus_popup_visible:
        for idx, rect in focus_popup_rects:
            if _point_in_rect(x, y, rect):
                _set_focus_mode(idx)
                return
        if focus_toggle_rect and _point_in_rect(x, y, focus_toggle_rect):
            with _focus_button_lock:
                _focus_popup_visible = False
                _focus_popup_rects = []
                _focus_popup_bounds = None
            return
        if focus_popup_bounds is not None and _point_in_rect(x, y, focus_popup_bounds):
            return
        with _focus_button_lock:
            _focus_popup_visible = False
            _focus_popup_rects = []
            _focus_popup_bounds = None
        return
    if shutter_popup_visible:
        for idx, rect in shutter_popup_rects:
            if _point_in_rect(x, y, rect):
                _set_shutter_option(idx)
                return
        if shutter_toggle_rect and _point_in_rect(x, y, shutter_toggle_rect):
            with _shutter_button_lock:
                _shutter_popup_visible = False
                _shutter_popup_rects = []
                _shutter_popup_bounds = None
            return
        if shutter_popup_bounds is not None and _point_in_rect(x, y, shutter_popup_bounds):
            return
        with _shutter_button_lock:
            _shutter_popup_visible = False
            _shutter_popup_rects = []
            _shutter_popup_bounds = None
        return
    if iso_popup_visible:
        for idx, rect in iso_popup_rects:
            if _point_in_rect(x, y, rect):
                _set_iso_option(idx)
                return
        if iso_toggle_rect and _point_in_rect(x, y, iso_toggle_rect):
            with _iso_button_lock:
                _iso_popup_visible = False
                _iso_popup_rects = []
                _iso_popup_bounds = None
            return
        if iso_popup_bounds is not None and _point_in_rect(x, y, iso_popup_bounds):
            return
        with _iso_button_lock:
            _iso_popup_visible = False
            _iso_popup_rects = []
            _iso_popup_bounds = None
        return
    if film_popup_visible:
        for idx, rect in film_popup_rects:
            if _point_in_rect(x, y, rect):
                _set_film_profile(idx)
                return
        if film_toggle_rect and _point_in_rect(x, y, film_toggle_rect):
            with _film_button_lock:
                _film_popup_visible = False
                _film_popup_rects = []
                _film_popup_bounds = None
            return
        if film_popup_bounds is not None and _point_in_rect(x, y, film_popup_bounds):
            return
        with _film_button_lock:
            _film_popup_visible = False
            _film_popup_rects = []
            _film_popup_bounds = None
        return
    if popup_visible:
        for idx, rect in popup_rects:
            if _point_in_rect(x, y, rect):
                if idx >= _GRID_POPUP_IDX_OFFSET:
                    _set_grid_overlay(idx - _GRID_POPUP_IDX_OFFSET)
                else:
                    _set_aspect_ratio(idx)
                return
        if toggle_rect and _point_in_rect(x, y, toggle_rect):
            with _aspect_button_lock:
                _aspect_popup_visible = False
                _aspect_popup_rects = []
                _aspect_popup_bounds = None
            return
        if popup_bounds is not None and _point_in_rect(x, y, popup_bounds):
            return
        with _aspect_button_lock:
            _aspect_popup_visible = False
            _aspect_popup_rects = []
            _aspect_popup_bounds = None
        return
    if _point_in_rect(x, y, _charge_button_rect):
        if charge_mode_active:
            _exit_charge_mode()
        else:
            _enter_charge_mode()
        return
    if _point_in_rect(x, y, _sleep_button_rect):
        # Sleep button is only shown from Charge mode
        if charge_mode_active:
            _exit_charge_mode()
            _enter_sleep_mode()
        return
    if _point_in_rect(x, y, _flip_button_rect):
        if flip_mode == FLIP_MODE_NORMAL:
            flip_mode = FLIP_MODE_MIRROR
        elif flip_mode == FLIP_MODE_MIRROR:
            flip_mode = FLIP_MODE_MIRROR_ROTATE
        else:
            flip_mode = FLIP_MODE_NORMAL
        _debounced_save_settings()
        return
    if _point_in_rect(x, y, _double_exposure_button_rect):
        _dismiss_all_popups(except_lock=_brenizer_lock)
        _multishot_popup_visible = True
        return
    # Handle multishot popup item taps (Double Exp / Brenizer / Video)
    if _multishot_popup_visible and _multishot_popup_rects:
        for idx, rect in _multishot_popup_rects:
            if _point_in_rect(x, y, rect):
                _multishot_popup_visible = False
                opt_key = MULTISHOT_OPTIONS[idx]["key"]
                if opt_key == "double_exposure":
                    # Exit video mode if active before toggling DE
                    if _video_mode_active:
                        _exit_video_mode()
                    with _double_exposure_lock:
                        double_exposure_enabled = not double_exposure_enabled
                    _clear_double_exposure_state()
                elif opt_key == "brenizer_5":
                    if _video_mode_active:
                        _exit_video_mode()
                    _enter_brenizer_mode(5)
                elif opt_key == "brenizer_6":
                    if _video_mode_active:
                        _exit_video_mode()
                    _enter_brenizer_mode(6)
                elif opt_key == "brenizer_9":
                    if _video_mode_active:
                        _exit_video_mode()
                    _enter_brenizer_mode(9)
                elif opt_key == "video_1080p":
                    if _video_mode_active and _video_resolution_idx == 0:
                        _exit_video_mode()  # Toggle off
                    else:
                        if _video_mode_active:
                            _exit_video_mode()
                        _enter_video_mode(0)
                elif opt_key == "video_4k":
                    if _video_mode_active and _video_resolution_idx == 1:
                        _exit_video_mode()  # Toggle off
                    else:
                        if _video_mode_active:
                            _exit_video_mode()
                        _enter_video_mode(1)
                return
        # Tap outside popup bounds — dismiss
        if _multishot_popup_bounds and not _point_in_rect(x, y, _multishot_popup_bounds):
            _multishot_popup_visible = False
            return
    if focus_toggle_rect and _point_in_rect(x, y, focus_toggle_rect):
        _dismiss_all_popups(except_lock=_focus_button_lock)
        with _focus_button_lock:
            _focus_popup_visible = True
        return
    if film_toggle_rect and _point_in_rect(x, y, film_toggle_rect):
        _dismiss_all_popups(except_lock=_film_button_lock)
        with _film_button_lock:
            _film_popup_visible = True
        return
    if toggle_rect and _point_in_rect(x, y, toggle_rect):
        _dismiss_all_popups(except_lock=_aspect_button_lock)
        with _aspect_button_lock:
            _aspect_popup_visible = True
        return
    if iso_toggle_rect and _point_in_rect(x, y, iso_toggle_rect):
        _dismiss_all_popups(except_lock=_iso_button_lock)
        with _iso_button_lock:
            _iso_popup_visible = True
        return
    if shutter_toggle_rect and _point_in_rect(x, y, shutter_toggle_rect):
        _dismiss_all_popups(except_lock=_shutter_button_lock)
        with _shutter_button_lock:
            _shutter_popup_visible = True
        return
    if _point_in_rect(x, y, _wifi_button_rect):
        # Show IP/credentials overlay for a limited time after pressing the Wi-Fi button
        global _wifi_info_until
        _wifi_info_until = time.monotonic() + WIFI_INFO_DURATION_S
        _toggle_wifi()
        return
    barrier_rects = [
        _charge_button_rect,
        _sleep_button_rect,
        _flip_button_rect,
        _double_exposure_button_rect,
        _rangefinder_button_rect,
        _wifi_button_rect,
        _exit_button_rect,
        film_toggle_rect,
        toggle_rect,
        iso_toggle_rect,
        shutter_toggle_rect,
        film_popup_bounds,
        popup_bounds,
        iso_popup_bounds,
        shutter_popup_bounds,
    ]
    for rect in barrier_rects:
        if _point_in_rect_padded(x, y, rect, TAP_ZOOM_BARRIER_PAD):
            return
    px, py, pw, ph = _preview_geom["x"], _preview_geom["y"], _preview_geom["w"], _preview_geom["h"]
    if pw <= 0 or ph <= 0:
        return
    if not (px <= x < px + pw and py <= y < py + ph):
        return
    rel_x = (x - px) / float(pw)
    rel_y = (y - py) / float(ph)
    rel_x = min(0.98, max(0.02, rel_x))
    rel_y = min(0.98, max(0.02, rel_y))
    if zoom_active:
        dist = np.hypot(rel_x - zoom_center_norm[0], rel_y - zoom_center_norm[1])
        if dist < 0.08 and len(ZOOM_LEVELS) > 0:
            _zoom_level_idx = (_zoom_level_idx + 1) % (len(ZOOM_LEVELS) + 1)
            if _zoom_level_idx >= len(ZOOM_LEVELS):
                zoom_active = False
                _zoom_level_idx = 0
    else:
        zoom_active = True
        _zoom_level_idx = 0
    zoom_center_norm = (rel_x, rel_y)

def _pil_render_text_block(lines, font_scale, thickness, rotate_180, max_h, pad_x, pad_y, line_gap):
    """Render a text block using PIL/Pillow TrueType fonts.

    Segments may use per-segment font_scale via dict:
        {"text": "SS ", "font_scale": 1.2}
    Segments without an explicit font_scale inherit the block-level font_scale.

    Returns a numpy (H, W, 3) image, or None if PIL is unavailable.
    """
    if not _PIL_AVAILABLE:
        return None

    render_scale = max(1.0, float(TEXT_RENDER_SCALE))
    # Map cv2 HERSHEY_PLAIN scale to approximate PIL font size.
    # HERSHEY_PLAIN scale 1.0 ≈ 10 px cap height; PIL needs ~13 px for similar.
    base_pil_size = max(8, int(round(font_scale * render_scale * 13)))
    base_font = _load_pil_font(base_pil_size)
    if base_font is None:
        return None

    scaled_pad_x = int(round(pad_x * render_scale))
    scaled_pad_y = int(round(pad_y * render_scale))
    scaled_line_gap = int(round(line_gap * render_scale))

    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    # Parse each line into measured segments, each with its own font.
    measured_lines = []
    max_w = 0
    total_h = 2 * scaled_pad_y
    for idx, raw_line in enumerate(lines):
        segments = []
        if isinstance(raw_line, (list, tuple)) and not isinstance(raw_line, str):
            for seg in raw_line:
                if isinstance(seg, dict):
                    text = seg.get("text", "")
                    seg_font_scale = seg.get("font_scale", None)
                    seg_color = seg.get("color", None)
                elif isinstance(seg, (list, tuple)) and seg:
                    text = str(seg[0])
                    seg_font_scale = None
                    seg_color = None
                else:
                    text = str(seg)
                    seg_font_scale = None
                    seg_color = None
                if not text:
                    continue
                if seg_font_scale is not None:
                    pil_size = max(8, int(round(seg_font_scale * render_scale * 13)))
                    seg_font = _load_pil_font(pil_size) or base_font
                else:
                    seg_font = base_font
                bbox = draw.textbbox((0, 0), text, font=seg_font)
                segments.append({
                    "text": text, "font": seg_font,
                    "w": max(1, bbox[2] - bbox[0]),
                    "x_off": -bbox[0], "y_off": -bbox[1],
                    "ascent": -bbox[1], "descent": bbox[3],
                    "color": seg_color,
                })
        else:
            text = str(raw_line) or " "
            bbox = draw.textbbox((0, 0), text, font=base_font)
            segments.append({
                "text": text, "font": base_font,
                "w": max(1, bbox[2] - bbox[0]),
                "x_off": -bbox[0], "y_off": -bbox[1],
                "ascent": -bbox[1], "descent": bbox[3],
            })

        if not segments:
            bbox = draw.textbbox((0, 0), " ", font=base_font)
            segments.append({
                "text": " ", "font": base_font,
                "w": max(1, bbox[2] - bbox[0]),
                "x_off": -bbox[0], "y_off": -bbox[1],
                "ascent": -bbox[1], "descent": bbox[3],
            })

        line_width = sum(s["w"] for s in segments)
        line_ascent = max(s["ascent"] for s in segments)
        line_descent = max(s["descent"] for s in segments)
        line_height = line_ascent + line_descent

        measured_lines.append({
            "segments": segments, "width": line_width,
            "height": max(1, line_height), "ascent": line_ascent,
        })
        max_w = max(max_w, line_width)
        total_h += max(1, line_height)
        if idx < len(lines) - 1:
            total_h += scaled_line_gap

    img_w = max(1, max_w + 2 * scaled_pad_x)
    img_h = max(1, total_h)

    pil_img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    y = scaled_pad_y
    for idx, line in enumerate(measured_lines):
        x = scaled_pad_x
        for seg in line["segments"]:
            # Align all segments to the line's common baseline
            baseline_y = y + line["ascent"]
            seg_y = baseline_y - seg["ascent"]
            draw.text((x + seg["x_off"], seg_y + seg["y_off"]),
                      seg["text"], fill=seg.get("color") or (255, 255, 255), font=seg["font"])
            x += seg["w"]
        y += line["height"]
        if idx < len(measured_lines) - 1:
            y += scaled_line_gap

    img = np.array(pil_img)

    if rotate_180:
        img = cv2.rotate(img, cv2.ROTATE_180)

    target_h = max(1, int(round(img.shape[0] / render_scale)))
    target_w = max(1, int(round(img.shape[1] / render_scale)))
    if target_h > max_h:
        resize_scale = max_h / float(target_h)
        target_h = max_h
        target_w = max(1, int(round(target_w * resize_scale)))

    if target_h != img.shape[0] or target_w != img.shape[1]:
        img = cv2.resize(img, (target_w, target_h),
                         interpolation=cv2.INTER_AREA if target_h < img.shape[0] else cv2.INTER_LINEAR)
    return img


_text_block_cache = {}
_TEXT_BLOCK_CACHE_MAX = 32

# Histogram throttle — compute every 2nd frame, reuse cached result on skip frames
_hist_cached_img = None
_hist_frame_toggle = False


def make_rotated_text_block(
    lines,
    font_scale=0.7,
    thickness=1,
    rotate_180=True,
    max_h=BAR_H - 8,
    pad_x=8,
    pad_y=6,
    line_gap=6,
):
    # Build a hashable key from the text content and rendering parameters.
    # Each segment becomes a (text, font_scale_or_None, color_or_None) tuple
    # so that segment boundaries are preserved and differently-styled inputs
    # never collide.
    key_lines = []
    for raw_line in lines:
        if isinstance(raw_line, (list, tuple)) and not isinstance(raw_line, str):
            segs = []
            for seg in raw_line:
                if isinstance(seg, dict):
                    segs.append((seg.get("text", ""), seg.get("font_scale", None), seg.get("color", None)))
                elif isinstance(seg, (list, tuple)) and seg:
                    segs.append((str(seg[0]), None, None))
                else:
                    segs.append((str(seg), None, None))
            key_lines.append(tuple(segs))
        else:
            key_lines.append(((str(raw_line), None, None),))
    cache_key = (tuple(key_lines), font_scale, thickness, rotate_180, max_h, pad_x, pad_y, line_gap)
    cached = _text_block_cache.get(cache_key)
    if cached is not None:
        return cached

    # Try PIL/TrueType first for cleaner text rendering.
    if _PIL_AVAILABLE:
        result = _pil_render_text_block(
            lines, font_scale, thickness, rotate_180, max_h, pad_x, pad_y, line_gap
        )
        if result is not None:
            if len(_text_block_cache) >= _TEXT_BLOCK_CACHE_MAX:
                try:
                    _text_block_cache.pop(next(iter(_text_block_cache)))
                except Exception:
                    _text_block_cache.clear()
            _text_block_cache[cache_key] = result
            return result

    # ---- Fallback: cv2 Hershey rendering ----
    font = cv2.FONT_HERSHEY_PLAIN

    render_scale = max(1.0, float(TEXT_RENDER_SCALE))
    scaled_font_scale = font_scale * render_scale
    scaled_thickness = max(1, int(round(thickness * render_scale)))
    scaled_pad_x = int(round(pad_x * render_scale))
    scaled_pad_y = int(round(pad_y * render_scale))
    scaled_line_gap = int(round(line_gap * render_scale))

    processed = []
    max_line_width = 0
    total_height = 2 * scaled_pad_y

    line_count = len(lines)
    for idx, raw_line in enumerate(lines):
        if isinstance(raw_line, (list, tuple)) and not isinstance(raw_line, str):
            segments = []
            for seg in raw_line:
                if isinstance(seg, dict):
                    text = seg.get("text", "")
                    seg_fs_override = seg.get("font_scale", None)
                    seg_color = seg.get("color", None)
                elif isinstance(seg, (list, tuple)) and seg:
                    text = str(seg[0])
                    seg_fs_override = None
                    seg_color = None
                else:
                    text = str(seg)
                    seg_fs_override = None
                    seg_color = None
                if not text:
                    continue
                if seg_fs_override is not None:
                    seg_scaled_fs = seg_fs_override * render_scale
                    seg_thick = max(1, int(round(thickness * render_scale * seg_fs_override / font_scale)))
                else:
                    seg_scaled_fs = scaled_font_scale
                    seg_thick = scaled_thickness
                size, base = cv2.getTextSize(text, font, seg_scaled_fs, seg_thick)
                segments.append(
                    {
                        "text": text,
                        "font_scale": seg_scaled_fs,
                        "thickness": seg_thick,
                        "width": size[0],
                        "height": size[1],
                        "baseline": base,
                        "color": seg_color,
                    }
                )
        else:
            text = str(raw_line)
            if not text:
                text = " "
            size, base = cv2.getTextSize(text, font, scaled_font_scale, scaled_thickness)
            segments = [
                {
                    "text": text,
                    "font_scale": scaled_font_scale,
                    "thickness": scaled_thickness,
                    "width": size[0],
                    "height": size[1],
                    "baseline": base,
                }
            ]

        if not segments:
            size, base = cv2.getTextSize(" ", font, scaled_font_scale, scaled_thickness)
            segments = [
                {
                    "text": " ",
                    "font_scale": scaled_font_scale,
                    "thickness": scaled_thickness,
                    "width": size[0],
                    "height": size[1],
                    "baseline": base,
                }
            ]

        line_width = sum(seg["width"] for seg in segments)
        line_height = max(seg["height"] for seg in segments)

        processed.append({"segments": segments, "width": line_width, "height": line_height})
        max_line_width = max(max_line_width, line_width)
        total_height += line_height
        if idx < line_count - 1:
            total_height += scaled_line_gap

    width = max_line_width + 2 * scaled_pad_x
    height = max(total_height, 1)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    y = scaled_pad_y
    for idx, line in enumerate(processed):
        y += line["height"]
        x = scaled_pad_x
        for seg in line["segments"]:
            _sc = seg.get("color")
            _bgr = (_sc[2], _sc[1], _sc[0]) if _sc else (255, 255, 255)
            cv2.putText(
                img,
                seg["text"],
                (x, y),
                font,
                seg.get("font_scale", scaled_font_scale),
                _bgr,
                seg["thickness"],
                cv2.LINE_AA,
            )
            x += seg["width"]
        if idx < len(processed) - 1:
            y += scaled_line_gap

    if rotate_180:
        img = cv2.rotate(img, cv2.ROTATE_180)

    target_h = max(1, int(round(img.shape[0] / render_scale)))
    target_w = max(1, int(round(img.shape[1] / render_scale)))
    if target_h > max_h:
        resize_scale = max_h / float(target_h)
        target_h = max_h
        target_w = max(1, int(round(target_w * resize_scale)))

    if target_h != img.shape[0] or target_w != img.shape[1]:
        img = cv2.resize(
            img,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA if (target_h < img.shape[0]) else cv2.INTER_LINEAR,
        )
    if len(_text_block_cache) >= _TEXT_BLOCK_CACHE_MAX:
        try:
            _text_block_cache.pop(next(iter(_text_block_cache)))
        except Exception:
            _text_block_cache.clear()
    _text_block_cache[cache_key] = img
    return img

# -------------------- Update prompt --------------------

UPDATE_FILENAME = "update.py"
_UPDATE_WINDOW_NAME = "Camera Update"


def _wrap_update_lines(text, width=36):
    if not text:
        return [""]
    wrapped = []
    for raw_line in str(text).splitlines():
        segments = textwrap.wrap(raw_line, width=width) or [""]
        wrapped.extend(segments)
    return wrapped or [""]


def _find_update_file():
    candidates = []
    candidates.append(Path(USB_MOUNT_POINT) / UPDATE_FILENAME)
    with _capture_dir_lock:
        capture_dir = _capture_dir
    if capture_dir:
        candidates.append(Path(capture_dir) / UPDATE_FILENAME)
    seen = set()
    for path in candidates:
        try:
            resolved = Path(path)
        except Exception:
            continue
        key = resolved.resolve().as_posix() if resolved.exists() else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            return resolved
    return None


def _apply_update_from_file(update_path):
    dest_path = Path(__file__).resolve()
    tmp_path = None
    try:
        data = Path(update_path).read_bytes()
    except Exception as exc:
        return False, f"Read error: {exc}"
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{dest_path.stem}_update_",
            suffix=".tmp",
            dir=str(dest_path.parent),
        )
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, dest_path)
        os.chmod(dest_path, 0o755)
    except Exception as exc:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False, f"Write error: {exc}"
    try:
        Path(update_path).unlink()
    except Exception as exc:
        print("Update cleanup error:", exc)
    return True, None


def _finalize_update_and_reboot():
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        os.sync()
    except Exception:
        pass
    last_exc = None
    for cmd in (["sudo", "reboot"], ["reboot"]):
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            return True, None
        except Exception as exc:
            last_exc = exc
    if last_exc is None:
        last_exc = RuntimeError("Unable to execute reboot command")
    return False, f"Reboot failed: {last_exc}"


def _show_update_error(window_name, message):
    acknowledged = {"value": False}

    def _ack_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            acknowledged["value"] = True

    cv2.setMouseCallback(window_name, _ack_cb)
    while not acknowledged["value"]:
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        lines = [["Update failed"]]
        for line in _wrap_update_lines(message):
            lines.append([line])
        lines.append(["Tap to continue"])
        block = make_rotated_text_block(
            lines,
            font_scale=1.0,
            thickness=1,
            rotate_180=True,
            max_h=SCREEN_H - 120,
            pad_x=12,
            pad_y=10,
            line_gap=8,
        )
        bh, bw = block.shape[:2]
        bx = max(0, (SCREEN_W - bw) // 2)
        by = max(0, (SCREEN_H - bh) // 2)
        blit_add(canvas, block, bx, by)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(50)
        if key != -1:
            acknowledged["value"] = True
    cv2.setMouseCallback(window_name, lambda *args: None)


def _prompt_for_update(update_path):
    window_name = _UPDATE_WINDOW_NAME
    selection = {"choice": None}
    btn_w = 180
    btn_h = 110
    center_y = SCREEN_H // 2 + 60
    yes_rect = (
        SCREEN_W // 2 - btn_w - 20,
        center_y,
        SCREEN_W // 2 - 20,
        center_y + btn_h,
    )
    no_rect = (
        SCREEN_W // 2 + 20,
        center_y,
        SCREEN_W // 2 + btn_w + 20,
        center_y + btn_h,
    )

    def _touch_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if _point_in_rect(x, y, yes_rect):
                selection["choice"] = True
            elif _point_in_rect(x, y, no_rect):
                selection["choice"] = False

    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, _touch_cb)

    while selection["choice"] is None:
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        lines = [
            ["Software update available"],
            [f"{update_path.name} detected on USB"],
            ["Install update now?"],
        ]
        block = make_rotated_text_block(
            lines,
            font_scale=1.15,
            thickness=1,
            rotate_180=True,
            max_h=SCREEN_H - 220,
            pad_x=14,
            pad_y=12,
            line_gap=10,
        )
        bh, bw = block.shape[:2]
        bx = max(0, (SCREEN_W - bw) // 2)
        by = max(40, (SCREEN_H - bh) // 2 - 60)
        blit_add(canvas, block, bx, by)

        for label, rect, color in (
            ("Yes", yes_rect, (60, 140, 80)),
            ("No", no_rect, (80, 80, 80)),
        ):
            x1, y1, x2, y2 = rect
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (230, 230, 230), 2)
            btn_block = make_rotated_text_block(
                [[label]],
                font_scale=1.05,
                thickness=1,
                rotate_180=True,
                max_h=max(20, (y2 - y1) - 24),
                pad_x=10,
                pad_y=8,
            )
            bh2, bw2 = btn_block.shape[:2]
            bbx = x1 + max(0, (x2 - x1 - bw2) // 2)
            bby = y1 + max(0, (y2 - y1 - bh2) // 2)
            blit_add(canvas, btn_block, bbx, bby)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(50)
        if key in (ord("y"), ord("Y")):
            selection["choice"] = True
        elif key in (ord("n"), ord("N"), 27):
            selection["choice"] = False

    cv2.setMouseCallback(window_name, lambda *args: None)
    choice = selection.get("choice")

    if choice and Path(update_path).is_file():
        success, error = _apply_update_from_file(update_path)
        if success:
            reboot_ok, reboot_err = _finalize_update_and_reboot()
            if reboot_ok:
                sys.exit(0)
            else:
                _show_update_error(window_name, reboot_err)
                choice = False
        else:
            _show_update_error(window_name, error)
            choice = False

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)
    return choice


def _check_for_pending_update(deadline=6.0):
    update_path = _find_update_file()
    if update_path is None:
        deadline_ts = time.monotonic() + max(0.0, float(deadline))
        while time.monotonic() < deadline_ts:
            time.sleep(0.3)
            update_path = _find_update_file()
            if update_path is not None:
                break
    if update_path and update_path.is_file():
        return update_path
    return None


_pending_update_lock = threading.Lock()
_pending_update_path = None
_update_prompt_ready = threading.Event()
_lazy_load_ready = threading.Event()


def _queue_update_prompt(update_path):
    global _pending_update_path
    if not update_path:
        return
    with _pending_update_lock:
        _pending_update_path = update_path
    _update_prompt_ready.set()


def _lazy_startup_loader():
    update_path = _check_for_pending_update()
    if update_path:
        _queue_update_prompt(update_path)

    # Preload film LUTs and other optional assets so they do not block startup.
    try:
        for key in list(_FILM_CURVE_DATA.keys()):
            _film_lut_for(key)
    except Exception as exc:
        print("Lazy load error:", exc)

    # Pre-warm icon button cache so the first few frames don't stutter while
    # rendering every button from scratch.  We use a tiny dummy destination
    # image just to satisfy the bounds calculation; only the (size, active,
    # accent_color, opacity) cache key matters.
    try:
        _btn_size = 72
        _dummy = np.zeros((_btn_size + 40, _btn_size + 40, 3), dtype=np.uint8)
        _warm_bounds = (0, 0, _btn_size, _btn_size)
        _warm_colors = [
            ((180, 180, 180), 0.9),       # default inactive
            (UI_ACCENT_SYSTEM, 0.9),      # flip / sleep
            (UI_ACCENT_EXPOSURE, 0.9),    # charge / shutter / iso manual
            (UI_ACCENT_SYSTEM, 0.85),     # sleep
            ((200, 200, 200), 0.85),      # exit / x-exit
            (UI_ACCENT_CREATIVE, 0.9),    # wifi / film / aspect / double-exp
            (UI_INACTIVE, 0.9),           # unified inactive
            (UI_ACCENT_FOCUS, 0.9),       # rangefinder active
        ]
        for _color, _opacity in _warm_colors:
            for _active in (False, True):
                _render_icon_button(_dummy, _warm_bounds, active=_active,
                                    accent_color=_color, inner_opacity=_opacity)
    except Exception as exc:
        print("Icon cache warm-up error:", exc)

    _lazy_load_ready.set()


# -------------------- Camera setup --------------------
_update_splash("Initializing camera...")
from picamera2 import Picamera2
picam2 = Picamera2()

# --- Sensor identification ---------------------------------------------------
# Known sensors the camera pipeline supports.  Each entry lists the full
# pixel-array size (used only as a fallback when libcamera doesn't report
# one) and whether the sensor is monochrome.  Add a row here to support a
# new sensor without touching anything else.
_KNOWN_SENSORS = {
    "imx585": {"full": (3856, 2180), "mono": True},   # legacy default
    "imx492": {"full": (8288, 5644), "mono": False},  # Sony 4/3" 47 MP
    "imx477": {"full": (4056, 3040), "mono": False},  # Raspberry Pi HQ
    "imx708": {"full": (4608, 2592), "mono": False},  # Pi Camera v3
    "imx219": {"full": (3280, 2464), "mono": False},  # Pi Camera v2
}

_sensor_model = str(picam2.camera_properties.get("Model", "")).lower()
_sensor_info = _KNOWN_SENSORS.get(_sensor_model, {"full": (3856, 2180), "mono": True})

FULL_W, FULL_H = picam2.camera_properties.get("PixelArraySize", (0, 0))
if not (FULL_W and FULL_H):
    FULL_W = FULL_H = 0
    for m in getattr(picam2, "sensor_modes", []):
        w, h = m.get("size", (0, 0))
        if w*h > FULL_W*FULL_H:
            FULL_W, FULL_H = w, h
if not (FULL_W and FULL_H):
    FULL_W, FULL_H = _sensor_info["full"]

# Detect monochrome vs. color from the libcamera ColorFilterArrangement
# property when it's exposed; otherwise fall back to the known-sensor
# table.  The libcamera enum uses 4 (MONO) for monochrome sensors — any
# other value is a Bayer pattern (RGGB/BGGR/GRBG/GBRG).
_cfa = picam2.camera_properties.get("ColorFilterArrangement", None)
if _cfa is None:
    IS_MONO_SENSOR = bool(_sensor_info["mono"])
else:
    try:
        IS_MONO_SENSOR = (int(_cfa) == 4)
    except (TypeError, ValueError):
        IS_MONO_SENSOR = bool(_sensor_info["mono"])

# Pick a native raw format for the still stream.  The original code
# hard-coded "R16" which only works for monochrome sensors (IMX585); a
# Bayer sensor like IMX492 needs an SRGGB/SBGGR format.  We query the
# advertised sensor modes to find the format libcamera actually exposes
# for the full-resolution capture and only fall back when nothing matches.
_raw_format = None
for m in getattr(picam2, "sensor_modes", []):
    if m.get("size") == (FULL_W, FULL_H):
        _raw_format = m.get("unpacked") or m.get("format")
        if _raw_format:
            break
if not _raw_format:
    # Last-resort fallback based on detected mono/color
    _raw_format = "R16" if IS_MONO_SENSOR else "SRGGB12"

# Adaptive buffer count: very large sensors (e.g. IMX492 at 47 MP) blow
# past the Pi 5's CMA allocation if we keep the legacy buffer_count=2, so
# drop to a single still buffer once the pixel count crosses ~20 MP.  The
# user gives up one frame of pipelining but the camera actually starts.
_STILL_BUFFER_COUNT = 1 if (FULL_W * FULL_H) > 20_000_000 else 2

print(
    f"[Camera] Sensor: {_sensor_model or 'unknown'} "
    f"{FULL_W}x{FULL_H} "
    f"{'mono' if IS_MONO_SENSOR else 'color'} "
    f"raw={_raw_format} buffers={_STILL_BUFFER_COUNT}"
)

preview_size = _choose_preview_size(FULL_W, FULL_H)
_geom_cache = {
    (preview_size[1], preview_size[0]): _compute_display_geometry(preview_size[0], preview_size[1])
}
preview_config = picam2.create_preview_configuration(
    main={"size": preview_size, "format": "RGB888"},
    lores=None,
    display=None,  # no internal preview thread
    raw=None
)
# Default "auto" range (1/8000..20s) – full sensor capability
DEFAULT_FRAME_LIMITS = (125, 20000000)
# Preview frame-rate floor: cap max exposure to 1/30s so the live view
# never drops below ~30 fps.  In AUTO mode the AE compensates with ISO;
# for manual slow-shutter speeds the preview is digitally brightened.
_PREVIEW_FRAME_LIMITS = (125, 33333)
preview_config["controls"]["AeMeteringMode"] = 2
preview_config["controls"]["NoiseReductionMode"] = 0
preview_config["controls"]["FrameDurationLimits"] = _PREVIEW_FRAME_LIMITS

_STILL_CONTROLS = {
    "AeMeteringMode": 2,
    "NoiseReductionMode": 0,
    "FrameDurationLimits": _PREVIEW_FRAME_LIMITS,
}


still_config = picam2.create_still_configuration(
    main={"size": (FULL_W, FULL_H), "format": "RGB888"},
    lores={"size": preview_size, "format": "RGB888"},
    raw={"size": (FULL_W, FULL_H), "format": _raw_format},
    controls=dict(_STILL_CONTROLS),
    buffer_count=_STILL_BUFFER_COUNT,
)

# Lightweight running config used during live view on large sensors.
# Producing a full-resolution RGB888 main stream every preview frame is
# what made the IMX492 (47 MP) unusable — the ISP simply can't scale
# 8288x5644 to RGB888 at 30 fps on a Pi 5.  For any sensor above ~20 MP
# we therefore run a small preview config (main-only, no raw) during
# the live view and briefly switch to `still_config` for the actual
# capture so JPG+DNG still come out at native resolution.
USE_LIGHTWEIGHT_PREVIEW = (FULL_W * FULL_H) > 20_000_000

# Log every advertised sensor mode — on an unfamiliar sensor like the
# IMX492 this is how you tell whether the driver actually exposes a
# binned readout or if you're stuck on the full array.  This is also
# why we *don't* explicitly pin a sensor mode via the `sensor=` kwarg
# any more: on some drivers (IMX492 among them) the advertised smaller
# modes are line-skipping rather than true binning, so forcing one of
# them bypasses libcamera's ISP alignment and produces severe banding
# on the preview.  Letting picamera2 auto-select via the main stream
# size is slower but produces clean pixels.
_all_sensor_modes = [m for m in getattr(picam2, "sensor_modes", []) if m.get("size")]
if _all_sensor_modes:
    print("[Camera] Advertised sensor modes:")
    for _m in _all_sensor_modes:
        print(
            f"  - {_m.get('size')} @ {_m.get('fps', '?')} fps "
            f"bit_depth={_m.get('bit_depth', '?')} "
            f"format={_m.get('unpacked') or _m.get('format')}"
        )

if USE_LIGHTWEIGHT_PREVIEW:
    # NB: this used to request RGB888.  On Pi 5 the ISP's YUV420 output
    # path costs roughly half the memory bandwidth of the equivalent
    # RGB888 at the same resolution, and cv2.cvtColor YUV→RGB is SIMD-
    # optimised so the per-frame conversion cost is far smaller than
    # what the ISP saves.  The rest of the pipeline still sees an RGB
    # frame — the conversion happens in the preview loop right after
    # capture_array.  (picamera2 requires main >= lores and always
    # has a main stream, so the answer to "can I use lores instead?"
    # is really "yes, by picking the cheap format for main" — which is
    # what we do here.)
    _preview_running_config = picam2.create_video_configuration(
        main={"size": preview_size, "format": "YUV420"},
        lores=None,
        raw=None,
        controls=dict(_STILL_CONTROLS),
        buffer_count=3,
    )
    PREVIEW_STREAM_NAME = "main"
    print(
        f"[Camera] Lightweight preview: main={preview_size} YUV420 "
        f"(full-res capture via switch_mode)"
    )
else:
    _preview_running_config = still_config
    PREVIEW_STREAM_NAME = "lores"

_aspect_capture_dims = [
    _largest_ratio_crop_dims(FULL_W, FULL_H, opt["ratio"]) for opt in ASPECT_OPTIONS
]

# Apply CPU thermal cap in background – sysfs writes don't need to block startup.
threading.Thread(target=_apply_cpu_thermal_cap, daemon=True).start()

# Run the live-view config.  On small sensors this is still_config (raw
# stream permanently available → instant capture); on large sensors it's
# the lightweight preview config and we switch modes for each capture.
_update_splash("Starting camera...")
picam2.configure(_preview_running_config)
picam2.start()
picam2.set_controls({"FrameDurationLimits": _PREVIEW_FRAME_LIMITS})

# On large sensors the per-frame CPU budget is tighter (bigger frames,
# more expensive debayer, color→luma conversion, etc.), so drop the
# focus-peaking preset to LOW by default.  The user can still raise it
# from the settings UI if they want.
if USE_LIGHTWEIGHT_PREVIEW:
    FOCUS_PERF_LEVEL = 0

# Pre-cache geometry for full resolution frames
_geom_cache[(FULL_H, FULL_W)] = _compute_display_geometry(FULL_W, FULL_H)

# Kick off lazy loaders so the preview appears immediately while secondary assets load.
threading.Thread(target=_lazy_startup_loader, daemon=True).start()

# -------------------- Shutter set mode (dual use BTN26) --------------------
# None = Auto; otherwise microseconds
SHUTTER_OPTIONS_US = [None, 125000, 66667, 33333, 16667, 8000, 4000, 2000, 1000, 500, 250]
SHUTTER_LABELS     = ["AUTO","1/8","1/15","1/30","1/60","1/125","1/250","1/500","1/1000","1/2000","1/4000"]

current_shutter_idx = _bounded_int(_PERSISTED_SETTINGS.get("shutter_idx", 0), 0, 0, len(SHUTTER_OPTIONS_US) - 1)
shutter_set_mode = False        # False = normal capture
_hold_fired = False             # guards short vs long press
shoot_event = threading.Event() # main loop performs capture

# --- Slow-shutter preview simulation ---
# When the effective exposure exceeds _PREVIEW_MAX_EXPOSURE_US the sensor is
# kept at a fast frame rate and the preview frame is digitally brightened so
# the live view remains smooth while the histogram and exposure appearance
# stay correct.
_PREVIEW_MAX_EXPOSURE_US = 33333           # 1/30s – keep preview >= 30 fps
_preview_brightness_gain = 1.0             # digital gain for preview brightness
_slow_shutter_capture_us = None            # real exposure µs for capture (None = use current)

def _apply_shutter_controls():
    """Apply the currently selected shutter to the *running* pipeline (auto-ISO).

    When a manual shutter speed is slower than 1/30s the sensor stays at
    1/30s for smooth preview; the brightness difference is compensated with
    a digital gain applied in the main loop.  For capture the real exposure
    time is restored momentarily.
    """
    global _preview_brightness_gain, _slow_shutter_capture_us
    us = SHUTTER_OPTIONS_US[current_shutter_idx]
    iso_gain = _current_iso_gain()
    ev = _current_exp_comp_value()
    controls = {}

    if us is None:
        # AUTO mode – cap frame duration so AE cannot drag the preview
        # frame rate below ~30 fps.  The AE compensates with ISO.
        controls["FrameDurationLimits"] = _PREVIEW_FRAME_LIMITS
        _preview_brightness_gain = 1.0
        _slow_shutter_capture_us = None
        if iso_gain is None:
            controls["AeEnable"] = True
        else:
            controls["AeEnable"] = True
            controls["AnalogueGain"] = float(iso_gain)
    else:
        if us > _PREVIEW_MAX_EXPOSURE_US:
            # Slow manual shutter: preview at 1/30s, simulate brightness.
            # Digital gain is only applied when ISO is manual (AE off),
            # because with auto ISO the AE already raises gain to match
            # the target exposure — adding digital gain on top would
            # double-compensate and over-expose the preview.
            _slow_shutter_capture_us = int(us)
            controls["ExposureTime"] = _PREVIEW_MAX_EXPOSURE_US
            controls["FrameDurationLimits"] = (_PREVIEW_MAX_EXPOSURE_US,
                                               _PREVIEW_MAX_EXPOSURE_US)
            if iso_gain is None:
                # Auto ISO: AE compensates via ISO, no digital gain needed
                _preview_brightness_gain = 1.0
            else:
                # Manual ISO: simulate the longer exposure digitally
                _preview_brightness_gain = float(us) / float(_PREVIEW_MAX_EXPOSURE_US)
        else:
            _preview_brightness_gain = 1.0
            _slow_shutter_capture_us = None
            controls["ExposureTime"] = int(us)
            controls["FrameDurationLimits"] = (int(us), int(us))

        if iso_gain is None:
            controls["AeEnable"] = True
        else:
            controls["AeEnable"] = False
            controls["AnalogueGain"] = float(iso_gain)

    # Apply exposure compensation (works with AeEnable=True)
    controls["ExposureValue"] = float(ev)

    picam2.set_controls(controls)

# Apply persisted shutter/ISO settings to the camera pipeline so manual
# exposure values saved from the previous session take effect immediately.
# Without this, the camera starts with AeEnable=True (full auto-exposure)
# regardless of what the UI displays, causing the sensor to auto-compensate
# for aperture changes on manual lenses.
_apply_shutter_controls()


def _enter_charge_mode():
    global charge_mode_active, _charge_shutdown_triggered
    if charge_mode_active:
        return

    # Exit video mode first so the flag is cleared and still_config is restored
    # cleanly before charge mode stops the pipeline.
    if _video_mode_active:
        _exit_video_mode()

    charge_mode_active = True
    _charge_shutdown_triggered = False
    _record_activity(wake=True)
    shoot_event.clear()
    try:
        picam2.stop()
    except Exception as exc:
        print("Preview stop error:", exc)
    _charge_stats_cache["timestamp"] = 0.0
    _update_charge_stats(force=True)


def _exit_charge_mode():
    global charge_mode_active, _charge_shutdown_triggered, _delete_confirm_active
    global _video_mode_active, _video_recording, _video_config
    _delete_confirm_active = False
    if not charge_mode_active:
        return
    charge_mode_active = False
    _charge_shutdown_triggered = False
    # Charge mode always restores still_config, so ensure the video mode flag
    # is cleared.  _enter_charge_mode() calls _exit_video_mode() on entry, but
    # if that failed or a race left the flag set the shutter button would
    # route to the record start/stop path on a still pipeline.
    with _video_lock:
        _video_mode_active = False
        _video_recording = False
        _video_config = None
    _charge_stats_cache["timestamp"] = 0.0
    _record_activity(wake=True)
    try:
        picam2.configure(_preview_running_config)
    except Exception:
        pass
    try:
        picam2.start()
        picam2.set_controls({"FrameDurationLimits": _PREVIEW_FRAME_LIMITS})
        time.sleep(0.1)
        _apply_shutter_controls()
    except Exception as exc:
        print("Preview restart error:", exc)

def _toggle_shutter_set_mode():
    global shutter_set_mode
    shutter_set_mode = not shutter_set_mode
    mode = "ON" if shutter_set_mode else "OFF"
    print(f"[Shutter Set Mode] {mode} – {SHUTTER_LABELS[current_shutter_idx]}")
    # Apply selection immediately to preview so you see the effect live
    _apply_shutter_controls()

def _trigger_shutter_osd():
    """Show the shutter speed OSD flash after a swipe gesture."""
    global _shutter_osd_text, _shutter_osd_fade_target, _shutter_osd_timer
    _shutter_osd_text = SHUTTER_LABELS[current_shutter_idx]
    _shutter_osd_fade_target = 1.0
    _shutter_osd_timer = _SHUTTER_OSD_HOLD_S

def _cycle_shutter():
    global current_shutter_idx
    current_shutter_idx = (current_shutter_idx + 1) % len(SHUTTER_OPTIONS_US)
    print(f"[Shutter] {SHUTTER_LABELS[current_shutter_idx]}")
    _debounced_save_settings()
    _apply_shutter_controls()
    _trigger_shutter_osd()


def _cycle_shutter_reverse():
    """Cycle shutter speed in reverse (toward slower / Auto)."""
    global current_shutter_idx
    current_shutter_idx = (current_shutter_idx - 1) % len(SHUTTER_OPTIONS_US)
    print(f"[Shutter] {SHUTTER_LABELS[current_shutter_idx]}")
    _debounced_save_settings()
    _apply_shutter_controls()
    _trigger_shutter_osd()

# Button wiring (hold=2s) — also initialises PLD sense button deferred earlier.
from gpiozero import Button
_init_pld_button()
button = Button(26, pull_up=True, bounce_time=0.05, hold_time=2.0)

def _toggle_minimal_mode():
    """Toggle minimal UI mode (hide all buttons except preview, histogram, battery, settings)."""
    global _ui_minimal_mode, _minimal_mode_timer, shutter_set_mode
    global _flip_button_rect, _wifi_button_rect, _charge_button_rect, _exit_button_rect
    global _double_exposure_button_rect, _rangefinder_button_rect, _shutter_toggle_rect
    global _aspect_toggle_rect, _film_toggle_rect, _iso_toggle_rect
    global _shutter_popup_visible, _aspect_popup_visible, _film_popup_visible, _iso_popup_visible
    global _focus_popup_visible, _focus_toggle_rect

    _ui_minimal_mode = not _ui_minimal_mode
    _minimal_mode_timer = None

    # Exit shutter set mode on 7-second hold (shutter set mode only for 2-7 second holds)
    if shutter_set_mode:
        shutter_set_mode = False
        print("[Shutter] Set mode OFF (7s hold)")

    if _ui_minimal_mode:
        # Clear all button rects so taps don't register on invisible buttons
        _flip_button_rect = (0, 0, 0, 0)
        _wifi_button_rect = (0, 0, 0, 0)
        _charge_button_rect = (0, 0, 0, 0)
        _exit_button_rect = (0, 0, 0, 0)
        _double_exposure_button_rect = (0, 0, 0, 0)
        _rangefinder_button_rect = (0, 0, 0, 0)
        _focus_toggle_rect = (0, 0, 0, 0)
        _shutter_toggle_rect = (0, 0, 0, 0)
        _aspect_toggle_rect = (0, 0, 0, 0)
        _film_toggle_rect = (0, 0, 0, 0)
        _iso_toggle_rect = (0, 0, 0, 0)
        # Close any open popups
        _shutter_popup_visible = False
        _aspect_popup_visible = False
        _film_popup_visible = False
        _iso_popup_visible = False
        _focus_popup_visible = False

    print(f"[UI] Minimal mode {'enabled' if _ui_minimal_mode else 'disabled'}")
    _debounced_save_settings()

def _on_held():
    global _hold_fired, _minimal_mode_timer
    if sleep_mode_active:
        _sleep_wake_event.set()
        return
    if charge_mode_active:
        return
    # In video mode, long-press exits video mode entirely.
    # Read flag under lock, but call _exit_video_mode() outside to avoid
    # deadlock (_exit_video_mode also acquires _video_lock).
    with _video_lock:
        _in_video = _video_mode_active
    if _in_video:
        _hold_fired = True
        _exit_video_mode()
        return
    _hold_fired = True
    _toggle_shutter_set_mode()
    # Start timer for 7-second total hold to toggle minimal UI mode
    if _minimal_mode_timer is not None:
        _minimal_mode_timer.cancel()
    _minimal_mode_timer = threading.Timer(_MINIMAL_MODE_HOLD_EXTRA_S, _toggle_minimal_mode)
    _minimal_mode_timer.start()

def _on_released():
    global _hold_fired, zoom_active, _zoom_level_idx, _minimal_mode_timer
    # Cancel minimal mode timer if button released before 7 seconds
    if _minimal_mode_timer is not None:
        _minimal_mode_timer.cancel()
        _minimal_mode_timer = None
    if sleep_mode_active:
        _sleep_wake_event.set()
        return
    if charge_mode_active:
        return
    if _hold_fired:
        _hold_fired = False
        return
    # Short press — in video mode, toggle recording on/off
    with _video_lock:
        in_video = _video_mode_active
        is_recording = _video_recording
    if in_video:
        _record_activity(wake=True)
        if is_recording:
            _stop_video_recording()
        else:
            _start_video_recording()
        return
    # Short press — normal still mode
    if shutter_set_mode:
        _cycle_shutter()
    else:
        # Block shutter during Brenizer completion screen
        with _brenizer_lock:
            if _brenizer_active and _brenizer_state == "complete":
                return
        zoom_active = False
        _zoom_level_idx = 0
        _record_activity(wake=True)
        _start_capture_sequence()
        shoot_event.set()

button.when_held = _on_held
button.when_released = _on_released

# -------------------- Display window --------------------
# Window already created at top of file for early splash screen;
# just wire up the touch callback now that _on_touch is defined.
cv2.setMouseCallback("Camera", _on_touch)

# ---- Graceful shutdown: flush settings + stop camera on SIGTERM/SIGINT ----
_shutdown_flag = threading.Event()


def _graceful_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT: flush pending settings and signal main loop to exit."""
    print(f"[Shutdown] Signal {signum} received, shutting down gracefully...")
    _shutdown_flag.set()


signal.signal(signal.SIGTERM, _graceful_shutdown)
signal.signal(signal.SIGINT, _graceful_shutdown)


def _flush_pending_settings():
    """Atexit handler: cancel debounce timer and write settings immediately."""
    with _settings_save_timer_lock:
        if _settings_save_timer is not None:
            _settings_save_timer.cancel()
    _save_persistent_settings()


atexit.register(_flush_pending_settings)

# -------------------- Main loop --------------------
canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
_black_canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
# Pre-allocated buffer for flip operations (avoids per-frame allocation)
_flip_buffer = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
# Capture confirmation: black-out "shutter flash" (Sony-style)
# Frame-based counter is more reliable than time-based when capture blocks the loop.
_BLACK_FLASH_FRAMES = 3  # ~250ms at 24fps (more noticeable)
_black_flash_frames = 0

_last_meta_time = 0.0
_meta = {}
_META_REFRESH_S = 0.20  # refresh exposure metadata ~5x/sec; UI doesn't need per-frame

# --- Frame pacing + smoothing (production UX) ---
TARGET_FPS = 20
TARGET_FPS_IDLE = 10  # Lower FPS when backlight is dimmed to save power
_FRAME_DT = 1.0 / TARGET_FPS
_FRAME_DT_IDLE = 1.0 / TARGET_FPS_IDLE
_next_frame_t = time.monotonic()

EMA_ALPHA = 0.20  # 0.10-0.30 feels good; lower = smoother/slower response
_ema_shutter_us = None
_ema_iso = None

def _ema(prev, new, alpha=EMA_ALPHA):
    if new is None:
        return prev
    if prev is None:
        return float(new)
    return float(prev) * (1.0 - alpha) + float(new) * alpha

def _pace_frame():
    """Cap UI refresh using monotonic scheduling. Drops to TARGET_FPS_IDLE when
    the backlight is dimmed (idle for 2+ min) to reduce CPU load and save battery."""
    global _next_frame_t
    dt = _FRAME_DT_IDLE if _backlight_dimmed else _FRAME_DT
    _next_frame_t += dt
    now = time.monotonic()
    sleep_t = _next_frame_t - now
    if sleep_t > 0:
        time.sleep(sleep_t)
    else:
        # We're behind; reset so we don't spiral.
        _next_frame_t = now
try:
    while not _shutdown_flag.is_set():
        # Update popup fade animations
        _update_popup_fades(_FRAME_DT)

        # --- Low power sleep mode ---
        if sleep_mode_active:
            # Wait for shutter press to wake
            if _sleep_wake_event.is_set():
                _sleep_wake_event.clear()
                _exit_sleep_mode()
                _record_activity(wake=True)
            else:
                # While asleep, re-assert fan/backlight off occasionally (driver may override when hot).
                try:
                    now = time.monotonic()
                    if now - _sleep_state.get('last_fan_assert', 0) > 2.0:
                        _set_fan_state(0)
                        _set_backlight_brightness(0)
                        _sleep_state['last_fan_assert'] = now
                except Exception:
                    pass
                time.sleep(0.10)
            continue
        _update_backlight()
        _sleep_button_rect = (0, 0, 0, 0)
        if _update_prompt_ready.is_set():
            with _pending_update_lock:
                update_path = _pending_update_path
                _pending_update_path = None
                _update_prompt_ready.clear()

            if update_path:
                # Exit video mode cleanly before the update prompt takes over
                if _video_mode_active:
                    _exit_video_mode()

                try:
                    picam2.stop()
                except Exception:
                    pass

                _prompt_for_update(update_path)

                try:
                    picam2.configure(_preview_running_config)
                    picam2.start()
                    picam2.set_controls({"FrameDurationLimits": _PREVIEW_FRAME_LIMITS})
                    _apply_shutter_controls()
                except Exception as exc:
                    print("Preview restart error after update prompt:", exc)
                _record_activity(wake=True)
                continue

        if charge_mode_active:
            lines, pct = _update_charge_stats()
            _maybe_trigger_charge_shutdown(pct)
            canvas[:] = 0
            block = make_rotated_text_block(
                lines,
                font_scale=1.1,
                thickness=1,
                rotate_180=True,
                max_h=SCREEN_H - BAR_H,
            )
            bh, bw = block.shape[:2]
            cx = max(0, (SCREEN_W - bw) // 2)
            cy = max(0, (SCREEN_H - bh) // 2)
            blit_add(canvas, block, cx, cy)
            # Controls in Charge mode (Charge toggles this screen; Sleep enters low-power mode)
            button_size = 72
            control_gap = 18
            column_right = SCREEN_W - 18
            column_top = 16

            charge_rect = draw_charge_button(canvas, True, column_top, column_right, size=button_size, margin=16)
            _charge_button_rect = charge_rect if charge_rect else (0, 0, 0, 0)
            if charge_rect:
                column_top = charge_rect[3] + control_gap

            sleep_rect = draw_sleep_button(canvas, False, column_top, column_right, size=button_size, margin=16)
            _sleep_button_rect = sleep_rect if sleep_rect else (0, 0, 0, 0)
            if sleep_rect:
                column_top = sleep_rect[3] + control_gap

            trash_rect = draw_trash_button(canvas, column_top, column_right, size=button_size, margin=16)
            _trash_button_rect = trash_rect if trash_rect else (0, 0, 0, 0)
            if trash_rect:
                column_top = trash_rect[3] + control_gap

            exit_rect = draw_x_exit_button(canvas, column_top, column_right, size=button_size, margin=16)
            _exit_button_rect = exit_rect if exit_rect else (0, 0, 0, 0)

            # Draw delete confirmation overlay on top of everything if active
            if _delete_confirm_active:
                _draw_delete_confirm_overlay(canvas)

            # Optimized: use pre-allocated buffer for flip operations
            if flip_mode == FLIP_MODE_MIRROR:
                cv2.flip(canvas, 1, dst=_flip_buffer)
                display_canvas = _flip_buffer
            elif flip_mode == FLIP_MODE_MIRROR_ROTATE:
                cv2.flip(canvas, 0, dst=_flip_buffer)
                display_canvas = _flip_buffer
            else:
                display_canvas = canvas
            cv2.imshow("Camera", display_canvas)
            if cv2.waitKey(1) == 27:
                break
            _pace_frame()
            continue
        # --- Video mode: auto-stop on duration limit, low battery, or low disk ---
        with _video_lock:
            _vmode = _video_mode_active
            _vrec = _video_recording
        if _vmode and _vrec:
            if _video_elapsed_s() >= _VIDEO_MAX_DURATION_S:
                print("[Video] Maximum clip duration reached — auto-stopping")
                _stop_video_recording()
            elif not _disk_can_capture():
                print("[Video] Disk space critical — auto-stopping recording")
                _stop_video_recording()
            elif not _batt_can_record():
                with _batt_lock:
                    _vbatt = batt_percent_cached
                print(f"[Video] Battery critically low ({_vbatt}%) — auto-stopping recording")
                _stop_video_recording()

        # Handle capture requests first (single-threaded Picamera2 usage)
        # Skip still capture when in video mode (recording controlled by shutter handler)
        if _vmode and shoot_event.is_set():
            shoot_event.clear()
        if shoot_event.is_set() and not shutter_set_mode:
            shoot_event.clear()

            # --- Disk space guard ---
            if not _disk_can_capture():
                print("[Capture] Blocked: disk space critical — clear internal memory to continue")
                continue

            # Determine if this is a Brenizer capture
            with _brenizer_lock:
                _bren_capturing = _brenizer_active and _brenizer_state == "awaiting_capture"
                _bren_tile_num = _brenizer_tile_idx + 1 if _bren_capturing else 0

            if _bren_capturing:
                # Brenizer mode: override paths to sequence directory
                _ensure_dir(_brenizer_seq_dir)
                bren_base = f"brenizer_{_brenizer_seq_id}_{_bren_tile_num:02d}"
                jpg_path = os.path.join(_brenizer_seq_dir, f"{bren_base}.jpg")
                dng_path = os.path.join(_brenizer_seq_dir, f"{bren_base}.dng")
            else:
                with _capture_dir_lock:
                    target_dir = _capture_dir
                _ensure_dir(target_dir)
                base_name = _next_capture_basename()
                jpg_path = os.path.join(target_dir, f"{base_name}.jpg")
                dng_path = os.path.join(target_dir, f"{base_name}.dng")

            with _aspect_button_lock:
                capture_aspect_idx = _aspect_idx
            target_ratio = ASPECT_OPTIONS[capture_aspect_idx]["ratio"]
            target_dims = _aspect_capture_dims[capture_aspect_idx]
            with _film_button_lock:
                capture_film_idx = _film_idx
                capture_film_key = FILM_OPTIONS[capture_film_idx]["key"]
            full_frame = None
            capture_error = None
            capture_request = None
            # Lift the preview frame-rate cap so the capture can use the full
            # exposure range.  Manual slow shutter: exact target exposure.
            # AUTO mode: capture immediately with current AE state (preview
            # already converged at up to 1/30s, ISO compensates in low light).
            # For Brenizer mode: after first shot exposure is locked, so
            # _slow_shutter_capture_us should be None (AE disabled).
            _cap_slow_us = _slow_shutter_capture_us
            # On large sensors (IMX492) the live view runs in a lightweight
            # preview config with no raw stream and a small main stream, so
            # we briefly switch to still_config to get full-resolution RGB +
            # raw for the capture, then switch back.  Slow-shutter controls
            # must be (re)applied *after* the mode switch so they take effect
            # on the still config's sensor mode.
            _switched_for_capture = False
            if USE_LIGHTWEIGHT_PREVIEW:
                try:
                    picam2.switch_mode(still_config)
                    _switched_for_capture = True
                except Exception as mode_err:
                    print(f"[Capture] switch_mode(still) failed: {mode_err}")
            if _cap_slow_us is not None and not _bren_capturing:
                picam2.set_controls({
                    "ExposureTime": int(_cap_slow_us),
                    "FrameDurationLimits": (int(_cap_slow_us), int(_cap_slow_us)),
                })
                # Wait for sensor to apply the new exposure (≥2 frame durations)
                time.sleep(_cap_slow_us / 1e6 * 2 + 0.1)
            _pending_dng_path = None
            try:
                # Capture RAW DNG from the raw stream. If this fails, still save the JPG.
                try:
                    picam2.capture_file(dng_path, name="raw")
                    # Flash after raw capture - signals user can move camera
                    _black_flash_frames = _BLACK_FLASH_FRAMES
                    cv2.imshow("Camera", _black_canvas)
                    cv2.waitKey(1)
                    # DNG rotation + USB sync moved to save worker to avoid
                    # blocking the main display loop with disk I/O.
                    _pending_dng_path = dng_path
                except Exception as dng_err:
                    print("DNG capture error:", dng_err)
                capture_request = picam2.capture_request()
                full_frame = capture_request.make_array("main")
                capture_request.release()
            except Exception as e:
                capture_error = e
                if capture_request is not None:
                    try:
                        capture_request.release()
                    except Exception:
                        pass
            finally:
                # Always restore the lightweight preview config so the live
                # view comes back, even if the capture itself raised.
                if _switched_for_capture:
                    try:
                        picam2.switch_mode(_preview_running_config)
                        picam2.set_controls({"FrameDurationLimits": _PREVIEW_FRAME_LIMITS})
                    except Exception as mode_err:
                        print(f"[Capture] switch_mode(preview) failed: {mode_err}")

            if capture_error is not None:
                print("Capture error:", capture_error)
            elif full_frame is not None:
                if _bren_capturing:
                    # Brenizer mode: save frame and advance tile
                    _capture_save_queue.put(
                        (
                            full_frame,
                            target_dims,
                            target_ratio,
                            capture_film_key,
                            jpg_path,
                            None,  # no overlay frame
                            _pending_dng_path,
                        )
                    )
                    # Store cell-sized thumbnail for minimap display
                    _cell_w, _cell_h = 56, 36  # must match minimap cell size
                    try:
                        _tile_thumb = cv2.resize(full_frame, (_cell_w, _cell_h),
                                                 interpolation=cv2.INTER_AREA)
                        if _tile_thumb.ndim == 3 and _tile_thumb.shape[2] == 4:
                            _tile_thumb = cv2.cvtColor(_tile_thumb, cv2.COLOR_BGRA2BGR)
                    except Exception:
                        _tile_thumb = None
                    _brenizer_tile_thumbs.append(_tile_thumb)
                    _brenizer_on_capture(jpg_path)
                else:
                    # Normal / Double Exposure capture flow
                    overlay_frame = None
                    skip_save = False
                    with _double_exposure_lock:
                        if double_exposure_enabled:
                            if double_exposure_first_frame is None:
                                double_exposure_first_frame = full_frame
                                _double_preview_cache = {"key": None, "image": None}
                                skip_save = True
                            else:
                                overlay_frame = double_exposure_first_frame
                                double_exposure_first_frame = None
                                _double_preview_cache = {"key": None, "image": None}
                        else:
                            double_exposure_first_frame = None
                            _double_preview_cache = {"key": None, "image": None}
                    if skip_save:
                        # Restore preview frame rate even when skipping save
                        if _cap_slow_us is not None:
                            _apply_shutter_controls()
                        _record_activity(wake=True)
                        continue
                    # Backpressure guard: warn if save queue is backing up
                    _save_depth = _capture_save_queue.qsize()
                    if _save_depth >= 5:
                        print(f"[Capture] Save queue depth: {_save_depth} — saves may be slow")
                    _capture_save_queue.put(
                        (
                            full_frame,
                            target_dims,
                            target_ratio,
                            capture_film_key,
                            jpg_path,
                            overlay_frame,
                            _pending_dng_path,
                        )
                    )

            # Restore preview frame-rate cap after capture
            if _cap_slow_us is not None and not _bren_capturing:
                _apply_shutter_controls()

        # Preview frame + overlays.  The frame comes from either the
        # small-sensor still_config's lores stream (RGB888) or the
        # large-sensor lightweight config's main stream (YUV420).
        try:
            frame = picam2.capture_array(PREVIEW_STREAM_NAME)
        except Exception as e:
            print("Preview capture error:", e)
            time.sleep(0.05)
            continue
        # Metadata capture can be expensive; refresh at a low rate for UI.
        now = time.time()
        if now - _last_meta_time >= _META_REFRESH_S:
            try:
                _meta = picam2.capture_metadata()
            except Exception:
                _meta = {}
            _last_meta_time = now
        meta = _meta

        if frame.ndim == 2:
            # YUV420 planar from the lightweight preview config — one
            # SIMD-accelerated conversion per frame gives us the RGB
            # array the rest of the pipeline expects, and we still come
            # out ahead of asking the ISP for RGB888 directly.
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # --- Slow-shutter preview simulation: digitally brighten the preview
        #     frame to match the target exposure so the histogram and live view
        #     show the correct brightness even though the sensor is running at
        #     a faster frame rate.  cv2.convertScaleAbs clips to [0,255].
        _gain = _preview_brightness_gain
        if _gain > 1.01:
            cv2.convertScaleAbs(frame, alpha=_gain, beta=0, dst=frame)

        with _film_button_lock:
            film_entry = FILM_OPTIONS[_film_idx]
            film_key = film_entry["key"]
            film_label = film_entry["label"]

        zoom_factor = 0.0
        if zoom_active and ZOOM_LEVELS:
            zoom_factor = ZOOM_LEVELS[_zoom_level_idx]
            view_frame = _apply_zoom(frame, zoom_factor, zoom_center_norm)
        else:
            view_frame = frame

        fh, fw = view_frame.shape[:2]
        aspect_ratio = _aspect_ratio_current  # Use animated ratio for smooth transitions
        if aspect_ratio > 0:
            crop_w, crop_h = _largest_ratio_crop_dims(fw, fh, aspect_ratio)
            if (crop_w, crop_h) != (fw, fh):
                view_frame = _center_crop(view_frame, crop_w, crop_h)
                fh, fw = view_frame.shape[:2]
        geom_key = (fh, fw)
        if geom_key not in _geom_cache:
            if len(_geom_cache) > 16:
                _geom_cache.clear()
            _geom_cache[geom_key] = _compute_display_geometry(fw, fh)
        new_w, new_h, x_off, y_off, bar_y = _geom_cache[geom_key]
        scaled = cv2.resize(view_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        assist_frame = _prepare_focus_assist_frame(view_frame)

        _preview_geom["x"], _preview_geom["y"], _preview_geom["w"], _preview_geom["h"] = x_off, y_off, new_w, new_h

        if rangefinder_assist_enabled:
            _update_rangefinder_shift(assist_frame if assist_frame is not None else scaled)

        # Apply film simulation to display-sized frame instead of full-resolution
        # frame to avoid buffer starvation freezes (12.3M → ~150K pixels).
        if film_key != "none":
            scaled = apply_film_simulation_rgb(scaled, film_key)

        if focus_peaking_enabled:
            disp, gray_for_hist = apply_focus_peaking(scaled, assist_frame=assist_frame)
        else:
            disp = scaled
            gray_for_hist = _mono_to_gray(scaled)

        double_overlay = _get_double_preview_overlay(
            new_w,
            new_h,
            aspect_ratio,
            film_key,
            zoom_factor if zoom_active else 1.0,
            zoom_center_norm if zoom_active else None,
        )
        if double_overlay is not None:
            cv2.addWeighted(double_overlay, _DOUBLE_PREVIEW_ALPHA, disp, 1.0 - _DOUBLE_PREVIEW_ALPHA, 0, dst=disp)

        if rangefinder_assist_enabled:
            _draw_rangefinder_overlay(disp)

        # Draw grid overlay on preview only (not included in captured image)
        _draw_grid_overlay(disp, _current_grid_key())

        # Optimized: clear only margins instead of entire canvas
        _clear_canvas_margins(canvas, x_off, y_off, new_w, new_h)
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = disp
        hist_w = min(256, new_w // 3)
        # Throttle histogram to every 2nd frame — imperceptible at 20fps
        _hist_frame_toggle = not _hist_frame_toggle
        if _hist_frame_toggle or _hist_cached_img is None:
            _hist_cached_img = draw_histogram(gray_for_hist, height=BAR_H, width=hist_w, rotated=True)
        left_margin = 18
        hist_x = left_margin
        blit_add(canvas, _hist_cached_img, hist_x, bar_y)

        # Battery icon in bottom row, left-aligned after histogram
        # Only re-render when battery % or charge state changes
        with _batt_lock:
            pct = batt_percent_cached
        ac_state = _read_pld_state()
        _batt_key = (pct, ac_state)
        if _batt_render_key != _batt_key:
            batt_iw, batt_ih = 30, 13
            bolt_size = 14 if ac_state == 1 else 0
            batt_buf = _batt_buf_with_bolt if bolt_size else _batt_buf_no_bolt
            batt_buf[:] = 0
            batt_buf_h = batt_buf.shape[0]
            batt_x_off = bolt_size + 2 if bolt_size else 0
            batt_y_off = (batt_buf_h - batt_ih) // 2
            draw_battery(batt_buf, pct, bar_y=batt_y_off, right_margin=0, left_margin=batt_x_off, align_top=True)
            if bolt_size:
                _bolt_canvas[:] = 0
                _draw_charge_icon(_bolt_canvas, True)
                bolt_y_in_buf = (batt_buf_h - bolt_size) // 2
                blit_add(batt_buf, _bolt_canvas, 0, bolt_y_in_buf)
            _batt_render_cache = cv2.rotate(batt_buf, cv2.ROTATE_180)
            _batt_render_key = _batt_key
        batt_x = SCREEN_W - 18 - _batt_render_cache.shape[1]
        fixed_bar_y = SCREEN_H - BAR_H
        batt_y = fixed_bar_y + (BAR_H - _batt_render_cache.shape[0]) // 2
        blit_add(canvas, _batt_render_cache, batt_x, batt_y)

        # --- Disk space warning indicator ---
        with _disk_lock:
            _dl = _disk_level
            _dfm = _disk_free_mb
        if _dl in ("warning", "critical"):
            if _dl == "critical":
                _disk_text = "NO SPACE — Clear internal memory to continue"
                _disk_color = (0, 0, 255)  # red (BGR)
            else:
                _disk_text = "LOW DISK — Enter charging mode to clear memory"
                _disk_color = (0, 180, 255)  # orange (BGR)
            if _dfm is not None:
                _disk_text += f"  ({_dfm / 1024:.1f} GB free)"
            _disk_block = make_rotated_text_block(
                [[{"text": _disk_text, "color": _disk_color}]],
                font_scale=1.0, thickness=1, rotate_180=True, max_h=BAR_H,
            )
            if _disk_block is not None:
                _dbx = max(0, batt_x - _disk_block.shape[1] - 10)
                _dby = fixed_bar_y + (BAR_H - _disk_block.shape[0]) // 2
                blit_add_with_shadow(canvas, _disk_block, _dbx, _dby)

        # --- Low battery warning indicator ---
        # Reuse `pct` from the battery icon section above (same frame, same lock)
        if pct is not None and pct < _BATT_WARN_PCT:
            if pct < _BATT_CRITICAL_PCT:
                _batt_text = f"BATTERY CRITICAL ({pct:.0f}%) — Charge battery to record"
            else:
                _batt_text = f"LOW BATTERY ({pct:.0f}%)"
            _batt_block = make_rotated_text_block(
                [[{"text": _batt_text}]],
                font_scale=0.65, thickness=1, rotate_180=True, max_h=BAR_H,
            )
            if _batt_block is not None:
                _bbx = max(0, batt_x - _batt_block.shape[1] - 10)
                _bby = fixed_bar_y + (BAR_H - _batt_block.shape[0]) // 2
                # Stack below disk warning if both are visible
                if _dl in ("warning", "critical"):
                    _bby = max(0, fixed_bar_y - _batt_block.shape[0] - 2)
                blit_add_with_shadow(canvas, _batt_block, _bbx, _bby)

        hist_left = hist_x
        right_bound = hist_left - 20

        # Live metadata + status
        raw_shutter_us = meta.get("ExposureTime", 0)
        raw_iso = meta.get("AnalogueGain", 0) * 100

        # EMA smoothing for a more natural "camera-like" readout (does not affect exposure)
        _ema_shutter_us = _ema(_ema_shutter_us, raw_shutter_us)
        _ema_iso = _ema(_ema_iso, raw_iso)

        shutter_us = int(_ema_shutter_us) if _ema_shutter_us else raw_shutter_us
        iso = float(_ema_iso) if _ema_iso else raw_iso
        target_label = SHUTTER_LABELS[current_shutter_idx]
        with _aspect_button_lock:
            aspect_label = ASPECT_OPTIONS[_aspect_idx]["label"]
        iso_label = _current_iso_label()

        shutter_label = target_label
        if not shutter_set_mode and SHUTTER_OPTIONS_US[current_shutter_idx] is None and shutter_us:
            shutter_label = format_shutter_speed(shutter_us)

        iso_display = iso_label
        if not shutter_set_mode and _current_iso_gain() is None and iso:
            iso_display = str(int(round(iso)))

        right_control_rects = []
        left_control_rects = []
        control_gap = 12
        column_top = 18
        left_column_top = 18
        button_size = 72
        column_right = SCREEN_W - 18
        left_column_right = LEFT_COLUMN_MARGIN + button_size

        # Draw UI buttons with fade (smooth hide/show for minimal mode)
        if _icons_fade > 0.01 and not _brenizer_active:
            flip_rect = draw_flip_button(
                canvas,
                flip_mode != FLIP_MODE_NORMAL,
                left_column_top,
                left_column_right,
                size=button_size,
                margin=LEFT_COLUMN_MARGIN,
            )
            _flip_button_rect = flip_rect if flip_rect else (0, 0, 0, 0)
            if flip_rect:
                left_control_rects.append(flip_rect)
                left_column_top = flip_rect[3] + control_gap
            else:
                left_column_top += button_size + control_gap

            shutter_rect = _draw_shutter_controls(
                canvas,
                left_column_top,
                left_column_right,
                size=button_size,
                margin=LEFT_COLUMN_MARGIN,
            )
            if shutter_rect:
                left_control_rects.append(shutter_rect)
                left_column_top = shutter_rect[3] + control_gap
            else:
                left_column_top += button_size + control_gap

            # Multishot button (Double Exposure / Brenizer) — replaces standalone DE button
            double_rect = _draw_multishot_controls(
                canvas,
                left_column_top,
                left_column_right,
                size=button_size,
                margin=LEFT_COLUMN_MARGIN,
            )
            _double_exposure_button_rect = double_rect if double_rect else (0, 0, 0, 0)
            if double_rect:
                left_control_rects.append(double_rect)
                left_column_top = double_rect[3] + control_gap
            else:
                left_column_top += button_size + control_gap

            focus_rect = _draw_focus_controls(
                canvas,
                left_column_top,
                left_column_right,
                size=button_size,
                margin=LEFT_COLUMN_MARGIN,
            )
            _rangefinder_button_rect = focus_rect if focus_rect else (0, 0, 0, 0)
            if focus_rect:
                left_control_rects.append(focus_rect)
                left_column_top = focus_rect[3] + control_gap
            else:
                left_column_top += button_size + control_gap

            aspect_rect = _draw_aspect_controls(
                canvas,
                column_top,
                column_right,
                size=button_size,
                margin=16,
            )
            if aspect_rect:
                right_control_rects.append(aspect_rect)
                column_top = aspect_rect[3] + control_gap
            else:
                column_top += button_size + control_gap

            film_rect = _draw_film_controls(
                canvas,
                column_top,
                column_right,
                size=button_size,
                margin=16,
            )
            if film_rect:
                right_control_rects.append(film_rect)
                column_top = film_rect[3] + control_gap
            else:
                column_top += button_size + control_gap

            wifi_rect = draw_wifi_button(
                canvas,
                wifi_enabled,
                column_top,
                column_right,
                size=button_size,
                margin=16,
            )
            if wifi_rect:
                _wifi_button_rect = wifi_rect
                right_control_rects.append(wifi_rect)
                column_top = wifi_rect[3] + control_gap
            else:
                _wifi_button_rect = (0, 0, 0, 0)
                column_top += button_size + control_gap

            charge_rect = draw_charge_button(
                canvas,
                charge_mode_active,
                column_top,
                column_right,
                size=button_size,
                margin=16,
            )
            _charge_button_rect = charge_rect if charge_rect else (0, 0, 0, 0)
            if charge_rect:
                right_control_rects.append(charge_rect)
                column_top = charge_rect[3] + control_gap
            else:
                column_top += button_size + control_gap

            iso_rect = _draw_iso_controls(
                canvas,
                column_top,
                column_right,
                size=button_size,
                margin=16,
            )
            if iso_rect:
                right_control_rects.append(iso_rect)
                column_top = iso_rect[3] + control_gap
            else:
                column_top += button_size + control_gap

            # Draw Exp Comp popup overlay (if visible or fading)
            _draw_exp_comp_popup(canvas)

        if right_control_rects:
            control_left = min(rect[0] for rect in right_control_rects)
            right_bound = min(right_bound, control_left - 18)
        right_bound = min(SCREEN_W - 18, right_bound)

        # Skip SS/ISO and zoom text when Brenizer or Video mode provides its own status overlay
        if not _brenizer_active and not _video_mode_active:
            _set_suffix = "  SET   " if shutter_set_mode else "   "
            shutter_line = make_rotated_text_block(
                [[{"text": "SS ", "font_scale": 1.2, "color": UI_TEXT_PRIMARY}, {"text": f"{shutter_label}{_set_suffix}", "color": UI_TEXT_PRIMARY}, {"text": "ISO ", "font_scale": 1.0, "color": UI_TEXT_PRIMARY}, {"text": iso_display, "font_scale": 1.4, "color": UI_TEXT_PRIMARY}]],
                font_scale=1.8,
                thickness=2,
                rotate_180=True,
                max_h=BAR_H + 8,
                pad_x=4,
                pad_y=2,
                line_gap=2,
            )
            with _double_exposure_lock:
                double_on = double_exposure_enabled
                double_waiting = double_exposure_first_frame is not None
            double_suffix = ""
            if double_on:
                double_suffix = "   DE: 1/2" if double_waiting else "   DE: ON"
            ev_label = _current_exp_comp_label()
            ev_suffix = f"   EV: {ev_label}" if ev_label != "0" else ""
            zoom_line = make_rotated_text_block(
                [[{"text": f"{zoom_factor:.1f}x", "color": UI_TEXT_PRIMARY}, {"text": "   IC: ", "color": UI_TEXT_PRIMARY}, {"text": f"{image_count}", "color": UI_TEXT_PRIMARY}, {"text": f"{ev_suffix}{double_suffix}", "color": UI_TEXT_PRIMARY}]],
                font_scale=1.0,
                thickness=1,
                rotate_180=True,
                max_h=BAR_H - 10,
                pad_x=4,
                pad_y=2,
                line_gap=2,
            )

            text_left = 18
            if left_control_rects:
                left_bound = max(rect[2] for rect in left_control_rects)
                text_left = max(text_left, left_bound + 18)
            # If zoom text would straddle the preview edge, push it inside
            if text_left < x_off and text_left + zoom_line.shape[1] > x_off:
                text_left = x_off + 8
            text_top = max(12, y_off + 12)

            # Center: SS/ISO aligned to screen center (with drop shadow)
            center_x = SCREEN_W // 2 - shutter_line.shape[1] // 2
            blit_add_with_shadow(canvas, shutter_line, center_x, text_top)

            # Left: zoom, image count, DE status
            # If zoom text would overlap centered SS/ISO, stack below instead
            min_gap = 12
            if text_left + zoom_line.shape[1] + min_gap > center_x:
                zoom_y = text_top + shutter_line.shape[0] + 4
            else:
                zoom_y = text_top + (shutter_line.shape[0] - zoom_line.shape[0]) // 2
            blit_add_with_shadow(canvas, zoom_line, text_left, zoom_y)

        # --- Brenizer Mode overlay rendering ---
        with _brenizer_lock:
            _bren_overlay_active = _brenizer_active
            _bren_overlay_state = _brenizer_state
            _bren_overlay_help = _brenizer_show_help

        if _bren_overlay_active:
            if _bren_overlay_help:
                # Full-screen help/instruction overlay
                _render_brenizer_help_overlay(canvas)
            elif _bren_overlay_state == "complete":
                # Completion screen with buttons
                _render_brenizer_complete_screen(canvas)
            elif _bren_overlay_state == "awaiting_capture":
                # Live capture guidance: mini-map + status + direction prompt
                # Mini-map in bottom-right area, above histogram bar
                # Pass the live preview frame so the active tile shows the camera feed
                minimap_x = SCREEN_W - 180
                minimap_y = bar_y - 125
                _render_brenizer_minimap(canvas, minimap_x, minimap_y,
                                        live_frame=view_frame)

                # Status text replaces normal SS/ISO at top
                status_bottom = _render_brenizer_status(canvas, x_off, y_off, new_w, new_h)
                # Direction prompt below status
                _render_brenizer_direction_prompt(canvas, status_bottom)

        # --- Video Mode overlay rendering ---
        with _video_lock:
            _vid_overlay_active = _video_mode_active
        if _vid_overlay_active:
            _draw_video_overlay(canvas)

        with _wifi_lock:
            wifi_on = wifi_enabled
        show_wifi_info = wifi_on and (time.monotonic() < _wifi_info_until)
        if show_wifi_info:
            ip_addr = _get_ip_address()
            wifi_lines = [
                ["Wi-Fi Enabled"],
                [f"IP: {ip_addr}"],
                ["Port: 22"],
                ["Username: pi"],
                ["Password: monochrome"],
            ]
            wifi_block = make_rotated_text_block(
                wifi_lines,
                font_scale=1.1,
                thickness=2,
                rotate_180=True,
                max_h=min(SCREEN_H // 2, SCREEN_W // 2),
                pad_x=12,
                pad_y=8,
                line_gap=8,
            )
            wb_h, wb_w = wifi_block.shape[:2]
            wx = max(0, (SCREEN_W - wb_w) // 2)
            wy = max(0, (SCREEN_H - wb_h) // 2)
            bg_x1 = max(0, wx - 18)
            bg_y1 = max(0, wy - 18)
            bg_x2 = min(SCREEN_W, wx + wb_w + 18)
            bg_y2 = min(SCREEN_H, wy + wb_h + 18)
            if bg_x1 < bg_x2 and bg_y1 < bg_y2:
                wifi_bg_roi = canvas[bg_y1:bg_y2, bg_x1:bg_x2]
                cv2.convertScaleAbs(wifi_bg_roi, alpha=0.7, beta=0, dst=wifi_bg_roi)
            blit_add_with_shadow(canvas, wifi_block, wx, wy)

        # --- Shutter speed swipe OSD (centered fade-in / fade-out text) ---
        if _shutter_osd_fade > 0.01 and _shutter_osd_text:
            # Build text + pre-blurred shadow once per text change
            if _osd_composite_cache_key != _shutter_osd_text:
                _osd_font = max(28, int(SCREEN_H * 0.09))
                _osd_tw, _osd_th = _ui_measure_text(_shutter_osd_text, _osd_font)
                _osd_buf = np.zeros((_osd_th, _osd_tw, 3), dtype=np.uint8)
                _ui_draw_text(_osd_buf, _shutter_osd_text, 0, 0, _osd_font,
                              UI_ACCENT_CREATIVE, outline_bgr=(0, 0, 0))
                _osd_buf = cv2.rotate(_osd_buf, cv2.ROTATE_180)
                _osd_composite_text = _osd_buf
                # Pre-compute shadow (GaussianBlur) at full opacity
                _osd_composite_shadow = cv2.convertScaleAbs(
                    cv2.GaussianBlur(_osd_buf, (5, 5), 0), alpha=0.5, beta=0)
                _osd_composite_cache_key = _shutter_osd_text
            _osd_th, _osd_tw = _osd_composite_text.shape[:2]
            _osd_x = (SCREEN_W - _osd_tw) // 2
            _osd_y = (SCREEN_H - _osd_th) // 2
            _osd_x1 = max(0, _osd_x)
            _osd_y1 = max(0, _osd_y)
            _osd_x2 = min(SCREEN_W, _osd_x + _osd_tw)
            _osd_y2 = min(SCREEN_H, _osd_y + _osd_th)
            if _osd_x1 < _osd_x2 and _osd_y1 < _osd_y2:
                _osd_sx1 = _osd_x1 - _osd_x
                _osd_sy1 = _osd_y1 - _osd_y
                _osd_crop = (slice(_osd_sy1, _osd_sy1 + (_osd_y2 - _osd_y1)),
                             slice(_osd_sx1, _osd_sx1 + (_osd_x2 - _osd_x1)))
                _osd_roi = canvas[_osd_y1:_osd_y2, _osd_x1:_osd_x2]
                _osd_src = _osd_composite_text[_osd_crop]
                _osd_shd = _osd_composite_shadow[_osd_crop]
                if _shutter_osd_fade < 1.0:
                    _osd_src = cv2.convertScaleAbs(_osd_src, alpha=_shutter_osd_fade, beta=0)
                    _osd_shd = cv2.convertScaleAbs(_osd_shd, alpha=_shutter_osd_fade, beta=0)
                cv2.subtract(_osd_roi, _osd_shd, dst=_osd_roi)
                cv2.add(_osd_roi, _osd_src, dst=_osd_roi)

        # Apply capture black-flash overlay (if active)
        if _black_flash_frames > 0:
            display_canvas = _black_canvas
        else:
            # Optimized: use pre-allocated buffer for flip operations
            if flip_mode == FLIP_MODE_MIRROR:
                cv2.flip(canvas, 1, dst=_flip_buffer)
                display_canvas = _flip_buffer
            elif flip_mode == FLIP_MODE_MIRROR_ROTATE:
                cv2.flip(canvas, 0, dst=_flip_buffer)
                display_canvas = _flip_buffer
            else:
                display_canvas = canvas

        cv2.imshow("Camera", display_canvas)
        if cv2.waitKey(1) == 27:
            user_exit = True
            break

        # Decrement black-flash frames after presenting a frame
        if _black_flash_frames > 0:
            _black_flash_frames -= 1

        _pace_frame()
except Exception as exc:
    print("Camera preview crash:", exc)
    traceback.print_exc()
    sys.exit(1)

cv2.destroyAllWindows()
# Stop video recording gracefully before shutting down
try:
    if _video_recording:
        _stop_video_recording()
except Exception:
    pass
picam2.stop()
if user_exit:
    sys.exit(0)
sys.exit(0)
