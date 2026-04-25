#!/usr/bin/env python3
"""
Minimal IMX492 mono camera skeleton.

Preview:  half-res sensor mode (4096×2796 @ 12-bit) → 800×480 display
Capture:  full-res raw DNG + 4000×4000 16 MP JPG (press SPACE)
Quit:     press ESC or 'q'

This is the clean-room rebuild that avoids the banding issue the full
app (wlf8.py) hits when pinning the half-res sensor mode.  The test
script proved all three picamera2 config approaches work in isolation;
this skeleton validates that a full capture+preview loop can work too.
"""

import cv2
import numpy as np
import os
import struct
import time
from pathlib import Path
from picamera2 import Picamera2

# ── Display ──────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 800, 480

# ── Sensor ───────────────────────────────────────────────────────────
CAPTURE_W, CAPTURE_H = 4000, 4000  # ISP-downsampled main stream
PREVIEW_MODE_SIZE = (4096, 2796)   # half-res sensor mode for preview
PREVIEW_MODE_BIT_DEPTH = 12
PREVIEW_RAW_FMT = "R12_CSI2P"

# ── Output ───────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.expanduser("~/captures")
JPEG_QUALITY = 92


def _rotate_dng_180(dng_path):
    """Flip DNG orientation tag to 180° (EXIF orientation=3) in-place."""
    orientation_tag = 0x0112
    type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}

    if not dng_path or not os.path.isfile(dng_path):
        return
    try:
        with open(dng_path, "r+b") as f:
            header = f.read(8)
            if len(header) < 8:
                return
            endian = header[:2]
            if endian == b"II":
                pfx = "<"
            elif endian == b"MM":
                pfx = ">"
            else:
                return
            if struct.unpack(pfx + "H", header[2:4])[0] != 42:
                return
            ifd_offset = struct.unpack(pfx + "I", header[4:8])[0]

            while ifd_offset:
                f.seek(ifd_offset)
                num_entries = struct.unpack(pfx + "H", f.read(2))[0]
                for _ in range(num_entries):
                    entry = f.read(12)
                    tag = struct.unpack(pfx + "H", entry[0:2])[0]
                    if tag == orientation_tag:
                        f.seek(f.tell() - 4)
                        f.write(struct.pack(pfx + "H", 3))
                        return
                next_ifd = struct.unpack(pfx + "I", f.read(4))[0]
                ifd_offset = next_ifd if next_ifd else 0
    except Exception as e:
        print(f"[DNG] rotate error: {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    picam2 = Picamera2()

    # Detect sensor properties
    full_w, full_h = picam2.camera_properties.get("PixelArraySize", (0, 0))
    if not (full_w and full_h):
        for m in getattr(picam2, "sensor_modes", []):
            w, h = m.get("size", (0, 0))
            if w * h > full_w * full_h:
                full_w, full_h = w, h

    # Find the raw format for full-res capture
    raw_format = None
    for m in getattr(picam2, "sensor_modes", []):
        if m.get("size") == (full_w, full_h):
            raw_format = m.get("unpacked") or m.get("format")
            if raw_format:
                break
    if not raw_format:
        raw_format = "R16"

    print(f"[Camera] Sensor: full={full_w}x{full_h} raw={raw_format}")
    print(f"[Camera] Capture: {CAPTURE_W}x{CAPTURE_H} JPG + full-res DNG")

    # Log all sensor modes
    for m in getattr(picam2, "sensor_modes", []):
        if m.get("size"):
            print(
                f"  - {m['size']} @ {m.get('fps', '?')} fps "
                f"bit_depth={m.get('bit_depth', '?')} "
                f"format={m.get('unpacked') or m.get('format')}"
            )

    # ── Preview config: half-res sensor mode ─────────────────────────
    # Forces the 4096×2796 12-bit mode via an explicit raw stream.
    # This is the approach that tested clean in test_modes.py (test 2).
    preview_config = picam2.create_video_configuration(
        main={"size": (SCREEN_W, SCREEN_H), "format": "RGB888"},
        raw={"size": PREVIEW_MODE_SIZE, "format": PREVIEW_RAW_FMT},
        buffer_count=4,
    )

    # ── Still config: full-res capture ───────────────────────────────
    still_config = picam2.create_still_configuration(
        main={"size": (CAPTURE_W, CAPTURE_H), "format": "RGB888"},
        raw={"size": (full_w, full_h), "format": raw_format},
        buffer_count=2,
    )

    # Start in preview mode
    picam2.configure(preview_config)
    picam2.start()
    print("[Camera] Preview started — SPACE to capture, ESC/q to quit")

    cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    image_count = 0
    capturing = False

    # Timing
    _dbg_n = 0
    _dbg_cap = 0.0
    _dbg_proc = 0.0

    while True:
        if capturing:
            # ── Capture path ─────────────────────────────────────────
            t_cap_start = time.perf_counter()

            # Flash feedback
            canvas[:] = 0
            cv2.imshow("Camera", canvas)
            cv2.waitKey(1)

            # Switch to full-res
            picam2.switch_mode(still_config)

            capture_request = picam2.capture_request()
            try:
                ts = time.strftime("%Y%m%d_%H%M%S")
                dng_path = os.path.join(OUTPUT_DIR, f"IMG_{ts}_{image_count:04d}.dng")
                jpg_path = os.path.join(OUTPUT_DIR, f"IMG_{ts}_{image_count:04d}.jpg")

                # Save DNG (full-res raw)
                capture_request.save_dng(dng_path)
                _rotate_dng_180(dng_path)

                # Get main stream (4000×4000 RGB888)
                frame = capture_request.make_array("main")
            finally:
                capture_request.release()

            # Switch back to preview immediately
            picam2.switch_mode(preview_config)

            # Save JPG in background-ish (after mode switch so preview resumes)
            if frame is not None:
                # Mono: collapse to single channel
                if frame.ndim == 3:
                    save_frame = np.ascontiguousarray(frame[:, :, 0])
                else:
                    save_frame = frame
                save_frame = cv2.rotate(save_frame, cv2.ROTATE_180)
                cv2.imwrite(jpg_path, save_frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                image_count += 1
                elapsed = (time.perf_counter() - t_cap_start) * 1000
                fsize_kb = os.path.getsize(jpg_path) // 1024
                print(f"[Capture] {jpg_path} ({fsize_kb} KB) + DNG  [{elapsed:.0f}ms]")

            capturing = False
            continue

        # ── Preview path ─────────────────────────────────────────────
        t0 = time.perf_counter()
        frame = picam2.capture_array("main")
        t_capture = time.perf_counter()

        # Rotate 180° for mounted display
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Scale to fit display
        fh, fw = frame.shape[:2]
        scale = min(SCREEN_W / fw, SCREEN_H / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        x_off = (SCREEN_W - disp_w) // 2
        y_off = (SCREEN_H - disp_h) // 2

        scaled = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        canvas[:] = 0
        canvas[y_off:y_off + disp_h, x_off:x_off + disp_w] = scaled

        cv2.imshow("Camera", canvas)
        t_end = time.perf_counter()

        # Timing stats every 60 frames
        _dbg_n += 1
        _dbg_cap += (t_capture - t0)
        _dbg_proc += (t_end - t_capture)
        if _dbg_n >= 60:
            avg_cap = _dbg_cap / _dbg_n * 1000
            avg_proc = _dbg_proc / _dbg_n * 1000
            print(
                f"[Perf] capture={avg_cap:.1f}ms  "
                f"process={avg_proc:.1f}ms  "
                f"total={avg_cap + avg_proc:.1f}ms  "
                f"({1000 / (avg_cap + avg_proc):.0f} fps)"
            )
            _dbg_n = 0
            _dbg_cap = 0.0
            _dbg_proc = 0.0

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' '):
            capturing = True

    picam2.stop()
    cv2.destroyAllWindows()
    print(f"[Camera] Done — {image_count} images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
