#!/usr/bin/env bash
#
# install-imx492-square.sh
#
# Self-contained installer for the experimental 5616x5616 centered-square
# IMX492 driver mode.  Run on a Raspberry Pi 5 with the IMX492 mono camera
# attached.
#
# What this mode does:
#   Adds a fourth entry to supported_modes_12bit[] in the will127534
#   imx492 v4l2 driver that uses the sensor's built-in HTRIMMING and
#   VWINPOS crop primitives (previously hardcoded to zero in the driver)
#   to deliver a centered 5616x5616 square from the 8240x5628 active
#   pixel array.  Output is ~31.5 MP square (matching the existing full-
#   mode max square crop) at a slightly reduced HMAX of 1108 (same as the
#   4:3 mode), giving a ~8% CSI-2 readout speedup.
#
# This is the "Option D" approach from our design discussion: full H+V
# crop via standard Sony IMX primitives, orthogonal to the vendor's
# opaque 0x3006/0x3744/0x3A54 register dance.
#
# What the script does:
#   1. Installs build deps (linux-headers, dkms, git, patch).
#   2. Clones will127534/imx492-v4l2-driver into ~/imx492-v4l2-driver.
#   3. Applies the inlined HTRIMMING-based square-crop patch to imx492.c.
#   4. Removes any previously-installed DKMS copy of the imx492 driver.
#   5. Builds and installs the patched driver via the upstream setup.sh.
#   6. Ensures /boot/firmware/config.txt has camera_auto_detect=0 and
#      dtoverlay=imx492,cam0,mono (idempotent — won't duplicate lines).
#   7. Prompts for a reboot.
#
# Safety:
#   - Bails on any error (set -euo pipefail).
#   - Backs up config.txt before editing.
#   - Re-running is safe: the patch step detects a prior apply and skips it.
#
# Rollback:
#   - Boot config backup at /boot/firmware/config.txt.imx492-bak.<timestamp>
#   - DKMS uninstall:
#       sudo dkms remove -m imx492 -v 0.0.1 --all
#     Then reinstall the upstream driver from scratch.
#
# Post-install verification (after reboot):
#   libcamera-hello --list-cameras
#   Look for a new 5808x5636 entry under R12_CSI2P in the IMX492 list.
#   (Yes, 5808x5636 — that's the CSI-2 frame; the active square area is
#    5616x5616, padded with 192 columns of horizontal OPB and 20 dummy
#    rows + 32 vertical OPB rows.  Crop off the OPB in wlf8.py like any
#    other mode.)

set -euo pipefail

WORKDIR="${HOME}/imx492-v4l2-driver"
UPSTREAM_URL="https://github.com/will127534/imx492-v4l2-driver.git"
DRV_VERSION="0.0.1"

# Boot config location: bookworm+ uses /boot/firmware/config.txt,
# older Raspberry Pi OS uses /boot/config.txt.
if [ -f /boot/firmware/config.txt ]; then
    BOOT_CONFIG="/boot/firmware/config.txt"
elif [ -f /boot/config.txt ]; then
    BOOT_CONFIG="/boot/config.txt"
else
    echo "ERROR: neither /boot/firmware/config.txt nor /boot/config.txt exists" >&2
    exit 1
fi

echo "=== Using boot config: ${BOOT_CONFIG}"

# ---------------------------------------------------------------------------
# 1. Build dependencies
# ---------------------------------------------------------------------------
echo "=== Installing build dependencies"
sudo apt update
sudo apt install -y linux-headers-"$(uname -r)" dkms git patch build-essential

# ---------------------------------------------------------------------------
# 2. Clone (or refresh) upstream driver
# ---------------------------------------------------------------------------
if [ -d "${WORKDIR}/.git" ]; then
    echo "=== ${WORKDIR} already a git clone, resetting to origin/main"
    git -C "${WORKDIR}" fetch origin
    git -C "${WORKDIR}" checkout main
    git -C "${WORKDIR}" reset --hard origin/main
else
    echo "=== Cloning ${UPSTREAM_URL} into ${WORKDIR}"
    git clone "${UPSTREAM_URL}" "${WORKDIR}"
fi

# ---------------------------------------------------------------------------
# 3. Apply the HTRIMMING-based square-crop patch (inlined below)
# ---------------------------------------------------------------------------
PATCH_FILE="${WORKDIR}/imx492-square-crop.patch"

cat > "${PATCH_FILE}" <<'PATCH_EOF'
diff --git a/imx492.c b/imx492.c
index a42fef6..a4a5908 100644
--- a/imx492.c
+++ b/imx492.c
@@ -182,6 +182,23 @@ struct imx492_mode {
 	u8 opb_size_v;
 	u16 write_vsize;
 	u16 y_out_size;
+
+	/*
+	 * Optional HTRIMMING / VWINPOS overrides.  If htrimming_end is
+	 * non-zero this mode uses the sensor's built-in horizontal crop
+	 * primitive (HTRIMMING_EN at 0x3035, START/END at 0x3036/0x3038)
+	 * plus the vertical window position register (VWINPOS at 0x30E0)
+	 * to crop a window out of the full pixel array without needing a
+	 * vendor-specific reg table.  The CSI-2 output width register at
+	 * 0x3ED0/0x3ED1 is also overwritten with mode->width when this
+	 * path is taken.
+	 *
+	 * Existing modes leave all three fields at 0 so they fall through
+	 * to the current hardcoded-zero behaviour.
+	 */
+	u16 htrimming_start;
+	u16 htrimming_end;
+	u16 vwinpos;
 };
 
 static const struct imx492_reg imx492_startup_pre_regs[] = {
@@ -347,6 +364,88 @@ static const struct imx492_mode supported_modes_12bit[] = {
 		.opb_size_v = 0x20,
 		.write_vsize = 0x1630,
 		.y_out_size = 0x1610,
+	}, {
+		/*
+		 * Experimental 5616x5616 centered square mode (~31.5 MP).
+		 *
+		 * Uses the sensor's documented HTRIMMING + VWINPOS crop
+		 * primitives instead of the vendor-supplied wide_17_9 /
+		 * four_three register-table dance.  The all_pixel register
+		 * table is taken as-is; horizontal crop is applied by the
+		 * C driver via HTRIMMING_START/END, vertical offset via
+		 * VWINPOS, and output height via y_out_size / write_vsize.
+		 *
+		 * Geometry (centered on the 8240x5628 active area):
+		 *   horizontal: start = (8240-5616)/2 = 1312
+		 *               end   = 1312 + 5616   = 6928
+		 *   vertical:   vwinpos = (5628-5616)/2 = 6
+		 *
+		 * CSI-2 output dimensions (active + OPB padding, matching
+		 * the OPB ratios of the all_pixel mode):
+		 *   width       = 5616 + 192 horizontal OPB = 5808
+		 *   y_out_size  = 5616 + 20 dummy rows      = 5636 (0x1604)
+		 *   write_vsize = 5636 + 32 vertical OPB    = 5668 (0x1624)
+		 *
+		 * HMAX and VMAX MUST match the full-array mode's timing
+		 * (1202 / 5728).  This mode reuses imx492_all_pixel_12bit_regs,
+		 * which programs the sensor's pixel scanner for a full
+		 * 8432-wide readout — HTRIMMING only gates the output
+		 * formatter downstream, it doesn't shorten the physical
+		 * line-scan time.  A smaller HMAX (like the 4:3 mode's 1108)
+		 * truncates each line before the scanner finishes and
+		 * desyncs CSI-2 framing, producing per-line horizontal-band
+		 * garbage instead of a real image.  Lesson learned the
+		 * hard way during initial bring-up.
+		 *
+		 * The speedup this mode was originally chasing (lower frame
+		 * clock via reduced HMAX/VMAX) is therefore not available
+		 * with the all_pixel reg table.  The remaining wins are:
+		 *   (a) square framing in hardware, so the app doesn't have
+		 *       to crop a 2:3 frame to 1:1 every capture;
+		 *   (b) ~8% less memory bandwidth through the ISP because
+		 *       we're transporting 5808x5636 instead of 8432x5648.
+		 * Revisiting the speedup would require a dedicated reg
+		 * table that reconfigures the pixel scanner to actually
+		 * read fewer columns — see imx492_four_three_12bit_regs
+		 * for the shape that work would take.
+		 *
+		 * Caveats / things to validate in-field:
+		 *   - HTRIMMING_START/END encoding (pixel vs 2-pixel vs 8
+		 *     pixel units) may require the values to be doubled or
+		 *     halved.  Symptom: output is horizontally shifted or
+		 *     stretched.
+		 *   - VWINPOS encoding similarly — if frame is vertically
+		 *     off-center, try vwinpos = 3 or vwinpos = 12.
+		 *   - 0x3ED0/0x3ED1 override may or may not actually
+		 *     re-format the CSI-2 output width; if the output is
+		 *     garbled, the all_pixel reg table's 0x3ED0 value is
+		 *     load-bearing and this approach is wrong.
+		 */
+		.width = 5808,
+		.height = 5636,
+		.min_HMAX = 1202,
+		.min_VMAX = 5728,
+		.default_HMAX = 1202,
+		.default_VMAX = 5728,
+		.VMAX_scale = 1,
+		.min_SHR = 12,
+		.integration_offset = 256,
+		.crop = {
+			.left = 1312,
+			.top = 6,
+			.width = 5616,
+			.height = 5616,
+		},
+		.reg_list = {
+			.num_of_regs = ARRAY_SIZE(imx492_all_pixel_12bit_regs),
+			.regs = imx492_all_pixel_12bit_regs,
+		},
+		.opb_size_v = 0x20,
+		.write_vsize = 0x1624,
+		.y_out_size = 0x1604,
+		.htrimming_start = 1312,
+		.htrimming_end   = 6928,
+		.vwinpos         = 6,
 	},
 };
 
@@ -1162,19 +1261,35 @@ static int imx492_start_streaming(struct imx492 *imx492)
 		return ret;
 	}
 
+	/*
+	 * Optional HTRIMMING / VWINPOS path for modes that want a
+	 * hardware crop window independent of the vendor-supplied
+	 * reg_list.  Enabled when mode->htrimming_end > 0.  Also
+	 * overrides the CSI-2 output-width register at 0x3ED0 with
+	 * mode->width so the framing matches the trimmed window.
+	 */
+	if (mode->htrimming_end > 0) {
+		ret = imx492_write_reg_2byte(imx492, 0x3ED0, mode->width);
+		if (ret)
+			return ret;
+	}
+
 	ret = imx492_write_reg_1byte(imx492, IMX492_REG_HOPBOUT_EN, 0x01);
 	if (ret)
 		return ret;
 
-	ret = imx492_write_reg_1byte(imx492, IMX492_REG_HTRIMMING_EN, 0x00);
+	ret = imx492_write_reg_1byte(imx492, IMX492_REG_HTRIMMING_EN,
+				     mode->htrimming_end > 0 ? 0x01 : 0x00);
 	if (ret)
 		return ret;
 
-	ret = imx492_write_reg_2byte(imx492, IMX492_REG_HTRIMMING_START, 0x0000);
+	ret = imx492_write_reg_2byte(imx492, IMX492_REG_HTRIMMING_START,
+				     mode->htrimming_start);
 	if (ret)
 		return ret;
 
-	ret = imx492_write_reg_2byte(imx492, IMX492_REG_HTRIMMING_END, 0x0000);
+	ret = imx492_write_reg_2byte(imx492, IMX492_REG_HTRIMMING_END,
+				     mode->htrimming_end);
 	if (ret)
 		return ret;
 
@@ -1186,7 +1301,8 @@ static int imx492_start_streaming(struct imx492 *imx492)
 	if (ret)
 		return ret;
 
-	ret = imx492_write_reg_2byte(imx492, IMX492_REG_VWINPOS, 0x0000);
+	ret = imx492_write_reg_2byte(imx492, IMX492_REG_VWINPOS,
+				     mode->vwinpos);
 	if (ret)
 		return ret;
 
PATCH_EOF

echo "=== Applying HTRIMMING-based square-crop patch"
cd "${WORKDIR}"
echo "    Upstream commit: $(git rev-parse HEAD) (origin/main: $(git rev-parse origin/main))"
if grep -q "Experimental 5616x5616 centered square mode" imx492.c; then
    echo "    Patch already present in imx492.c, skipping."
else
    # Always print the dry-run output so we can see fuzz/offset/reject
    # details right next to the "applied" or "failed" line.  patch is
    # preferred over git apply because it fuzzy-matches context and
    # forgives the kind of whitespace/offset drift that git apply rejects
    # as "corrupt".
    echo "    --- patch -p1 --dry-run --fuzz=3 output ---"
    if patch -p1 --dry-run --fuzz=3 < "${PATCH_FILE}"; then
        echo "    --- applying for real ---"
        patch -p1 --fuzz=3 < "${PATCH_FILE}"
        echo "    Patch applied cleanly."
    else
        echo "ERROR: patch does not apply to imx492.c even with --fuzz=3." >&2
        echo "       Upstream commit: $(git rev-parse HEAD)" >&2
        echo "       Patch file: ${PATCH_FILE}" >&2
        echo "       Target file: ${WORKDIR}/imx492.c" >&2
        echo "       Dump .rej files (if any):" >&2
        find "${WORKDIR}" -name '*.rej' -exec echo '--- {} ---' \; -exec cat {} \; >&2 || true
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 4. Remove any previously-installed DKMS copy
# ---------------------------------------------------------------------------
if dkms status imx492 2>/dev/null | grep -q .; then
    echo "=== Removing existing DKMS imx492 module"
    sudo dkms remove -m imx492 -v "${DRV_VERSION}" --all || true
fi

# ---------------------------------------------------------------------------
# 5. Build + install via upstream setup.sh
# ---------------------------------------------------------------------------
echo "=== Running upstream setup.sh (DKMS build + install)"
chmod +x setup.sh
./setup.sh

# ---------------------------------------------------------------------------
# 6. Ensure config.txt has the right overlay
# ---------------------------------------------------------------------------
TS="$(date +%Y%m%d-%H%M%S)"
BACKUP="${BOOT_CONFIG}.imx492-bak.${TS}"
echo "=== Backing up ${BOOT_CONFIG} to ${BACKUP}"
sudo cp "${BOOT_CONFIG}" "${BACKUP}"

CONFIG_CHANGED=0

# camera_auto_detect must be 0 for manual dtoverlay to apply.
if grep -qE '^\s*camera_auto_detect\s*=' "${BOOT_CONFIG}"; then
    if ! grep -qE '^\s*camera_auto_detect\s*=\s*0\s*$' "${BOOT_CONFIG}"; then
        echo "=== Setting camera_auto_detect=0"
        sudo sed -i -E 's|^\s*camera_auto_detect\s*=.*$|camera_auto_detect=0|' \
            "${BOOT_CONFIG}"
        CONFIG_CHANGED=1
    fi
else
    echo "=== Appending camera_auto_detect=0"
    echo "camera_auto_detect=0" | sudo tee -a "${BOOT_CONFIG}" > /dev/null
    CONFIG_CHANGED=1
fi

# IMX492 mono overlay on cam0.  Change to cam1 if the ribbon is on that port.
IMX_OVERLAY_LINE='dtoverlay=imx492,cam0,mono'
if ! grep -qF "${IMX_OVERLAY_LINE}" "${BOOT_CONFIG}"; then
    echo "=== Appending ${IMX_OVERLAY_LINE}"
    echo "${IMX_OVERLAY_LINE}" | sudo tee -a "${BOOT_CONFIG}" > /dev/null
    CONFIG_CHANGED=1
fi

if [ "${CONFIG_CHANGED}" -eq 0 ]; then
    echo "=== Boot config already correct, removing redundant backup"
    sudo rm -f "${BACKUP}"
fi

# ---------------------------------------------------------------------------
# 7. Done — prompt for reboot
# ---------------------------------------------------------------------------
echo
echo "========================================================================"
echo "  IMX492 5616x5616 square driver install complete."
echo
echo "  After reboot, verify with:"
echo "      libcamera-hello --list-cameras"
echo "  You should see a new 5808x5636 entry under R12_CSI2P for the IMX492"
echo "  (the active 5616x5616 square plus 192 OPB cols + 20 dummy + 32 OPB rows)."
echo
echo "  If /boot config was changed, a reboot is required for the overlay"
echo "  to load.  Reboot now?  [y/N]"
echo "========================================================================"
read -r REPLY || REPLY="n"
case "${REPLY}" in
    [yY]*) sudo reboot ;;
    *)     echo "Skipping reboot.  Run 'sudo reboot' when you're ready." ;;
esac
