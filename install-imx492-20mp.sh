#!/usr/bin/env bash
#
# install-imx492-20mp.sh
#
# Self-contained installer for the experimental 20 MP IMX492 driver mode.
# Run on a Raspberry Pi 5 with the IMX492 mono camera attached.
#
# What it does:
#   1. Installs build deps (linux-headers, dkms, git, patch).
#   2. Clones will127534/imx492-v4l2-driver into ~/imx492-v4l2-driver.
#   3. Applies the inlined 20 MP vertical-crop mode patch to imx492.c.
#   4. Removes any previously-installed DKMS copy of the imx492 driver.
#   5. Builds and installs the patched driver via the upstream setup.sh.
#   6. Ensures /boot/firmware/config.txt has camera_auto_detect=0 and
#      dtoverlay=imx492,cam0,mono (idempotent — won't duplicate lines).
#   7. Prompts for a reboot.
#
# Safety:
#   - Bails on any error (set -euo pipefail).
#   - Backs up config.txt before editing.
#   - Does not touch anything outside ~/imx492-v4l2-driver and the boot config.
#   - Re-running is safe: the patch step detects a prior apply and skips it.
#
# Rollback:
#   - Boot config backup at /boot/firmware/config.txt.imx492-bak.<timestamp>
#   - DKMS uninstall:
#       sudo dkms remove -m imx492 -v 0.0.1 --all
#     Then reinstall the upstream driver from scratch.

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
# 3. Apply the 20 MP mode patch (inlined below)
# ---------------------------------------------------------------------------
PATCH_FILE="${WORKDIR}/imx492-20mp-mode.patch"

cat > "${PATCH_FILE}" <<'PATCH_EOF'
diff --git a/imx492.c b/imx492.c
--- a/imx492.c
+++ b/imx492.c
@@ -347,6 +347,49 @@ static const struct imx492_mode supported_modes_12bit[] = {
 		.opb_size_v = 0x20,
 		.write_vsize = 0x1630,
 		.y_out_size = 0x1610,
+	}, {
+		/*
+		 * Experimental 20 MP vertical-crop mode.
+		 *
+		 * Reuses the wide 17:9 register table and simply asks the
+		 * sensor for fewer output rows via write_vsize / y_out_size.
+		 * Output is 8432 x 2372 = ~20.0 MP, aspect ~3.56:1.
+		 *
+		 * Purpose: test whether shrinking y_out_size proportionally
+		 * reduces CSI-2 readout time on this sensor, which is the
+		 * real bottleneck on Pi 5.  The mystery vertical-window
+		 * registers at 0x3A54 / 0x3A55 stay at the wide-mode values,
+		 * so centered-crop behaviour is not guaranteed — the output
+		 * may be top-shifted within the wide window rather than
+		 * centered on the full frame.  Validate in-field before
+		 * relying on the .top value below.
+		 *
+		 * min_VMAX = 2452 = 2372 + 80 vblank margin.
+		 * write_vsize = 0x0964 = 2404  (2372 + 32 OPB rows).
+		 * y_out_size  = 0x0944 = 2372.
+		 */
+		.width = 8432,
+		.height = 2372,
+		.min_HMAX = 1202,
+		.min_VMAX = 2452,
+		.default_HMAX = 1202,
+		.default_VMAX = 2452,
+		.VMAX_scale = 1,
+		.min_SHR = 12,
+		.integration_offset = 256,
+		.crop = {
+			.left = 0,
+			.top = 1628,
+			.width = 8240,
+			.height = 2360,
+		},
+		.reg_list = {
+			.num_of_regs = ARRAY_SIZE(imx492_wide_17_9_12bit_regs),
+			.regs = imx492_wide_17_9_12bit_regs,
+		},
+		.opb_size_v = 0x20,
+		.write_vsize = 0x0964,
+		.y_out_size = 0x0944,
 	},
 };

PATCH_EOF

echo "=== Applying 20 MP mode patch"
cd "${WORKDIR}"
if git apply --check "${PATCH_FILE}" 2>/dev/null; then
    git apply "${PATCH_FILE}"
    echo "    Patch applied cleanly."
elif grep -q "Experimental 20 MP vertical-crop mode" imx492.c; then
    echo "    Patch already present in imx492.c, skipping."
else
    echo "ERROR: patch does not apply cleanly and the marker comment is not" >&2
    echo "       already present.  Upstream may have moved — inspect" >&2
    echo "       ${PATCH_FILE} against ${WORKDIR}/imx492.c manually." >&2
    exit 1
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
echo "  IMX492 20 MP driver install complete."
echo
echo "  After reboot, verify with:"
echo "      libcamera-hello --list-cameras"
echo "  You should see an 8432x2372 entry under R12_CSI2P for the IMX492."
echo
echo "  If /boot config was changed, a reboot is required for the overlay"
echo "  to load.  Reboot now?  [y/N]"
echo "========================================================================"
read -r REPLY || REPLY="n"
case "${REPLY}" in
    [yY]*) sudo reboot ;;
    *)     echo "Skipping reboot.  Run 'sudo reboot' when you're ready." ;;
esac
