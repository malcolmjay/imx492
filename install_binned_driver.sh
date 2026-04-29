#!/bin/bash
#
# Install the patched IMX492 driver with 2×2 binned mode support.
#
# Run this on a fresh Raspberry Pi 5 with the IMX492 sensor connected.
# Requires: git, build-essential, linux-headers for your kernel.
#
# Usage:
#   bash install_binned_driver.sh
#
# After install the Pi will reboot automatically.
# The binned mode (3792×2840 @ ~35 fps, 12-bit) will appear alongside
# the existing unbinned modes in `libcamera-hello --list-cameras`.
#
set -euo pipefail

echo "=== IMX492 Binned Mode Driver Install ==="
echo ""

# ── Prerequisites ─────────────────────────────────────────────────
echo "[1/6] Checking prerequisites..."

if [ "$(id -u)" -eq 0 ]; then
    echo "ERROR: Do not run this script as root. It will use sudo when needed."
    exit 1
fi

for cmd in git make gcc; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' not found. Install with:"
        echo "  sudo apt update && sudo apt install -y git build-essential"
        exit 1
    fi
done

KDIR="/lib/modules/$(uname -r)/build"
if [ ! -d "$KDIR" ]; then
    echo "ERROR: Kernel headers not found at $KDIR"
    echo "  sudo apt update && sudo apt install -y linux-headers-$(uname -r)"
    exit 1
fi

echo "  OK"

# ── Clone the upstream IMX492 driver ──────────────────────────────
echo "[2/6] Cloning upstream IMX492 V4L2 driver..."

WORKDIR="$HOME/imx492-v4l2-driver"
if [ -d "$WORKDIR" ]; then
    echo "  $WORKDIR already exists, pulling latest..."
    git -C "$WORKDIR" pull --ff-only || true
else
    git clone https://github.com/will127534/imx492-v4l2-driver.git "$WORKDIR"
fi

echo "  OK"

# ── Fetch the patched driver files ────────────────────────────────
echo "[3/6] Fetching patched driver files (binned mode)..."

PATCH_REPO="$HOME/imx492-patch"
BRANCH="claude/add-imx492-sensor-support-iBV0t"

if [ -d "$PATCH_REPO" ]; then
    git -C "$PATCH_REPO" fetch origin "$BRANCH"
    git -C "$PATCH_REPO" checkout "$BRANCH"
    git -C "$PATCH_REPO" reset --hard "origin/$BRANCH"
else
    git clone -b "$BRANCH" https://github.com/malcolmjay/imx492.git "$PATCH_REPO"
fi

cp "$PATCH_REPO/driver/imx492.c" "$WORKDIR/imx492.c"
cp "$PATCH_REPO/driver/imx492_mode_tables.h" "$WORKDIR/imx492_mode_tables.h"

echo "  OK"

# ── Build ─────────────────────────────────────────────────────────
echo "[4/6] Building kernel module..."

make -C "$WORKDIR" clean
make -C "$WORKDIR"

if [ ! -f "$WORKDIR/imx492.ko" ]; then
    echo "ERROR: Build failed — imx492.ko not produced."
    exit 1
fi

echo "  OK"

# ── Install ───────────────────────────────────────────────────────
echo "[5/6] Installing kernel module..."

MODDIR="/lib/modules/$(uname -r)"
INSTALL_PATH="$MODDIR/kernel/drivers/media/i2c"

sudo cp "$WORKDIR/imx492.ko" "$INSTALL_PATH/imx492.ko"

# Remove any DKMS copy that would take priority over ours
DKMS_PATH="$MODDIR/updates/dkms/imx492.ko.xz"
if [ -f "$DKMS_PATH" ]; then
    echo "  Removing old DKMS module at $DKMS_PATH"
    sudo rm "$DKMS_PATH"
fi

sudo depmod -a

echo "  OK — installed to $INSTALL_PATH/imx492.ko"

# ── Reboot ────────────────────────────────────────────────────────
echo "[6/6] Rebooting..."
echo ""
echo "After reboot, verify with:"
echo "  libcamera-hello --list-cameras"
echo ""
echo "You should see a 3792x2840 R12_CSI2P mode at ~35 fps."
echo ""

sudo reboot
