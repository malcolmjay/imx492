#!/bin/bash
#
# Install IMX492 2×2 binned mode driver patch.
#
# This patches the will127534 IMX492 V4L2 driver to add a 3792×2840
# (3704×2778 active) 12-bit 2×2 binned readout mode, ported from the
# IMX294 driver.  The sensors share the same die.
#
# Usage:
#   cd /path/to/imx492-v4l2-driver
#   bash /path/to/install_binned_mode.sh
#
# After patching, rebuild the driver with:
#   make clean && make && sudo make install
#   sudo reboot
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/imx492-binned-mode.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Patch file not found: $PATCH_FILE"
    exit 1
fi

if [ ! -f "imx492.c" ] || [ ! -f "imx492_mode_tables.h" ]; then
    echo "ERROR: Run this script from the imx492-v4l2-driver directory."
    echo "  cd /path/to/imx492-v4l2-driver"
    echo "  bash $0"
    exit 1
fi

echo "Applying IMX492 binned mode patch..."
if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "Patch applied successfully via git apply."
elif patch --dry-run -p1 < "$PATCH_FILE" >/dev/null 2>&1; then
    patch -p1 < "$PATCH_FILE"
    echo "Patch applied successfully via patch."
else
    echo "ERROR: Patch does not apply cleanly. The driver source may have"
    echo "       already been patched or has diverged from the expected version."
    exit 1
fi

echo ""
echo "Next steps:"
echo "  make clean && make"
echo "  sudo make install"
echo "  sudo reboot"
