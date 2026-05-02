# IMX492 V4L2 Driver — Patched with 2×2 Binned Mode

A patched Linux V4L2 driver for the Sony IMX492 image sensor (46.4 MP, 2.315 µm quad-Bayer) adding a **2×2 binned readout mode** that outputs 3792×2840 at ~35 fps in 12-bit. Built for Raspberry Pi 5 with the PiSP ISP.

Based on the upstream driver by [will127534](https://github.com/will127534/imx492-v4l2-driver).

## Available Sensor Modes

### 10-bit (R10_CSI2P)

| Resolution | Description | Approx. FPS |
|------------|-------------|-------------|
| 8432×5648  | Full array  | ~14 fps     |
| 8432×4348  | 17:9 crop   | ~18 fps     |
| 7680×5648  | 4:3 crop    | ~15 fps     |

### 12-bit (R12_CSI2P)

| Resolution | Description         | Approx. FPS |
|------------|---------------------|-------------|
| 8432×5648  | Full array          | ~10 fps     |
| 8432×4348  | 17:9 crop           | ~14 fps     |
| 7680×5648  | 4:3 crop            | ~11 fps     |
| **3792×2824** | **2×2 binned (new)** | **~35 fps** |
| 5808×5636  | Square 1:1 crop     | ~10 fps     |

The binned mode is the primary addition. It reads the full sensor area with 2×2 pixel binning, producing a ~10.8 MP output at significantly higher frame rates than any unbinned mode.

## Installation

### Prerequisites

- Raspberry Pi 5 with IMX492 sensor connected
- Raspberry Pi OS with kernel headers installed
- `git`, `make`, `gcc` (from `build-essential`)

```bash
sudo apt update && sudo apt install -y git build-essential linux-headers-$(uname -r)
```

### One-Step Install Script

The easiest way to install is with the included script:

```bash
git clone -b claude/add-imx492-sensor-support-iBV0t https://github.com/malcolmjay/imx492.git ~/imx492-patch
bash ~/imx492-patch/install_binned_driver.sh
```

The script will clone the upstream driver, copy the patched files, build, install, and reboot.

### Manual Installation

If you prefer to install manually:

```bash
# 1. Clone the upstream IMX492 V4L2 driver
git clone https://github.com/will127534/imx492-v4l2-driver.git ~/imx492-v4l2-driver

# 2. Clone this repository (patched driver files)
git clone -b claude/add-imx492-sensor-support-iBV0t \
    https://github.com/malcolmjay/imx492.git ~/imx492-patch

# 3. Copy patched driver files over the upstream source
cp ~/imx492-patch/driver/imx492.c ~/imx492-v4l2-driver/imx492.c
cp ~/imx492-patch/driver/imx492_mode_tables.h ~/imx492-v4l2-driver/imx492_mode_tables.h

# 4. Build the kernel module
cd ~/imx492-v4l2-driver
make clean && make

# 5. Install the module
sudo cp imx492.ko /lib/modules/$(uname -r)/kernel/drivers/media/i2c/imx492.ko

# 6. Remove any conflicting DKMS module
DKMS_PATH="/lib/modules/$(uname -r)/updates/dkms/imx492.ko.xz"
[ -f "$DKMS_PATH" ] && sudo rm "$DKMS_PATH"

# 7. Rebuild module dependencies and reboot
sudo depmod -a
sudo reboot
```

### Verify Installation

After reboot, list the available camera modes:

```bash
libcamera-hello --list-cameras
```

You should see the 3792×2824 R12_CSI2P mode listed among the available modes.

### Test the Binned Mode

```bash
libcamera-hello -t 10000 --mode 3792:2824:12:P --viewfinder-width 1264 --viewfinder-height 942
```

### Cleanup (Remove Old Driver Versions)

If you have remnants of previous driver installs causing conflicts:

```bash
sudo find /lib/modules/$(uname -r) -name 'imx492.ko*' -delete && \
sudo cp ~/imx492-v4l2-driver/imx492.ko /lib/modules/$(uname -r)/kernel/drivers/media/i2c/imx492.ko && \
sudo depmod -a && \
sudo reboot
```

## What Was Changed

### Background

The Sony IMX492 and IMX294 share the same sensor die (46.4 MP, 2.315 µm pitch, quad-Bayer CFA). The IMX294's Linux driver (maintained in the Raspberry Pi kernel tree) already supports a fully validated 2×2 binned readout mode. This patch ports that binned mode to the IMX492 driver.

### Key Changes

**New register table (`imx492_binned_12bit_regs`):**
The binned mode register table was rebuilt from scratch using the IMX294 driver's validated `common_regs` + `mode_00_regs` (202 register entries total). An earlier attempt that started from the IMX492's unbinned register table and only overrode the MDSEL mode-select registers produced horizontal stretching and static — the ~100 output-formatting registers in the 0x3Exx range were left at unbinned values, which is incorrect for binned readout.

**HCOUNT override in `start_streaming()`:**
Binned mode requires HCOUNT = 1200 (0x04B0) on both HCOUNT1 and HCOUNT2 registers. Unbinned modes use HCOUNT = 0. This value comes directly from the IMX294's `common_regs` and controls the horizontal readout counter for binned pixel merging.

**Skip 0x3ED0 CSI-2 width register for binned mode:**
The driver normally writes the output width to register 0x3ED0 to set the CSI-2 line length. For binned mode, this register must not be written — the sensor auto-determines the correct CSI-2 line width from the HTRIMMING window. The IMX294 driver never writes any 0x3Exx registers, confirming this is the correct approach.

**HTRIMMING crop boundaries:**
Binned mode uses HTRIMMING_START = 0x0030 (48) and HTRIMMING_END = 0x0F00 (3840), matching the IMX294's validated boundaries. The HTRIMMING window defines the horizontal crop in binned pixel coordinates. Values beyond 0x0F00 read into optical black columns and produce static on the right edge of the frame.

**HOPBOUT_EN disabled for binned mode:**
Register 0x3034 (HOPBOUT_EN) is set to 0x00 for binned mode, disabling the horizontal optical black output that is incompatible with binned readout.

**V4L2 crop rectangle in native coordinates:**
The crop rectangle reported to libcamera uses native (unbinned) pixel coordinates: 7408×5556 with offsets of 80,48. This is required for correct mode selection and scaling by the PiSP ISP.

**Mode timing:**
- HMAX = 1875, VMAX = 1600, VMAX_scale = 2
- min_SHR = 5, integration_offset = 551
- All values ported from the IMX294 `mode_00` definition

### Files Modified

| File | Description |
|------|-------------|
| `driver/imx492.c` | Added binned mode entry to `supported_modes_12bit[]`, HCOUNT/HOPBOUT_EN/0x3ED0 logic in `start_streaming()`, TRY crop handling in `set_pad_format()` |
| `driver/imx492_mode_tables.h` | Added `imx492_binned_12bit_regs[]` (202 entries from IMX294 common_regs + mode_00_regs) |
| `install_binned_driver.sh` | One-step install script for clean Raspberry Pi setups |

## License

This driver is based on the [upstream IMX492 V4L2 driver](https://github.com/will127534/imx492-v4l2-driver) and is distributed under the same license (GPL-2.0).
