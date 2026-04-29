// SPDX-License-Identifier: GPL-2.0
/*
 * A V4L2 driver for Sony imx492 cameras.
 *
 */
#include <linux/clk.h>
#include <linux/delay.h>
#include <linux/gpio/consumer.h>
#include <linux/i2c.h>
#include <linux/module.h>
#include <linux/of_device.h>
#include <linux/pm_runtime.h>
#include <linux/regulator/consumer.h>
#include <linux/version.h>
#if __has_include(<linux/unaligned.h>)
#include <linux/unaligned.h>
#else
#include <asm/unaligned.h>
#endif
#include <media/v4l2-ctrls.h>
#include <media/v4l2-device.h>
#include <media/v4l2-event.h>
#include <media/v4l2-fwnode.h>
#include <media/v4l2-mediabus.h>
#include <linux/moduleparam.h>



int debug = 0;
module_param(debug, int, 0660);
MODULE_PARM_DESC(debug, "Debug flag");

#define DEBUG_PRINTK(fmt, ...) do { if (debug) printk(KERN_DEBUG "%s: " fmt, __this_module.name, ##__VA_ARGS__); } while(0)

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 12, 0)
#define imx492_subdev_state_get_format(sd, state, pad) \
	v4l2_subdev_state_get_format(state, pad)
#define imx492_subdev_state_get_crop(sd, state, pad) \
	v4l2_subdev_state_get_crop(state, pad)
#else
#define imx492_subdev_state_get_format(sd, state, pad) \
	v4l2_subdev_get_try_format(sd, state, pad)
#define imx492_subdev_state_get_crop(sd, state, pad) \
	v4l2_subdev_get_try_crop(sd, state, pad)
#endif


/* Chip ID */
#define IMX492_REG_CHIP_ID		0x3000
#define IMX492_CHIP_ID			0x0000

#define IMX492_REG_MODE_SELECT		0x3000
#define IMX492_MODE_STANDBY		0x01
#define IMX492_MODE_STREAMING		0x00
#define IMX492_STREAM_DELAY_US		25000
#define IMX492_STREAM_DELAY_RANGE_US	1000
#define IMX492_AUTOSUSPEND_DELAY_MS	1000

#define IMX492_XCLK_FREQ		24000000
#define IMX492_LINK_RATE		1728000000ULL
#define IMX492_LINK_FREQ		(IMX492_LINK_RATE / 2)
#define IMX492_NUM_DATA_LANES	4

/* VMAX internal VBLANK*/
#define IMX492_REG_VMAX		0x30A9
#define IMX492_VMAX_MAX		0xfffff

/* HMAX internal HBLANK*/
#define IMX492_REG_HMAX		0x30AC
#define IMX492_HMAX_MAX		0xffff

#define IMX492_REG_HCOUNT1     0x3084
#define IMX492_REG_HCOUNT2     0x3086
#define IMX492_REG_HOPBOUT_EN  0x3034
#define IMX492_REG_HTRIMMING_EN 0x3035
#define IMX492_REG_HTRIMMING_START 0x3036
#define IMX492_REG_HTRIMMING_END 0x3038
#define IMX492_REG_VWINPOS     0x30E0
#define IMX492_REG_PSSLVS1 0x332C
#define IMX492_REG_PSSLVS2 0x334A
#define IMX492_REG_PSSLVS3 0x35B6
#define IMX492_REG_PSSLVS4 0x35B8
#define IMX492_REG_PSSLVS0 0x36BC
#define IMX492_REG_OPB_SIZE_V  0x312F
#define IMX492_REG_WRITE_VSIZE 0x3130
#define IMX492_REG_Y_OUT_SIZE  0x3132


/* SHR internal */
#define IMX492_REG_SHR		0x302C
#define IMX492_SHR_MIN		12

/* Exposure control */
#define IMX492_EXPOSURE_MIN			1
#define IMX492_EXPOSURE_STEP		1
#define IMX492_EXPOSURE_DEFAULT		1000
#define IMX492_EXPOSURE_MAX		0xfffff

/* Analog gain control */
#define IMX492_REG_ANALOG_GAIN		0x300A
#define IMX492_ANA_GAIN_MIN		0
#define IMX492_ANA_GAIN_MAX		1957
#define IMX492_ANA_GAIN_STEP		1
#define IMX492_ANA_GAIN_DEFAULT		0x0

/*
 * BLKLEVEL is an 8-bit register. In 12-bit mode each register step moves the
 * output black level by 4 digits, so the support package's 200-digit nominal
 * black level corresponds to a register value of 50.
 */
#define IMX492_REG_BLKLEVEL		0x3042
#define IMX492_BLKLEVEL_MIN		0
#define IMX492_BLKLEVEL_MAX		0xff
#define IMX492_BLKLEVEL_DEFAULT		50

#define IMX492_REG_TEST_PATTERN_CTRL	0x303A
#define IMX492_REG_TEST_PATTERN_SEL	0x303B
#define IMX492_TEST_PATTERN_ENABLE_MIPI	0x11

/* Embedded metadata stream structure */
#define IMX492_EMBEDDED_LINE_WIDTH 16384
#define IMX492_NUM_EMBEDDED_LINES 1

enum pad_types {
	IMAGE_PAD,
	METADATA_PAD,
	NUM_PADS
};

/* IMX492 native active pixel array size. */
#define IMX492_NATIVE_WIDTH		8240U
#define IMX492_NATIVE_HEIGHT		5628U
#define IMX492_PIXEL_ARRAY_LEFT	0U
#define IMX492_PIXEL_ARRAY_TOP		0U
#define IMX492_PIXEL_ARRAY_WIDTH	8240U
#define IMX492_PIXEL_ARRAY_HEIGHT	5628U

struct imx492_reg {
	u16 address;
	u8 val;
};

struct IMX492_reg_list {
	unsigned int num_of_regs;
	const struct imx492_reg *regs;
};

/* Mode : resolution and related config&values */
struct imx492_mode {
	/* Frame width */
	unsigned int width;

	/* Frame height */
	unsigned int height;

	/* minimum H-timing */
	uint64_t min_HMAX;

	/* minimum V-timing */
	uint64_t min_VMAX;

	/* default H-timing */
	uint64_t default_HMAX;

	/* default V-timing */
	uint64_t default_VMAX;

    /* V-timing Scaling*/
    uint64_t VMAX_scale;

	/* minimum SHR */
	uint64_t min_SHR;

    unsigned int integration_offset;

	/* Analog crop rectangle. */
	struct v4l2_rect crop;

	/* Default register values */
	struct IMX492_reg_list reg_list;

	u8 opb_size_v;
	u16 write_vsize;
	u16 y_out_size;

	/*
	 * Optional HTRIMMING / VWINPOS overrides.  If htrimming_end is
	 * non-zero this mode uses the sensor's built-in horizontal crop
	 * primitive (HTRIMMING_EN at 0x3035, START/END at 0x3036/0x3038)
	 * plus the vertical window position register (VWINPOS at 0x30E0)
	 * to crop a window out of the full pixel array without needing a
	 * vendor-specific reg table.  The CSI-2 output width register at
	 * 0x3ED0/0x3ED1 is also overwritten with mode->width when this
	 * path is taken.
	 *
	 * Existing modes leave all three fields at 0 so they fall through
	 * to the current hardcoded-zero behaviour.
	 */
	u16 htrimming_start;
	u16 htrimming_end;
	u16 vwinpos;

	/* Set for 2×2 binned modes (affects HOPBOUT_EN, HCOUNT). */
	u8 is_binned;
};

static const struct imx492_reg imx492_startup_pre_regs[] = {
	{0x3033, 0x30},
	{0x303C, 0x01},
	{0x3000, 0x12},
	{0x310B, 0x00},
};

static const struct imx492_reg imx492_startup_post_regs[] = {
	{0xFFFE, 0x0A},
	{0x3000, 0x02},
	{0x35E5, 0x92},
	{0x35E5, 0x9A},
	{0x3000, 0x00},
	{0xFFFE, 0x0A},
	{0x3033, 0x20},
	{0x3017, 0xA8},
};

#include "imx492_mode_tables.h"

static const struct imx492_mode supported_modes_10bit[] = {
	{
		.width = 8432,
		.height = 5648,
		.min_HMAX = 920,
		.min_VMAX = 5728,
		.default_HMAX = 920,
		.default_VMAX = 5728,
		.VMAX_scale = 1,
		.min_SHR = 12,
		.integration_offset = 217,
		.crop = {
			.left = 0,
			.top = 0,
			.width = 8240,
			.height = 5628,
		},
		.reg_list = {
			.num_of_regs = ARRAY_SIZE(imx492_all_pixel_10bit_regs),
			.regs = imx492_all_pixel_10bit_regs,
		},
		.opb_size_v = 0x20,
		.write_vsize = 0x1630,
		.y_out_size = 0x1610,
	}, {
		.width = 8432,
		.height = 4348,
		.min_HMAX = 920,
		.min_VMAX = 4428,
		.default_HMAX = 920,
		.default_VMAX = 4428,
		.VMAX_scale = 1,
		.min_SHR = 12,
		.integration_offset = 217,
		.crop = {
			.left = 0,
			.top = 646,
			.width = 8240,
			.height = 4336,
		},
		.reg_list = {
			.num_of_regs = ARRAY_SIZE(imx492_wide_17_9_10bit_regs),
			.regs = imx492_wide_17_9_10bit_regs,
		},
		.opb_size_v = 0x20,
		.write_vsize = 0x111c,
		.y_out_size = 0x10fc,
	}, {
		.width = 7680,
		.height = 5648,
		.min_HMAX = 842,
		.min_VMAX = 5728,
		.default_HMAX = 842,
		.default_VMAX = 5728,
		.VMAX_scale = 1,
		.min_SHR = 12,
		.integration_offset = 217,
		.crop = {
			.left = 392,
			.top = 0,
			.width = 7456,
			.height = 5628,
		},
		.reg_list = {
			.num_of_regs = ARRAY_SIZE(imx492_four_three_10bit_regs),
			.regs = imx492_four_three_10bit_regs,
		},
		.opb_size_v = 0x20,
		.write_vsize = 0x1630,
		.y_out_size = 0x1610,
	},
};

static const struct imx492_mode supported_modes_12bit[] = {
	{
		.width = 8432,
		.height = 5648,
		.min_HMAX = 1202,
		.min_VMAX = 5728,
		.default_HMAX = 1202,
		.default_VMAX = 5728,
		.VMAX_scale = 1,
		.min_SHR = 12,
		.integration_offset = 256,
		.crop = {
			.left = 0,
			.top = 0,
			.width = 8240,
			.height = 5628,
		},
		.reg_list = {
			.num_of_regs = ARRAY_SIZE(imx492_all_pixel_12bit_regs),
			.regs = imx492_all_pixel_12bit_regs,
		},
		.opb_size_v = 0x20,
		.write_vsize = 0x1630,
		.y_out_size = 0x1610,
	}, {
		.width = 8432,
		.height = 4348,
		.min_HMAX = 1202,
		.min_VMAX = 4428,
		.default_HMAX = 1202,
		.default_VMAX = 4428,
		.VMAX_scale = 1,
		.min_SHR = 12,
		.integration_offset = 256,
		.crop = {
			.left = 0,
			.top = 646,
			.width = 8240,
			.height = 4336,
		},
		.reg_list = {
			.num_of_regs = ARRAY_SIZE(imx492_wide_17_9_12bit_regs),
			.regs = imx492_wide_17_9_12bit_regs,
		},
		.opb_size_v = 0x20,
		.write_vsize = 0x111c,
		.y_out_size = 0x10fc,
	}, {
		.width = 7680,
		.height = 5648,
		.min_HMAX = 1108,
		.min_VMAX = 5728,
		.default_HMAX = 1108,
		.default_VMAX = 5728,
		.VMAX_scale = 1,
		.min_SHR = 12,
		.integration_offset = 256,
		.crop = {
			.left = 392,
			.top = 0,
			.width = 7456,
			.height = 5628,
		},
		.reg_list = {
			.num_of_regs = ARRAY_SIZE(imx492_four_three_12bit_regs),
			.regs = imx492_four_three_12bit_regs,
		},
		.opb_size_v = 0x20,
		.write_vsize = 0x1630,
		.y_out_size = 0x1610,
	}, {
		/*
		 * 2×2 binned 12-bit mode: 3792×2840 output, 3704×2778 active.
		 *
		 * Ported from the IMX294 driver's mode_00 (full-array 2×2
		 * binned readout).  The IMX492 and IMX294 share the same
		 * die — the user confirmed dtoverlay=imx294 drives the
		 * IMX492 cleanly in this mode at ~40 fps.
		 *
		 * Timing values (HMAX, VMAX, SHR, integration_offset) are
		 * taken directly from the IMX294 mode_00 definition.
		 * VMAX_scale=2 because each VMAX tick covers two sensor
		 * rows in binned readout.
		 */
		.width = 3792,
		.height = 2840,
		.min_HMAX = 1730,
		.min_VMAX = 1444,
		.default_HMAX = 1875,
		.default_VMAX = 1600,
		.VMAX_scale = 2,
		.min_SHR = 5,
		.integration_offset = 551,
		/*
		 * Crop in native (unbinned) pixel coordinates.
		 * HTRIMMING window (48..3840) includes 44 OPB columns
		 * on each side of the 3704 active binned pixels.
		 * 3704×2 = 7408 native active width.
		 */
		.crop = {
			.left = 80,
			.top = 48,
			.width = 7408,
			.height = 5556,
		},
		.reg_list = {
			.num_of_regs = ARRAY_SIZE(imx492_binned_12bit_regs),
			.regs = imx492_binned_12bit_regs,
		},
		.opb_size_v = 0x10,
		.write_vsize = 0x0B18,
		.y_out_size = 0x0B08,
		.htrimming_start = 0x0030,
		.htrimming_end = 0x0F00,
		.is_binned = 1,
	}, {
		/*
		 * Experimental 5616x5616 centered square mode (~31.5 MP).
		 *
		 * Uses the sensor's documented HTRIMMING + VWINPOS crop
		 * primitives instead of the vendor-supplied wide_17_9 /
		 * four_three register-table dance.  The all_pixel register
		 * table is taken as-is; horizontal crop is applied by the
		 * C driver via HTRIMMING_START/END, vertical offset via
		 * VWINPOS, and output height via y_out_size / write_vsize.
		 *
		 * Geometry (centered on the 8240x5628 active area):
		 *   horizontal: start = (8240-5616)/2 = 1312
		 *               end   = 1312 + 5616   = 6928
		 *   vertical:   vwinpos = (5628-5616)/2 = 6
		 *
		 * CSI-2 output dimensions (active + OPB padding, matching
		 * the OPB ratios of the all_pixel mode):
		 *   width       = 5616 + 192 horizontal OPB = 5808
		 *   y_out_size  = 5616 + 20 dummy rows      = 5636 (0x1604)
		 *   write_vsize = 5636 + 32 vertical OPB    = 5668 (0x1624)
		 *
		 * HMAX and VMAX MUST match the full-array mode's timing
		 * (1202 / 5728).  This mode reuses imx492_all_pixel_12bit_regs,
		 * which programs the sensor's pixel scanner for a full
		 * 8432-wide readout — HTRIMMING only gates the output
		 * formatter downstream, it doesn't shorten the physical
		 * line-scan time.  A smaller HMAX (like the 4:3 mode's 1108)
		 * truncates each line before the scanner finishes and
		 * desyncs CSI-2 framing, producing per-line horizontal-band
		 * garbage instead of a real image.  Lesson learned the
		 * hard way during initial bring-up.
		 *
		 * The speedup this mode was originally chasing (lower frame
		 * clock via reduced HMAX/VMAX) is therefore not available
		 * with the all_pixel reg table.  The remaining wins are:
		 *   (a) square framing in hardware, so the app doesn't have
		 *       to crop a 2:3 frame to 1:1 every capture;
		 *   (b) ~8% less memory bandwidth through the ISP because
		 *       we're transporting 5808x5636 instead of 8432x5648.
		 * Revisiting the speedup would require a dedicated reg
		 * table that reconfigures the pixel scanner to actually
		 * read fewer columns — see imx492_four_three_12bit_regs
		 * for the shape that work would take.
		 *
		 * Caveats / things to validate in-field:
		 *   - HTRIMMING_START/END encoding (pixel vs 2-pixel vs 8
		 *     pixel units) may require the values to be doubled or
		 *     halved.  Symptom: output is horizontally shifted or
		 *     stretched.
		 *   - VWINPOS encoding similarly — if frame is vertically
		 *     off-center, try vwinpos = 3 or vwinpos = 12.
		 *   - 0x3ED0/0x3ED1 override may or may not actually
		 *     re-format the CSI-2 output width; if the output is
		 *     garbled, the all_pixel reg table's 0x3ED0 value is
		 *     load-bearing and this approach is wrong.
		 */
		.width = 5808,
		.height = 5636,
		.min_HMAX = 1202,
		.min_VMAX = 5728,
		.default_HMAX = 1202,
		.default_VMAX = 5728,
		.VMAX_scale = 1,
		.min_SHR = 12,
		.integration_offset = 256,
		.crop = {
			.left = 1312,
			.top = 6,
			.width = 5616,
			.height = 5616,
		},
		.reg_list = {
			.num_of_regs = ARRAY_SIZE(imx492_all_pixel_12bit_regs),
			.regs = imx492_all_pixel_12bit_regs,
		},
		.opb_size_v = 0x20,
		.write_vsize = 0x1624,
		.y_out_size = 0x1604,
		.htrimming_start = 1312,
		.htrimming_end   = 6928,
		.vwinpos         = 6,
	},
};

static const struct imx492_mode supported_modes_14bit[] = {
	{
		/*
		 * 2×2 binned 14-bit mode: 3792×2840 output, 3704×2778 active.
		 *
		 * Same geometry and timing as the 12-bit binned mode, but uses
		 * the sensor's 14-bit ADC path (MDSEL2=0x0B, MDSEL8=0x03).
		 * Higher bit depth costs ~14 % framerate vs 12-bit at the same
		 * MIPI link frequency, matching the IMX294 driver's
		 * mode_00_14bit definition.
		 */
		.width = 3792,
		.height = 2840,
		.min_HMAX = 1730,
		.min_VMAX = 1444,
		.default_HMAX = 1875,
		.default_VMAX = 1600,
		.VMAX_scale = 2,
		.min_SHR = 5,
		.integration_offset = 551,
		.crop = {
			.left = 80,
			.top = 48,
			.width = 7408,
			.height = 5556,
		},
		.reg_list = {
			.num_of_regs = ARRAY_SIZE(imx492_binned_14bit_regs),
			.regs = imx492_binned_14bit_regs,
		},
		.opb_size_v = 0x10,
		.write_vsize = 0x0B18,
		.y_out_size = 0x0B08,
		.htrimming_start = 0x0030,
		.htrimming_end = 0x0F00,
		.is_binned = 1,
	},
};

/*
 * The supported formats.
 * This table MUST contain 4 entries per format, to cover the various flip
 * combinations in the order
 * - no flip
 * - h flip
 * - v flip
 * - h&v flips
 */
static const u32 codes[] = {
	/* 10-bit modes. */
	MEDIA_BUS_FMT_SRGGB10_1X10,
	MEDIA_BUS_FMT_SGRBG10_1X10,
	MEDIA_BUS_FMT_SGBRG10_1X10,
	MEDIA_BUS_FMT_SBGGR10_1X10,
	/* 12-bit modes. */
	MEDIA_BUS_FMT_SRGGB12_1X12,
	MEDIA_BUS_FMT_SGRBG12_1X12,
	MEDIA_BUS_FMT_SGBRG12_1X12,
	MEDIA_BUS_FMT_SBGGR12_1X12,
	/* 14-bit modes. */
	MEDIA_BUS_FMT_SRGGB14_1X14,
	MEDIA_BUS_FMT_SGRBG14_1X14,
	MEDIA_BUS_FMT_SGBRG14_1X14,
	MEDIA_BUS_FMT_SBGGR14_1X14,
};

static const u32 mono_codes[] = {
	MEDIA_BUS_FMT_Y10_1X10,
	MEDIA_BUS_FMT_Y12_1X12,
	MEDIA_BUS_FMT_Y14_1X14,
};

static const s64 imx492_link_freq_menu[] = {
	IMX492_LINK_FREQ,
};

static const char * const imx492_test_pattern_menu[] = {
	"Disabled",
	"Solid Black",
	"Solid White",
	"Solid Pattern 2",
	"Solid Pattern 3",
	"Vertical Color Bars",
	"Horizontal Color Bars",
};

static const u8 imx492_test_pattern_sel[] = {
	0x00,
	0x01,
	0x02,
	0x03,
	0x0A,
	0x0B,
};

/* regulator supplies */
static const char * const imx492_supply_name[] = {
	/* Supplies can be enabled in any order */
	"VANA",  /* Analog (2.8V) supply */
	"VDIG",  /* Digital Core (1.05V) supply */
	"VDDL",  /* IF (1.8V) supply */
};

#define imx492_NUM_SUPPLIES ARRAY_SIZE(imx492_supply_name)

/*
 * Initialisation delay between XCLR low->high and the moment when the sensor
 * can start capture (i.e. can leave software standby), given by T7 in the
 * datasheet is 8ms.  This does include I2C setup time as well.
 *
 * Note, that delay between XCLR low->high and reading the CCI ID register (T6
 * in the datasheet) is much smaller - 600us.
 */
#define imx492_XCLR_MIN_DELAY_US	100000
#define imx492_XCLR_DELAY_RANGE_US	1000

struct imx492_compatible_data {
	unsigned int chip_id;
	bool mono;
	struct IMX492_reg_list extra_regs;
};

struct imx492 {
	struct v4l2_subdev sd;
	struct media_pad pad[NUM_PADS];

	unsigned int fmt_code;

	struct clk *xclk;
	u32 xclk_freq;

	struct gpio_desc *reset_gpio;
	struct regulator_bulk_data supplies[imx492_NUM_SUPPLIES];

	struct v4l2_ctrl_handler ctrl_handler;
	/* V4L2 Controls */
	struct v4l2_ctrl *link_freq;
	struct v4l2_ctrl *pixel_rate;
	struct v4l2_ctrl *exposure;
	struct v4l2_ctrl *vflip;
	struct v4l2_ctrl *hflip;
	struct v4l2_ctrl *vblank;
	struct v4l2_ctrl *hblank;
	struct v4l2_ctrl *blacklevel;
	struct v4l2_ctrl *test_pattern;

	/* Current mode */
	const struct imx492_mode *mode;

	uint16_t HMAX;
	uint32_t VMAX;
	/*
	 * Mutex for serialized access:
	 * Protect sensor module set pad format and start/stop streaming safely.
	 */
	struct mutex mutex;

	/* Streaming on/off */
	bool streaming;

	/* Any extra information related to different compatible sensors */
	const struct imx492_compatible_data *compatible_data;
	bool mono;
};

static inline struct imx492 *to_imx492(struct v4l2_subdev *_sd)
{
	return container_of(_sd, struct imx492, sd);
}

static inline void get_mode_table(unsigned int code,
			  const struct imx492_mode **mode_list,
			  unsigned int *num_modes)
{
	switch (code) {
	case MEDIA_BUS_FMT_SBGGR10_1X10:
	case MEDIA_BUS_FMT_SGBRG10_1X10:
	case MEDIA_BUS_FMT_SGRBG10_1X10:
	case MEDIA_BUS_FMT_SRGGB10_1X10:
	case MEDIA_BUS_FMT_Y10_1X10:
		*mode_list = supported_modes_10bit;
		*num_modes = ARRAY_SIZE(supported_modes_10bit);
		break;
	case MEDIA_BUS_FMT_SBGGR12_1X12:
	case MEDIA_BUS_FMT_SGBRG12_1X12:
	case MEDIA_BUS_FMT_SGRBG12_1X12:
	case MEDIA_BUS_FMT_SRGGB12_1X12:
	case MEDIA_BUS_FMT_Y12_1X12:
		*mode_list = supported_modes_12bit;
		*num_modes = ARRAY_SIZE(supported_modes_12bit);
		break;
	case MEDIA_BUS_FMT_SBGGR14_1X14:
	case MEDIA_BUS_FMT_SGBRG14_1X14:
	case MEDIA_BUS_FMT_SGRBG14_1X14:
	case MEDIA_BUS_FMT_SRGGB14_1X14:
	case MEDIA_BUS_FMT_Y14_1X14:
		*mode_list = supported_modes_14bit;
		*num_modes = ARRAY_SIZE(supported_modes_14bit);
		break;
	default:
		*mode_list = NULL;
		*num_modes = 0;
	}
}

static u32 imx492_get_format_bpp(u32 code)
{
	switch (code) {
	case MEDIA_BUS_FMT_SBGGR10_1X10:
	case MEDIA_BUS_FMT_SGBRG10_1X10:
	case MEDIA_BUS_FMT_SGRBG10_1X10:
	case MEDIA_BUS_FMT_SRGGB10_1X10:
	case MEDIA_BUS_FMT_Y10_1X10:
		return 10;
	case MEDIA_BUS_FMT_SBGGR12_1X12:
	case MEDIA_BUS_FMT_SGBRG12_1X12:
	case MEDIA_BUS_FMT_SGRBG12_1X12:
	case MEDIA_BUS_FMT_SRGGB12_1X12:
	case MEDIA_BUS_FMT_Y12_1X12:
		return 12;
	case MEDIA_BUS_FMT_SBGGR14_1X14:
	case MEDIA_BUS_FMT_SGBRG14_1X14:
	case MEDIA_BUS_FMT_SGRBG14_1X14:
	case MEDIA_BUS_FMT_SRGGB14_1X14:
	case MEDIA_BUS_FMT_Y14_1X14:
		return 14;
	default:
		return 12;
	}
}

static u64 imx492_get_pixel_rate(u32 code)
{
	u64 pixel_rate = IMX492_LINK_FREQ * 2 * IMX492_NUM_DATA_LANES;

	do_div(pixel_rate, imx492_get_format_bpp(code));

	return pixel_rate;
}


/* Read registers up to 2 at a time */
static int imx492_read_reg(struct imx492 *imx492, u16 reg, u32 len, u32 *val)
{
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	struct i2c_msg msgs[2];
	u8 addr_buf[2] = { reg >> 8, reg & 0xff };
	u8 data_buf[4] = { 0, };
	int ret;

	if (len > 4)
		return -EINVAL;

	/* Write register address */
	msgs[0].addr = client->addr;
	msgs[0].flags = 0;
	msgs[0].len = ARRAY_SIZE(addr_buf);
	msgs[0].buf = addr_buf;

	/* Read data from register */
	msgs[1].addr = client->addr;
	msgs[1].flags = I2C_M_RD;
	msgs[1].len = len;
	msgs[1].buf = &data_buf[4 - len];

	ret = i2c_transfer(client->adapter, msgs, ARRAY_SIZE(msgs));
	if (ret != ARRAY_SIZE(msgs))
		return -EIO;

	*val = get_unaligned_be32(data_buf);

	return 0;
}

/* Write registers 1 byte at a time */
static int imx492_write_reg_1byte(struct imx492 *imx492, u16 reg, u8 val)
{
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	u8 buf[3];

	put_unaligned_be16(reg, buf);
	buf[2]  = val;
	if (i2c_master_send(client, buf, 3) != 3)
		return -EIO;

	return 0;
}

/* Write registers 2 byte at a time */
static int imx492_write_reg_2byte(struct imx492 *imx492, u16 reg, u16 val)
{
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	u8 buf[4];

	put_unaligned_be16(reg, buf);
	buf[2]  = val;
	buf[3]  = val>>8;
	if (i2c_master_send(client, buf, 4) != 4)
		return -EIO;

	return 0;
}

/* Write registers 3 byte at a time */
static int imx492_write_reg_3byte(struct imx492 *imx492, u16 reg, u32 val)
{
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	u8 buf[5];

	put_unaligned_be16(reg, buf);
	buf[2]  = val;
	buf[3]  = val>>8;
	buf[4]  = val>>16;
	if (i2c_master_send(client, buf, 5) != 5)
		return -EIO;

	return 0;
}

/* Write a list of 1 byte registers */
static int imx492_write_regs(struct imx492 *imx492,
			     const struct imx492_reg *regs, u32 len)
{
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	unsigned int i;
	int ret;

	for (i = 0; i < len; i++) {
		if (regs[i].address == 0xFFFE) {
			usleep_range(regs[i].val*1000,(regs[i].val+1)*1000);
		}
		else{
			ret = imx492_write_reg_1byte(imx492, regs[i].address, regs[i].val);
			if (ret) {
				dev_err_ratelimited(&client->dev,
						    "Failed to write reg 0x%4.4x. error = %d\n",
						    regs[i].address, ret);

				return ret;
			}
		}
	}

	return 0;
}

static u32 imx492_default_format_code(const struct imx492 *imx492, u32 code)
{
	u32 bpp = imx492_get_format_bpp(code);

	if (imx492->mono) {
		switch (bpp) {
		case 10: return MEDIA_BUS_FMT_Y10_1X10;
		case 14: return MEDIA_BUS_FMT_Y14_1X14;
		default: return MEDIA_BUS_FMT_Y12_1X12;
		}
	}

	switch (bpp) {
	case 10: return MEDIA_BUS_FMT_SRGGB10_1X10;
	case 14: return MEDIA_BUS_FMT_SRGGB14_1X14;
	default: return MEDIA_BUS_FMT_SRGGB12_1X12;
	}
}

static u32 imx492_get_format_code(struct imx492 *imx492, u32 code)
{
	unsigned int i;

	lockdep_assert_held(&imx492->mutex);

	if (imx492->mono) {
		for (i = 0; i < ARRAY_SIZE(mono_codes); i++)
			if (mono_codes[i] == code)
				return mono_codes[i];

		return imx492_default_format_code(imx492, code);
	}

	for (i = 0; i < ARRAY_SIZE(codes); i++)
		if (codes[i] == code)
			return codes[i];

	return imx492_default_format_code(imx492, code);
}

static void imx492_set_default_format(struct imx492 *imx492)
{
	/* Set default mode to max resolution. */
	imx492->mode = &supported_modes_12bit[0];
	imx492->fmt_code = imx492->mono ? MEDIA_BUS_FMT_Y12_1X12 :
					    MEDIA_BUS_FMT_SRGGB12_1X12;
}


static int imx492_open(struct v4l2_subdev *sd, struct v4l2_subdev_fh *fh)
{
	struct imx492 *imx492 = to_imx492(sd);
	struct v4l2_mbus_framefmt *try_fmt_img =
		imx492_subdev_state_get_format(sd, fh->state, IMAGE_PAD);
	struct v4l2_mbus_framefmt *try_fmt_meta =
		imx492_subdev_state_get_format(sd, fh->state, METADATA_PAD);
	struct v4l2_rect *try_crop;

	mutex_lock(&imx492->mutex);

	/* Initialize try_fmt for the image pad */
	try_fmt_img->width = supported_modes_12bit[0].width;
	try_fmt_img->height = supported_modes_12bit[0].height;
	try_fmt_img->code = imx492_get_format_code(imx492,
						   imx492->mono ?
						   MEDIA_BUS_FMT_Y12_1X12 :
						   MEDIA_BUS_FMT_SRGGB12_1X12);
	try_fmt_img->field = V4L2_FIELD_NONE;

	/* Initialize try_fmt for the embedded metadata pad */
	try_fmt_meta->width = IMX492_EMBEDDED_LINE_WIDTH;
	try_fmt_meta->height = IMX492_NUM_EMBEDDED_LINES;
	try_fmt_meta->code = MEDIA_BUS_FMT_SENSOR_DATA;
	try_fmt_meta->field = V4L2_FIELD_NONE;

	/* Initialize try_crop */
	try_crop = imx492_subdev_state_get_crop(sd, fh->state, IMAGE_PAD);
	try_crop->left = IMX492_PIXEL_ARRAY_LEFT;
	try_crop->top = IMX492_PIXEL_ARRAY_TOP;
	try_crop->width = IMX492_PIXEL_ARRAY_WIDTH;
	try_crop->height = IMX492_PIXEL_ARRAY_HEIGHT;

	mutex_unlock(&imx492->mutex);

	return 0;
}


static u64 calculate_v4l2_cid_exposure(u64 hmax, u64 vmax, u64 shr, u64 svr, u64 offset) {
    u64 numerator;
    numerator = (vmax * (svr + 1) - shr) * hmax + offset;

    do_div(numerator, hmax);
    numerator = clamp_t(uint32_t, numerator, 0, 0xFFFFFFFF);
    return numerator;
}

static void calculate_min_max_v4l2_cid_exposure(u64 hmax, u64 vmax, u64 min_shr, u64 svr, u64 offset, u64 *min_exposure, u64 *max_exposure) {
    u64 max_shr = (svr + 1) * vmax - 4;
    max_shr = min_t(uint64_t, max_shr, 0xFFFF);

    *min_exposure = calculate_v4l2_cid_exposure(hmax, vmax, max_shr, svr, offset);
    *max_exposure = calculate_v4l2_cid_exposure(hmax, vmax, min_shr, svr, offset);
}


/*
Integration Time [s] = [{VMAX × (SVR + 1) – (SHR)}
 × HMAX + offset] / (72 × 10^6)

Integration Time [s] = exposure * HMAX / (72 × 10^6)
*/

static uint32_t calculate_shr(uint32_t exposure, uint32_t hmax, uint64_t vmax, uint32_t svr, uint32_t offset) {
    uint64_t temp;
    uint32_t shr;

    temp = ((uint64_t)exposure * hmax - offset);
    do_div(temp, hmax);
    shr = (uint32_t)(vmax * (svr + 1) - temp);

    return shr;
}

static int imx492_set_test_pattern(struct imx492 *imx492, u32 pattern)
{
	int ret;

	if (pattern == 0) {
		ret = imx492_write_reg_1byte(imx492, IMX492_REG_TEST_PATTERN_CTRL,
					     0x00);
		if (ret)
			return ret;

		return imx492_write_reg_1byte(imx492, IMX492_REG_TEST_PATTERN_SEL,
					      0x00);
	}

	if (pattern > ARRAY_SIZE(imx492_test_pattern_menu) - 1)
		return -EINVAL;

	ret = imx492_write_reg_1byte(imx492, IMX492_REG_TEST_PATTERN_CTRL,
				     IMX492_TEST_PATTERN_ENABLE_MIPI);
	if (ret)
		return ret;

	return imx492_write_reg_1byte(imx492, IMX492_REG_TEST_PATTERN_SEL,
				      imx492_test_pattern_sel[pattern - 1]);
}

static int imx492_set_ctrl(struct v4l2_ctrl *ctrl)
{
	struct imx492 *imx492 =
		container_of(ctrl->handler, struct imx492, ctrl_handler);
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	const struct imx492_mode *mode = imx492->mode;
    u64 shr, vblk, tmp;
	int ret = 0;
        u64 pixel_rate,hmax;
	/*
	 * The VBLANK control may change the limits of usable exposure, so check
	 * and adjust if necessary.
	 */
		if (ctrl->id == V4L2_CID_VBLANK){
			/* Honour the VBLANK limits when setting exposure. */
			u64 current_exposure, max_exposure, min_exposure, vmax;

        vmax = ((u64)mode->height + ctrl->val);
        do_div(vmax, mode->VMAX_scale);

			imx492 -> VMAX = vmax;
			
			calculate_min_max_v4l2_cid_exposure(imx492 -> HMAX, imx492 -> VMAX, (u64)mode->min_SHR, 0, mode->integration_offset, &min_exposure, &max_exposure);
			current_exposure = imx492->exposure->val;
			current_exposure = clamp_t(uint32_t, current_exposure, min_exposure, max_exposure);

		DEBUG_PRINTK("exposure_max:%lld, exposure_min:%lld, current_exposure:%lld\n",max_exposure, min_exposure, current_exposure);
		DEBUG_PRINTK("\tVMAX:%d, HMAX:%d\n",imx492->VMAX, imx492->HMAX);
		__v4l2_ctrl_modify_range(imx492->exposure, min_exposure,max_exposure, 1,current_exposure);
	}

	/*
	 * Applying V4L2 control value only happens
	 * when power is up for streaming
	 */
	if (pm_runtime_get_if_in_use(&client->dev) == 0)
		return 0;

	
	switch (ctrl->id) {
	case V4L2_CID_EXPOSURE:
		{
		DEBUG_PRINTK("V4L2_CID_EXPOSURE : %d\n",ctrl->val);
		DEBUG_PRINTK("\tvblank:%d, hblank:%d\n",imx492->vblank->val, imx492->hblank->val);
		DEBUG_PRINTK("\tVMAX:%d, HMAX:%d\n",imx492->VMAX, imx492->HMAX);
		shr = calculate_shr(ctrl->val, imx492->HMAX, imx492->VMAX, 0, mode->integration_offset);
		DEBUG_PRINTK("\tSHR:%lld\n",shr);
		ret = imx492_write_reg_2byte(imx492, IMX492_REG_SHR, shr);
		}
		break;
	case V4L2_CID_ANALOGUE_GAIN:
		DEBUG_PRINTK("V4L2_CID_ANALOGUE_GAIN : %d\n",ctrl->val);
		ret = imx492_write_reg_2byte(imx492, IMX492_REG_ANALOG_GAIN, ctrl->val);
		break;
	case V4L2_CID_BRIGHTNESS:
		DEBUG_PRINTK("V4L2_CID_BRIGHTNESS : %d\n", ctrl->val);
		ret = imx492_write_reg_1byte(imx492, IMX492_REG_BLKLEVEL,
					     min_t(u32, ctrl->val, IMX492_BLKLEVEL_MAX));
		break;
	case V4L2_CID_VBLANK:
		{
		DEBUG_PRINTK("V4L2_CID_VBLANK : %d\n",ctrl->val);
        tmp = ((u64)mode->height + ctrl->val);
        do_div(tmp, mode->VMAX_scale);
		imx492 -> VMAX = tmp;
		DEBUG_PRINTK("\tVMAX : %d\n",imx492 -> VMAX);
		ret = imx492_write_reg_3byte(imx492, IMX492_REG_VMAX, imx492 -> VMAX);
        vblk = imx492 -> VMAX  - mode-> min_VMAX;
        DEBUG_PRINTK("\tvblk : %lld\n",vblk);
        ret = imx492_write_reg_2byte(imx492, IMX492_REG_PSSLVS1, vblk);
        ret = imx492_write_reg_2byte(imx492, IMX492_REG_PSSLVS2, vblk);
        ret = imx492_write_reg_2byte(imx492, IMX492_REG_PSSLVS3, vblk);
        if(vblk <= 5){
            ret = imx492_write_reg_2byte(imx492, IMX492_REG_PSSLVS4, 0);
        }
        else{
            ret = imx492_write_reg_2byte(imx492, IMX492_REG_PSSLVS4, vblk - 5);
        }
        ret = imx492_write_reg_2byte(imx492, IMX492_REG_PSSLVS0, vblk);
		}
		break;
	case V4L2_CID_HBLANK:
		{
		DEBUG_PRINTK("V4L2_CID_HBLANK : %d\n",ctrl->val);
		//int hmax = (IMX492_NATIVE_WIDTH + ctrl->val) * 72000000; / IMX492_PIXEL_RATE;
		pixel_rate = (u64)mode->width * 72000000 * mode->VMAX_scale;
		do_div(pixel_rate,mode->min_HMAX);
		hmax = (u64)(mode->width + ctrl->val) * 72000000 * mode->VMAX_scale;
		do_div(hmax,pixel_rate);
		imx492 -> HMAX = hmax;
		DEBUG_PRINTK("\tHMAX : %d\n",imx492 -> HMAX);
		ret = imx492_write_reg_2byte(imx492, IMX492_REG_HMAX, hmax);
		}
		break;
	case V4L2_CID_TEST_PATTERN:
		DEBUG_PRINTK("V4L2_CID_TEST_PATTERN : %d\n", ctrl->val);
		ret = imx492_set_test_pattern(imx492, ctrl->val);
		break;
	default:
		dev_err(&client->dev,
			 "ctrl(id:0x%x,val:0x%x) is not handled\n",
			 ctrl->id, ctrl->val);
		ret = -EINVAL;
		break;
	}

	pm_runtime_put_autosuspend(&client->dev);

	return ret;
}

static const struct v4l2_ctrl_ops imx492_ctrl_ops = {
	.s_ctrl = imx492_set_ctrl,
};

static int imx492_enum_mbus_code(struct v4l2_subdev *sd,
				 struct v4l2_subdev_state *sd_state,
				 struct v4l2_subdev_mbus_code_enum *code)
{
	struct imx492 *imx492 = to_imx492(sd);

	if (code->pad >= NUM_PADS)
		return -EINVAL;

	if (code->pad == IMAGE_PAD) {
		if (imx492->mono) {
			if (code->index >= ARRAY_SIZE(mono_codes))
				return -EINVAL;

			code->code = mono_codes[code->index];
		} else {
			if (code->index >= (ARRAY_SIZE(codes) / 4))
				return -EINVAL;

			code->code = imx492_get_format_code(imx492,
							    codes[code->index * 4]);
		}
	} else {
		if (code->index > 0)
			return -EINVAL;

		code->code = MEDIA_BUS_FMT_SENSOR_DATA;
	}

	return 0;
}

static int imx492_enum_frame_size(struct v4l2_subdev *sd,
				  struct v4l2_subdev_state *sd_state,
				  struct v4l2_subdev_frame_size_enum *fse)
{
	struct imx492 *imx492 = to_imx492(sd);

	if (fse->pad >= NUM_PADS)
		return -EINVAL;

	if (fse->pad == IMAGE_PAD) {
		const struct imx492_mode *mode_list;
		unsigned int num_modes;

		get_mode_table(fse->code, &mode_list, &num_modes);

		if (fse->index >= num_modes)
			return -EINVAL;

		if (fse->code != imx492_get_format_code(imx492, fse->code))
			return -EINVAL;

		fse->min_width = mode_list[fse->index].width;
		fse->max_width = fse->min_width;
		fse->min_height = mode_list[fse->index].height;
		fse->max_height = fse->min_height;
	} else {
		if (fse->code != MEDIA_BUS_FMT_SENSOR_DATA || fse->index > 0)
			return -EINVAL;

		fse->min_width = IMX492_EMBEDDED_LINE_WIDTH;
		fse->max_width = fse->min_width;
		fse->min_height = IMX492_NUM_EMBEDDED_LINES;
		fse->max_height = fse->min_height;
	}

	return 0;
}

static void imx492_reset_colorspace(struct v4l2_mbus_framefmt *fmt)
{
	fmt->colorspace = V4L2_COLORSPACE_RAW;
	fmt->ycbcr_enc = V4L2_MAP_YCBCR_ENC_DEFAULT(fmt->colorspace);
	fmt->quantization = V4L2_MAP_QUANTIZATION_DEFAULT(true,
							  fmt->colorspace,
							  fmt->ycbcr_enc);
	fmt->xfer_func = V4L2_MAP_XFER_FUNC_DEFAULT(fmt->colorspace);
}

static void imx492_update_image_pad_format(struct imx492 *imx492,
					   const struct imx492_mode *mode,
					   struct v4l2_subdev_format *fmt)
{
	fmt->format.width = mode->width;
	fmt->format.height = mode->height;
	fmt->format.field = V4L2_FIELD_NONE;
	imx492_reset_colorspace(&fmt->format);
}

static void imx492_update_metadata_pad_format(struct v4l2_subdev_format *fmt)
{
	fmt->format.width = IMX492_EMBEDDED_LINE_WIDTH;
	fmt->format.height = IMX492_NUM_EMBEDDED_LINES;
	fmt->format.code = MEDIA_BUS_FMT_SENSOR_DATA;
	fmt->format.field = V4L2_FIELD_NONE;
}

static int imx492_get_pad_format(struct v4l2_subdev *sd,
				 struct v4l2_subdev_state *sd_state,
				 struct v4l2_subdev_format *fmt)
{
	struct imx492 *imx492 = to_imx492(sd);

	if (fmt->pad >= NUM_PADS)
		return -EINVAL;

	mutex_lock(&imx492->mutex);

	if (fmt->which == V4L2_SUBDEV_FORMAT_TRY) {
		struct v4l2_mbus_framefmt *try_fmt =
			imx492_subdev_state_get_format(&imx492->sd, sd_state,
						      fmt->pad);
		/* update the code which could change due to vflip or hflip: */
		try_fmt->code = fmt->pad == IMAGE_PAD ?
				imx492_get_format_code(imx492, try_fmt->code) :
				MEDIA_BUS_FMT_SENSOR_DATA;
		fmt->format = *try_fmt;
	} else {
		if (fmt->pad == IMAGE_PAD) {
			imx492_update_image_pad_format(imx492, imx492->mode,
						       fmt);
			fmt->format.code =
			       imx492_get_format_code(imx492, imx492->fmt_code);
		} else {
			imx492_update_metadata_pad_format(fmt);
		}
	}

	mutex_unlock(&imx492->mutex);
	return 0;
}

/* TODO */
static void imx492_set_framing_limits(struct imx492 *imx492)
{
	const struct imx492_mode *mode = imx492->mode;
	u64 def_hblank;
	u64 pixel_rate = imx492_get_pixel_rate(imx492->fmt_code);


	imx492->VMAX = mode->default_VMAX;
	imx492->HMAX = mode->default_HMAX;

	DEBUG_PRINTK("Pixel Rate : %lld\n",pixel_rate);


	//int def_hblank = mode->default_HMAX * IMX492_PIXEL_RATE / 72000000 - IMX492_NATIVE_WIDTH;
	def_hblank = mode->default_HMAX * pixel_rate;
	do_div(def_hblank,72000000);
	def_hblank = def_hblank - mode->width;
	__v4l2_ctrl_modify_range(imx492->hblank, 0,
				 IMX492_HMAX_MAX, 1, def_hblank);


	__v4l2_ctrl_s_ctrl(imx492->hblank, def_hblank);



	/* Update limits and set FPS to default */
	__v4l2_ctrl_modify_range(imx492->vblank, mode->min_VMAX*mode->VMAX_scale - mode->height,
				 IMX492_VMAX_MAX*mode->VMAX_scale - mode->height,
				 1, mode->default_VMAX*mode->VMAX_scale - mode->height);
	__v4l2_ctrl_s_ctrl(imx492->vblank, mode->default_VMAX*mode->VMAX_scale - mode->height);

	/* Setting this will adjust the exposure limits as well. */

	__v4l2_ctrl_s_ctrl(imx492->link_freq, 0);
	__v4l2_ctrl_modify_range(imx492->pixel_rate, pixel_rate, pixel_rate, 1, pixel_rate);

	DEBUG_PRINTK("Setting default HBLANK : %lld, VBLANK : %lld with PixelRate: %lld\n",def_hblank,mode->default_VMAX*mode->VMAX_scale - mode->height, pixel_rate);

}
/* TODO */
static int imx492_set_pad_format(struct v4l2_subdev *sd,
				 struct v4l2_subdev_state *sd_state,
				 struct v4l2_subdev_format *fmt)
{
	struct v4l2_mbus_framefmt *framefmt;
	const struct imx492_mode *mode;
	struct imx492 *imx492 = to_imx492(sd);

	if (fmt->pad >= NUM_PADS)
		return -EINVAL;

	mutex_lock(&imx492->mutex);

	if (fmt->pad == IMAGE_PAD) {
		const struct imx492_mode *mode_list;
		unsigned int num_modes;

		/* Bayer order varies with flips */
		fmt->format.code = imx492_get_format_code(imx492,
							  fmt->format.code);

		get_mode_table(fmt->format.code, &mode_list, &num_modes);

		mode = v4l2_find_nearest_size(mode_list,
					      num_modes,
					      width, height,
					      fmt->format.width,
					      fmt->format.height);
		imx492_update_image_pad_format(imx492, mode, fmt);
		if (fmt->which == V4L2_SUBDEV_FORMAT_TRY) {
			framefmt = imx492_subdev_state_get_format(sd, sd_state,
								fmt->pad);
			*framefmt = fmt->format;
			*imx492_subdev_state_get_crop(&imx492->sd, sd_state,
						      fmt->pad) = mode->crop;
		} else if (imx492->mode != mode) {
			imx492->mode = mode;
			imx492->fmt_code = fmt->format.code;
			imx492_set_framing_limits(imx492);
		}
	} else {
		if (fmt->which == V4L2_SUBDEV_FORMAT_TRY) {
			framefmt = imx492_subdev_state_get_format(sd, sd_state,
								fmt->pad);
			*framefmt = fmt->format;
		} else {
			/* Only one embedded data mode is supported */
			imx492_update_metadata_pad_format(fmt);
		}
	}

	mutex_unlock(&imx492->mutex);

	return 0;
}
/* TODO */
static const struct v4l2_rect *
__imx492_get_pad_crop(struct imx492 *imx492,
		      struct v4l2_subdev_state *sd_state,
		      unsigned int pad, enum v4l2_subdev_format_whence which)
{
	switch (which) {
	case V4L2_SUBDEV_FORMAT_TRY:
		return imx492_subdev_state_get_crop(&imx492->sd, sd_state, pad);
	case V4L2_SUBDEV_FORMAT_ACTIVE:
		return &imx492->mode->crop;
	}

	return NULL;
}

/* Start streaming */
static int imx492_start_streaming(struct imx492 *imx492)
{
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	const struct imx492_mode *mode = imx492->mode;
	const struct IMX492_reg_list *reg_list;
	int ret;

	ret = imx492_write_regs(imx492, imx492_startup_pre_regs,
				ARRAY_SIZE(imx492_startup_pre_regs));
	if (ret) {
		dev_err(&client->dev,
			"%s failed to run startup pre-sequence\n",
			__func__);
		return ret;
	}

	reg_list = &imx492->mode->reg_list;
	ret = imx492_write_regs(imx492, reg_list->regs, reg_list->num_of_regs);
	if (ret) {
		dev_err(&client->dev, "%s failed to set mode\n", __func__);
		return ret;
	}

	/*
	 * Optional HTRIMMING / VWINPOS path for modes that want a
	 * hardware crop window independent of the vendor-supplied
	 * reg_list.  Enabled when mode->htrimming_end > 0.
	 *
	 * Override the CSI-2 output-width register (0x3ED0) only for
	 * unbinned HTRIMMING modes (e.g. the square crop).  The binned
	 * mode's register table (from IMX294) omits all 0x3Exx
	 * registers, letting the sensor auto-determine its CSI-2 line
	 * width from the HTRIMMING window — matching IMX294 behavior.
	 */
	if (mode->htrimming_end > 0 && !mode->is_binned) {
		ret = imx492_write_reg_2byte(imx492, 0x3ED0, mode->width);
		if (ret)
			return ret;
	}

	/*
	 * Binned modes need HOPBOUT_EN=0 (horizontal output binning
	 * disabled — the 2×2 binning is done in the pixel array, not
	 * the output formatter).  Unbinned modes need HOPBOUT_EN=1.
	 */
	ret = imx492_write_reg_1byte(imx492, IMX492_REG_HOPBOUT_EN,
				     mode->is_binned ? 0x00 : 0x01);
	if (ret)
		return ret;

	ret = imx492_write_reg_1byte(imx492, IMX492_REG_HTRIMMING_EN,
				     mode->htrimming_end > 0 ? 0x01 : 0x00);
	if (ret)
		return ret;

	ret = imx492_write_reg_2byte(imx492, IMX492_REG_HTRIMMING_START,
				     mode->htrimming_start);
	if (ret)
		return ret;

	ret = imx492_write_reg_2byte(imx492, IMX492_REG_HTRIMMING_END,
				     mode->htrimming_end);
	if (ret)
		return ret;

	/*
	 * Binned modes: HCOUNT1/2 = 0x04B0 (1200), matching the IMX294
	 * driver's common_regs.  Unbinned modes: HCOUNT1/2 = 0.
	 */
	ret = imx492_write_reg_2byte(imx492, IMX492_REG_HCOUNT1,
				     mode->is_binned ? 0x04B0 : 0);
	if (ret)
		return ret;

	ret = imx492_write_reg_2byte(imx492, IMX492_REG_HCOUNT2,
				     mode->is_binned ? 0x04B0 : 0);
	if (ret)
		return ret;

	ret = imx492_write_reg_2byte(imx492, IMX492_REG_VWINPOS,
				     mode->vwinpos);
	if (ret)
		return ret;

	ret = imx492_write_reg_1byte(imx492, IMX492_REG_OPB_SIZE_V,
				     mode->opb_size_v);
	if (ret)
		return ret;

	ret = imx492_write_reg_2byte(imx492, IMX492_REG_WRITE_VSIZE,
				     mode->write_vsize);
	if (ret)
		return ret;

	ret = imx492_write_reg_2byte(imx492, IMX492_REG_Y_OUT_SIZE,
				     mode->y_out_size);
	if (ret)
		return ret;

	ret = imx492_write_regs(imx492, imx492_startup_post_regs,
					ARRAY_SIZE(imx492_startup_post_regs));
	if (ret) {
		dev_err(&client->dev,
			"%s failed to run startup post-sequence\n",
			__func__);
		return ret;
	}

	ret = imx492_set_test_pattern(imx492, imx492->test_pattern->val);
	if (ret)
		return ret;

	ret = __v4l2_ctrl_handler_setup(imx492->sd.ctrl_handler);
	if (ret)
		return ret;

	usleep_range(IMX492_STREAM_DELAY_US,
		     IMX492_STREAM_DELAY_US + IMX492_STREAM_DELAY_RANGE_US);

	return 0;
}


/* Stop streaming */
static void imx492_stop_streaming(struct imx492 *imx492)
{
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	int ret;

	/* set stream off register */
	ret = imx492_write_reg_1byte(imx492, IMX492_REG_MODE_SELECT, IMX492_MODE_STANDBY);
	if (ret)
		dev_err(&client->dev, "%s failed to set stream\n", __func__);
}

static int imx492_set_stream(struct v4l2_subdev *sd, int enable)
{
	struct imx492 *imx492 = to_imx492(sd);
	struct i2c_client *client = v4l2_get_subdevdata(sd);
	int ret = 0;

	mutex_lock(&imx492->mutex);
	if (imx492->streaming == enable) {
		mutex_unlock(&imx492->mutex);
		return 0;
	}

		if (enable) {
			ret = pm_runtime_get_sync(&client->dev);
			if (ret < 0) {
				pm_runtime_put_noidle(&client->dev);
			goto err_unlock;
		}

		/*
		 * Apply default & customized values
		 * and then start streaming.
		 */
			ret = imx492_start_streaming(imx492);
			if (ret)
				goto err_rpm_put;
		} else {
			imx492_stop_streaming(imx492);
			pm_runtime_mark_last_busy(&client->dev);
			pm_runtime_put_autosuspend(&client->dev);
		}

	imx492->streaming = enable;
	mutex_unlock(&imx492->mutex);

	return ret;

err_rpm_put:
	pm_runtime_mark_last_busy(&client->dev);
	pm_runtime_put_autosuspend(&client->dev);
err_unlock:
	mutex_unlock(&imx492->mutex);

	return ret;
}

/* Power/clock management functions */
static int imx492_power_on(struct device *dev)
{
	struct i2c_client *client = to_i2c_client(dev);
	struct v4l2_subdev *sd = i2c_get_clientdata(client);
	struct imx492 *imx492 = to_imx492(sd);
	int ret;

	ret = regulator_bulk_enable(imx492_NUM_SUPPLIES,
				    imx492->supplies);
	if (ret) {
		dev_err(&client->dev, "%s: failed to enable regulators\n",
			__func__);
		return ret;
	}

	ret = clk_prepare_enable(imx492->xclk);
	if (ret) {
		dev_err(&client->dev, "%s: failed to enable clock\n",
			__func__);
		goto reg_off;
	}

	gpiod_set_value_cansleep(imx492->reset_gpio, 1);
	usleep_range(imx492_XCLR_MIN_DELAY_US,
		     imx492_XCLR_MIN_DELAY_US + imx492_XCLR_DELAY_RANGE_US);

	return 0;

reg_off:
	regulator_bulk_disable(imx492_NUM_SUPPLIES, imx492->supplies);
	return ret;
}

static int imx492_power_off(struct device *dev)
{
	struct i2c_client *client = to_i2c_client(dev);
	struct v4l2_subdev *sd = i2c_get_clientdata(client);
	struct imx492 *imx492 = to_imx492(sd);

	gpiod_set_value_cansleep(imx492->reset_gpio, 0);
	regulator_bulk_disable(imx492_NUM_SUPPLIES, imx492->supplies);
	clk_disable_unprepare(imx492->xclk);

	return 0;
}

static int __maybe_unused imx492_suspend(struct device *dev)
{
	struct i2c_client *client = to_i2c_client(dev);
	struct v4l2_subdev *sd = i2c_get_clientdata(client);
	struct imx492 *imx492 = to_imx492(sd);

	if (imx492->streaming)
		imx492_stop_streaming(imx492);

	return 0;
}

static int __maybe_unused imx492_resume(struct device *dev)
{
	struct i2c_client *client = to_i2c_client(dev);
	struct v4l2_subdev *sd = i2c_get_clientdata(client);
	struct imx492 *imx492 = to_imx492(sd);
	int ret;

	if (imx492->streaming) {
		ret = imx492_start_streaming(imx492);
		if (ret)
			goto error;
	}

	return 0;

error:
	imx492_stop_streaming(imx492);
	imx492->streaming = 0;
	return ret;
}

static int imx492_get_regulators(struct imx492 *imx492)
{
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	unsigned int i;

	for (i = 0; i < imx492_NUM_SUPPLIES; i++)
		imx492->supplies[i].supply = imx492_supply_name[i];

	return devm_regulator_bulk_get(&client->dev,
				       imx492_NUM_SUPPLIES,
				       imx492->supplies);
}

/* Verify chip ID */
static int imx492_identify_module(struct imx492 *imx492, u32 expected_id)
{
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	int ret;
	u32 val;

	ret = imx492_read_reg(imx492, IMX492_REG_CHIP_ID,
			      1, &val);
	if (ret) {
		dev_err(&client->dev, "failed to read chip id %x, with error %d\n",
			expected_id, ret);
		return ret;
	}

	dev_info(&client->dev, "Device found\n");

	return 0;
}

static int imx492_get_selection(struct v4l2_subdev *sd,
				struct v4l2_subdev_state *sd_state,
				struct v4l2_subdev_selection *sel)
{
	switch (sel->target) {
	case V4L2_SEL_TGT_CROP: {
		struct imx492 *imx492 = to_imx492(sd);

		mutex_lock(&imx492->mutex);
		sel->r = *__imx492_get_pad_crop(imx492, sd_state, sel->pad,
						sel->which);
		mutex_unlock(&imx492->mutex);

		return 0;
	}

	case V4L2_SEL_TGT_NATIVE_SIZE:
		sel->r.left = 0;
		sel->r.top = 0;
		sel->r.width = IMX492_NATIVE_WIDTH;
		sel->r.height = IMX492_NATIVE_HEIGHT;

		return 0;

	case V4L2_SEL_TGT_CROP_DEFAULT:
	case V4L2_SEL_TGT_CROP_BOUNDS:
		sel->r.left = IMX492_PIXEL_ARRAY_LEFT;
		sel->r.top = IMX492_PIXEL_ARRAY_TOP;
		sel->r.width = IMX492_PIXEL_ARRAY_WIDTH;
		sel->r.height = IMX492_PIXEL_ARRAY_HEIGHT;

		return 0;
	}

	return -EINVAL;
}


static const struct v4l2_subdev_core_ops imx492_core_ops = {
	.subscribe_event = v4l2_ctrl_subdev_subscribe_event,
	.unsubscribe_event = v4l2_event_subdev_unsubscribe,
};

static const struct v4l2_subdev_video_ops imx492_video_ops = {
	.s_stream = imx492_set_stream,
};

static const struct v4l2_subdev_pad_ops imx492_pad_ops = {
	.enum_mbus_code = imx492_enum_mbus_code,
	.get_fmt = imx492_get_pad_format,
	.set_fmt = imx492_set_pad_format,
	.get_selection = imx492_get_selection,
	.enum_frame_size = imx492_enum_frame_size,
};

static const struct v4l2_subdev_ops imx492_subdev_ops = {
	.core = &imx492_core_ops,
	.video = &imx492_video_ops,
	.pad = &imx492_pad_ops,
};

static const struct v4l2_subdev_internal_ops imx492_internal_ops = {
	.open = imx492_open,
};




/* Initialize control handlers */
static int imx492_init_controls(struct imx492 *imx492)
{
	struct v4l2_ctrl_handler *ctrl_hdlr;
	struct i2c_client *client = v4l2_get_subdevdata(&imx492->sd);
	struct v4l2_fwnode_device_properties props;
	int ret;

	ctrl_hdlr = &imx492->ctrl_handler;
	ret = v4l2_ctrl_handler_init(ctrl_hdlr, 20);
	if (ret)
		return ret;

	mutex_init(&imx492->mutex);
	ctrl_hdlr->lock = &imx492->mutex;



	/*
	 * Create the controls here, but mode specific limits are setup
	 * in the imx492_set_framing_limits() call below.
	 */
	imx492->link_freq = v4l2_ctrl_new_int_menu(ctrl_hdlr, &imx492_ctrl_ops,
						   V4L2_CID_LINK_FREQ,
						   ARRAY_SIZE(imx492_link_freq_menu) - 1,
						   0, imx492_link_freq_menu);
	/* By default, PIXEL_RATE is read only */
	imx492->pixel_rate = v4l2_ctrl_new_std(ctrl_hdlr, &imx492_ctrl_ops,
					       V4L2_CID_PIXEL_RATE,
					       0xffff,
					       0xffff, 1,
					       0xffff);
	imx492->vblank = v4l2_ctrl_new_std(ctrl_hdlr, &imx492_ctrl_ops,
					   V4L2_CID_VBLANK, 0, 0xfffff, 1, 0);
	imx492->hblank = v4l2_ctrl_new_std(ctrl_hdlr, &imx492_ctrl_ops,
					   V4L2_CID_HBLANK, 0, 0xffff, 1, 0);
	imx492->blacklevel = v4l2_ctrl_new_std(ctrl_hdlr, &imx492_ctrl_ops,
					       V4L2_CID_BRIGHTNESS,
					       IMX492_BLKLEVEL_MIN,
					       IMX492_BLKLEVEL_MAX,
					       1, IMX492_BLKLEVEL_DEFAULT);
	imx492->test_pattern = v4l2_ctrl_new_std_menu_items(ctrl_hdlr,
						&imx492_ctrl_ops,
						V4L2_CID_TEST_PATTERN,
						ARRAY_SIZE(imx492_test_pattern_menu) - 1,
						0, 0, imx492_test_pattern_menu);

	imx492->exposure = v4l2_ctrl_new_std(ctrl_hdlr, &imx492_ctrl_ops,
					     V4L2_CID_EXPOSURE,
					     IMX492_EXPOSURE_MIN,
					     IMX492_EXPOSURE_MAX,
					     IMX492_EXPOSURE_STEP,
					     IMX492_EXPOSURE_DEFAULT);

	v4l2_ctrl_new_std(ctrl_hdlr, &imx492_ctrl_ops, V4L2_CID_ANALOGUE_GAIN,
			  IMX492_ANA_GAIN_MIN, IMX492_ANA_GAIN_MAX,
			  IMX492_ANA_GAIN_STEP, IMX492_ANA_GAIN_DEFAULT);

	if (imx492->link_freq)
		imx492->link_freq->flags |= V4L2_CTRL_FLAG_READ_ONLY;
	if (imx492->pixel_rate)
		imx492->pixel_rate->flags |= V4L2_CTRL_FLAG_READ_ONLY;

	if (ctrl_hdlr->error) {
		ret = ctrl_hdlr->error;
		dev_err(&client->dev, "%s control init failed (%d)\n",
			__func__, ret);
		goto error;
	}

	ret = v4l2_fwnode_device_parse(&client->dev, &props);
	if (ret)
		goto error;

	ret = v4l2_ctrl_new_fwnode_properties(ctrl_hdlr, &imx492_ctrl_ops,
					      &props);
	if (ret)
		goto error;

	imx492->sd.ctrl_handler = ctrl_hdlr;

	/* Setup exposure and frame/line length limits. */
	imx492_set_framing_limits(imx492);

	return 0;

error:
	v4l2_ctrl_handler_free(ctrl_hdlr);
	mutex_destroy(&imx492->mutex);

	return ret;
}

static void imx492_free_controls(struct imx492 *imx492)
{
	v4l2_ctrl_handler_free(imx492->sd.ctrl_handler);
	mutex_destroy(&imx492->mutex);
}


static const struct imx492_compatible_data imx492_compatible = {
	.chip_id = IMX492_CHIP_ID,
	.mono = false,
	.extra_regs = {
		.num_of_regs = 0,
		.regs = NULL
	}
};

static const struct of_device_id imx492_dt_ids[] = {
	{ .compatible = "sony,imx492", .data = &imx492_compatible },
	{ /* sentinel */ }
};

static int imx492_probe(struct i2c_client *client)
{
	struct device *dev = &client->dev;
	struct imx492 *imx492;
	const struct of_device_id *match;
	int ret;

	imx492 = devm_kzalloc(&client->dev, sizeof(*imx492), GFP_KERNEL);
	if (!imx492)
		return -ENOMEM;

	v4l2_i2c_subdev_init(&imx492->sd, client, &imx492_subdev_ops);

	match = of_match_device(imx492_dt_ids, dev);
	if (!match)
		return -ENODEV;
	imx492->compatible_data =
		(const struct imx492_compatible_data *)match->data;
	imx492->mono = imx492->compatible_data->mono ||
		       of_property_read_bool(dev->of_node, "mono-mode");

	if (imx492->mono)
		dev_info(dev, "mono mode enabled; expose Y10/Y12 formats\n");

	/* Get system clock (xclk) */
	imx492->xclk = devm_clk_get(dev, NULL);
	if (IS_ERR(imx492->xclk)) {
		dev_err(dev, "failed to get xclk\n");
		return PTR_ERR(imx492->xclk);
	}

	imx492->xclk_freq = clk_get_rate(imx492->xclk);
	if (imx492->xclk_freq != IMX492_XCLK_FREQ) {
		dev_err(dev, "xclk frequency not supported: %d Hz\n",
			imx492->xclk_freq);
		return -EINVAL;
	}

	ret = imx492_get_regulators(imx492);
	if (ret) {
		dev_err(dev, "failed to get regulators\n");
		return ret;
	}

	/* Request optional enable pin */
	imx492->reset_gpio = devm_gpiod_get_optional(dev, "reset",
						     GPIOD_OUT_HIGH);
	
	/*
	 * The sensor must be powered for imx492_identify_module()
	 * to be able to read the CHIP_ID register
	 */
	ret = imx492_power_on(dev);
	if (ret)
		return ret;

	ret = imx492_identify_module(imx492, imx492->compatible_data->chip_id);
	if (ret)
		goto error_power_off;

	/* Initialize default format */
	imx492_set_default_format(imx492);

	/* Enable runtime PM and let autosuspend turn the device off when idle. */
	pm_runtime_set_active(dev);
	pm_runtime_get_noresume(dev);
	pm_runtime_enable(dev);
	pm_runtime_set_autosuspend_delay(dev, IMX492_AUTOSUSPEND_DELAY_MS);
	pm_runtime_use_autosuspend(dev);

	/* This needs the pm runtime to be registered. */
	ret = imx492_init_controls(imx492);
	if (ret)
		goto error_power_off;

	/* Initialize subdev */
	imx492->sd.internal_ops = &imx492_internal_ops;
	imx492->sd.flags |= V4L2_SUBDEV_FL_HAS_DEVNODE |
			    V4L2_SUBDEV_FL_HAS_EVENTS;
	imx492->sd.entity.function = MEDIA_ENT_F_CAM_SENSOR;

	/* Initialize source pads */
	imx492->pad[IMAGE_PAD].flags = MEDIA_PAD_FL_SOURCE;
	imx492->pad[METADATA_PAD].flags = MEDIA_PAD_FL_SOURCE;

	ret = media_entity_pads_init(&imx492->sd.entity, NUM_PADS, imx492->pad);
	if (ret) {
		dev_err(dev, "failed to init entity pads: %d\n", ret);
		goto error_handler_free;
	}

	ret = v4l2_async_register_subdev_sensor(&imx492->sd);
	if (ret < 0) {
		dev_err(dev, "failed to register sensor sub-device: %d\n", ret);
		goto error_media_entity;
	}

	pm_runtime_mark_last_busy(dev);
	pm_runtime_put_autosuspend(dev);

	return 0;

error_media_entity:
	media_entity_cleanup(&imx492->sd.entity);

error_handler_free:
	imx492_free_controls(imx492);

error_power_off:
	pm_runtime_disable(&client->dev);
	pm_runtime_set_suspended(&client->dev);
	imx492_power_off(&client->dev);

	return ret;
}

static void imx492_remove(struct i2c_client *client)
{
	struct v4l2_subdev *sd = i2c_get_clientdata(client);
	struct imx492 *imx492 = to_imx492(sd);

	v4l2_async_unregister_subdev(sd);
	media_entity_cleanup(&sd->entity);
	imx492_free_controls(imx492);

	pm_runtime_disable(&client->dev);
	if (!pm_runtime_status_suspended(&client->dev))
		imx492_power_off(&client->dev);
	pm_runtime_set_suspended(&client->dev);

}

MODULE_DEVICE_TABLE(of, imx492_dt_ids);

static const struct dev_pm_ops imx492_pm_ops = {
	SET_SYSTEM_SLEEP_PM_OPS(imx492_suspend, imx492_resume)
	SET_RUNTIME_PM_OPS(imx492_power_off, imx492_power_on, NULL)
};

static struct i2c_driver imx492_i2c_driver = {
	.driver = {
		.name = "imx492",
		.of_match_table	= imx492_dt_ids,
		.pm = &imx492_pm_ops,
	},
	.probe = imx492_probe,
	.remove = imx492_remove,
};

module_i2c_driver(imx492_i2c_driver);

MODULE_AUTHOR("Will Whang <will@willwhang.com>");
MODULE_DESCRIPTION("Sony imx492 sensor driver");
MODULE_LICENSE("GPL v2");
