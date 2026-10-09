"""
There's probably a better way to do this but i cba to find out.

Bruteforcing:
- Sampled around 33 frames from varying degrees of darkness and brightness.
- 10k samples per frame (seed 2406755786128298521)
- 80k samples per scenes (seed 8654491321826895238)
- Grid Size: 17, 33, 65
- Identity Strength: 0.0001, 0.001, 0.01
- Smoothness: 0.01, 0.1, 1.0
- Iterations: 500-2000
- Stop Tolerance: 1e-5 to 1e-7

LUT then get compared to BD references via Butteraugli + SSIM2, finally
it selected five possible best LUTS:
1. g17/a-flex/b-balanced
2. g17/b-balanced/b-balanced
3. g33/a-flex/c-smooth (best "middleground" SSIM2 with good enough butteraugli)
4. g65/b-balanced/c-smooth (highest SSIM2 score and least worse-frame, butteraugli is worse compared to g17/g33)
5. g65/a-flex/c-smooth (more or less the same as above, with same butteraugli)

Apply Settings:
- Interpolation: Tetrahedral
- CPU: AVX512

Filters:
- NL Means (h=0.4, tr=2, a=2, s=4)
- Placebo Deband (radius=24, thr=4.0, grain=0.1, iterations=4)
- Grain (luma_scaling=10.0)
"""

import vapoursynth as vs
import vsfeel
from vsdeband import Grainer, deband_detail_mask, placebo_deband
from vsdenoise import nl_means
from vsrgtools import gauss_blur
from vstools import DitherType, Range, depth, get_y, join

core = vs.core
core.max_cache_size = 12 * 1024
core.max_vram_cache_size = 3 * 1024  # 3gb

LUT_FILE = "luts/boane-dcp-to-bluray-g65_b-balanced_c-smooth_it2000.cube"

clip = core.bs.VideoSource("boane-dcp-sample-tachyon-zone.mxf")
clip = core.resize.Bicubic(
    clip,
    format=vs.RGBS,
    matrix_in_s="rgb",
    transfer_in_s="st428",
    primaries_in_s="xyz",
    range_in_s="full",
    transfer_s="709",
    primaries_s="709",
    chromatic_adaptation=True,
)
clip = core.std.SetFrameProps(
    clip,
    _Range=0,     # Full range
    _Matrix=0,    # RGB / identity
    _Transfer=1,  # Rec.709
    _Primaries=1, # Rec.709
)

clip = core.timecube.Cube(clip, LUT_FILE, range=1, interp=1, cpu=3)
clip = core.resize.Bicubic(
    clip,
    format=vs.YUV444P10,
    matrix_s="709",
    transfer_s="709",
    primaries_s="709",
    range_in_s="full",
    range_s="full",
    dither_type="error_diffusion",
)
clip = depth(clip, 32, range_in=Range.FULL, range_out=Range.FULL, dither_type=DitherType.SIERRA_2_4A)
clip.set_output(0)

# (Vulkan) light NL Means (since i can't use bm3d :sadge:)
final = nl_means(clip, h=0.4, tr=2, a=2, s=4, backend=vsfeel.Backend)  # type: ignore
final = final.std.GPUDownload()

# Deband
# `deband_detail_mask` is a DETAIL mask: white = detail (keep it), black = flat (safe to deband).
dbd_mask = deband_detail_mask(final)
debanded = placebo_deband(final, radius=24, thr=4.0, grain=0.2, iterations=4)
# base = debanded, overlay = original -> smooth the flat areas, keep the detail.
final = core.std.MaskedMerge(debanded, final, dbd_mask)

# grain mask
grain_mask = core.vszip.AdaptiveGrainMask(final, luma_scaling=10.0)

# combined deband + grain mask
flat_mask = core.std.Invert(dbd_mask)
# ...then blur the mask so grain fades in/out rather than starting/stopping abruptly.
flat_mask = gauss_blur(flat_mask, 2.5, radius=16, planes=0)
grain_where = core.akarin.Expr([flat_mask, grain_mask], "x y *")


def quick_grain(clip: vs.VideoNode, mask: vs.VideoNode):
    y = get_y(clip)
    noisy_y = Grainer.FBM_SIMPLEX(y, strength=10.0, temporal=(0.25, 2))
    merged = core.std.MaskedMerge(y, noisy_y, mask)
    return join(merged, clip)


# masked grain
final = quick_grain(final, grain_where)

# final bits
output = depth(
    final, 10,
    range_in=Range.FULL,
    range_out=Range.FULL,
    dither_type=DitherType.SIERRA_2_4A,
)

output.set_output(0)
# set_output(final, "Final")
# set_output(dbd_mask, "Deband Mask")
# set_output(grain_mask, "Grain Mask")
# set_output(grain_where, "Grain+Deband Mask")

# final.set_output(1)
# dbd_mask.set_output(2)
# grain_mask.set_output(3)
# grain_where.set_output(4)
