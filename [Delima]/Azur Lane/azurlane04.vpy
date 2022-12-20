import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import get_y, iterate, split, join

core.max_cache_size = 11000

src = nao.src("Azur Lane - 04 (Amazon dAnime VBR 1080p).mkv")
# s = nao.src("03/Azur Lane - 04 (Wakanim RU 1080p).mkv")

# src_wkn = src_amz[0] + src_wkn # Sync to Amazon
#
# srcm = kgf.hardsubmask(src_wkn, src_amz)
# src = core.std.MaskedMerge(src_wkn, src_amz, srcm)
# src2m = fvf.Depth(kgf.hardsubmask_fades(src, src_amz, highpass=3500), 8)
# src = core.std.MaskedMerge(src, src_amz, src2m)

# src = nao.recursive_apply_mask(src, src_amz, '03/masks/')
src = fvf.Depth(src, 16)

# Shift 4:4:4
Y, U, V = split(src)
shift_U, shift_V = core.resize.Spline16(U, src.width, src.height, src_left=0.5), core.resize.Spline16(
    V, src.width, src.height, src_left=0.5
)

src = join([Y, shift_U, shift_V])

resized = core.resize.Spline36(src, 1280, 720)

# Masked DeHalo
mask = core.std.Sobel(resized, 0)
luma = core.std.ShufflePlanes(mask, 0, colorfamily=vs.GRAY)
mask_outer = iterate(luma, core.std.Maximum, 2)
mask_inner = iterate(mask_outer, core.std.Minimum, 3)
halos = core.std.Expr([mask_outer, mask_inner], "x y -")
dehalo = haf.DeHalo_alpha(resized)
masked_dehalo = core.std.MaskedMerge(resized, dehalo, halos)

# Anti-edge adapptive deband
sobel_edge = iterate(core.std.Sobel(get_y(masked_dehalo)), core.std.Inflate, 2)
adaptmask_light = kgf.adaptive_grain(masked_dehalo, luma_scaling=5, show_mask=True).std.Invert()
adaptmask_light = core.std.Expr([adaptmask_light, sobel_edge], "x y -")
normal_deband = core.neo_f3kdb.Deband(masked_dehalo, 14, 65, 50, 50, 15, 0, output_depth=16)

# shit
vid = core.std.MaskedMerge(masked_dehalo, normal_deband, adaptmask_light)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=140, thSADC=100)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=16)
vid = fvf.Depth(vid, 10)

vid.set_output()
