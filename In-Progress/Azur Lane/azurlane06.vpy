import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import get_y, iterate, split, join

core.max_cache_size = 11000

src_amz = nao.src("06/Azur Lane - 06 (Amazon dAnime VBR 1080p).mkv")
src_wkn = nao.src("06/Azur Lane - 06 (Wakanim RU 1080p).mkv")

src_wkn = src_amz[0] + src_wkn # Sync to Amazon

srcm = kgf.hardsubmask(src_wkn, src_amz)
src = core.std.MaskedMerge(src_wkn, src_amz, srcm)
src2m = fvf.Depth(kgf.hardsubmask_fades(src, src_amz, highpass=3500), 8)
src = core.std.MaskedMerge(src, src_amz, src2m)

src = nao.recursive_apply_mask(src, src_amz, 'masks/06/')
src = fvf.Depth(src, 16)

# Shift 4:4:4
Y, U, V = split(src)
shift_U, shift_V = core.resize.Spline16(U, src.width, src.height, src_left=0.5), core.resize.Spline16(V, src.width, src.height, src_left=0.5)

src = join([Y, shift_U, shift_V])

resized = core.resize.Bicubic(src, 1280, 720)

# Anti-edge adapptive deband
sobel_edge = iterate(core.std.Sobel(get_y(resized)), core.std.Inflate, 2)
adaptmask_light = kgf.adaptive_grain(resized, luma_scaling=5, show_mask=True).std.Invert()
adaptmask_light = core.std.Expr([adaptmask_light, sobel_edge], 'x y -')
normal_deband = core.f3kdb.Deband(resized, 14, 65, 50, 50, 15, 0, output_depth=16)

# shit
vid = core.std.MaskedMerge(resized, normal_deband, adaptmask_light)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=140, thSADC=100)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=16)
vid = fvf.Depth(vid, 10)

vid.set_output()