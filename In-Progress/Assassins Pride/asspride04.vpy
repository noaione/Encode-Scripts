import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import iterate, get_y, get_w

core.max_cache_size = 12000

# src_aod = nao.src("Assassins Pride - 02 (AoD 1080p+).mkv", depth=16)
src = nao.src("Assassins Pride - 04 (VRV HIDIVE 1080p).mkv", depth=16)

descaled = nao.masked_descale(src, get_w(844), 844, "bicubic", b=0.5, c=0, yuv444=True, expandN=3)

sobel_edge = iterate(core.std.Sobel(get_y(descaled)), core.std.Inflate, 2)
normal_deband = core.neo_f3kdb.Deband(descaled, 12, 60, 45, 45, 15, 0, output_depth=16)

vid = core.std.MaskedMerge(descaled, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=190, thSADC=150)
vid = vid.resize.Spline16(1280, 720)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=16)
# vid = taa.TAAmbk(vid, "Nnedi3", cycle=2)
vid = fvf.Depth(vid, 10)

vid.set_output()
