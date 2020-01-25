import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
from vsutil import insert_clip, get_y, iterate
from nnedi3_rpow2 import *

core.max_cache_size = 12 * 1024

wkn = nao.src("03/227 - 03 (Wakanim SC 1080p).mkv", depth=16)
funi = nao.src("03/227 - 03 (Funimation 1080p).mkv", depth=16, trims=[290, 0])

funi = funi[:12701] + funi[12706:]
funi = funi[:34397] + funi[34398:]

src_mask1 = kgf.hardsubmask(wkn, funi)
src_dialog = core.std.MaskedMerge(wkn, funi, src_mask1)
src_mask2 = kgf.hardsubmask_fades(src_dialog, funi)
src_mask2 = nao.sfr(src_mask2, 2819, "2795-2897")
src_full = core.std.MaskedMerge(src_dialog, funi, src_mask2)

src = fvf.rfs(src_dialog, src_full, "[10304 10336] [3238 3296] [29162 29214] [2795 2897]")

native_mask = nao.simple_native_mask(src, 1600, 900, 1, 1).resize.Spline16(1920, 1080)
descaled = nao.descale(src, 1600, 900, expandN=3)
rescaled = nnedi3_rpow2(descaled, nns=4, correct_shift=True, width=1920, height=1080)

rescaled = core.std.MaskedMerge(rescaled, src, native_mask)
src = fvf.rfs(rescaled, src, "[30297 32431]")

#stacked1 = core.std.StackVertical([wkn, funi])
#stacked2 = core.std.StackVertical([get_y(src), src_mask2])
#stackstack = core.std.StackHorizontal([get_y(stacked1), stacked2]).resize.Bicubic(1280, 720)

sobel_edge = iterate(core.std.Sobel(get_y(src)), core.std.Inflate, 2).std.Invert()
normal_deband = core.f3kdb.Deband(src, 12, 60, 55, 55, 15, 0, preset="medium/nograin", output_depth=16)

vid = core.std.MaskedMerge(src, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()