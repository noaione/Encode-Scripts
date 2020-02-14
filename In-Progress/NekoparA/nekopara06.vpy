import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
from vsutil import *
from nnedi3_rpow2 import *

core.max_cache_size = 12 * 1024

src_wkn = nao.src("06/NekoparA - 06 (Wakanim SC 1080p).mkv", depth=8)
src_amz = nao.src("06/NekoparA - 06 (Amazon dAnime VBR 1080p).mkv", depth=8, trims=[1])

src_mask1 = kgf.hardsubmask(src_wkn, src_amz)
src_dialog = core.std.MaskedMerge(src_wkn, src_amz, src_mask1)

exm1 = kgf.squaremask(src_wkn, 520, 80, 70, 900)
exm2  = kgf.squaremask(src_wkn, 535, 60, 1200, 900)
exmclamp = core.std.Expr(clips=[exm1, exm2], expr=["x y +"])
src_ex1 = core.std.MaskedMerge(src_dialog, src_amz, exmclamp)

src = fvf.rfs(src_dialog, src_ex1, "[33326 33444]")

planes = split(src)
planes[1], planes[2] = [core.resize.Bicubic(plane, src_left=0.25, src_top=-0.125) for plane in planes[1:]]
src = join(planes)

native_mask = nao.simple_native_mask(src, get_w(846), 846, 0, 0, 3, True)
descaled = nao.descale(src, get_w(846), 846, expandN=3)
rescaled = nnedi3_rpow2(descaled, nns=4, correct_shift=True, width=1920, height=1080)

rescaled = core.std.MaskedMerge(rescaled, src, native_mask)
src = fvf.rfs(rescaled, src, "[33326 33444]")
src = fvf.Depth(src, 16)

#stacked1 = core.std.StackVertical([src_wkn, src_amz])
#stacked2 = core.std.StackVertical([get_y(src_ex1), exmclamp])
#stackstack = core.std.StackHorizontal([get_y(stacked1), stacked2]).resize.Bicubic(1280, 720)

sobel_edge = iterate(core.std.Sobel(get_y(src)), core.std.Inflate, 2).std.Invert()
normal_deband = core.f3kdb.Deband(src, 15, 80, 75, 75, 25, 0, output_depth=16)

vid = core.std.MaskedMerge(src, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()