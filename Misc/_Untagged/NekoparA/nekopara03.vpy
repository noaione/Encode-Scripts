import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
from vsutil import insert_clip, get_y, iterate, get_w

core.max_cache_size = 12 * 1024

src_wkn = nao.src("03/NekoparA - 03 (Wakanim SC 1080p).mkv", depth=16)
src_amz = nao.src("03/NekoparA - 03 (Amazon dAnime VBR 1080p).mkv", depth=16, trims=[1])

src_mask1 = kgf.hardsubmask(src_wkn, src_amz)
src_dialog = core.std.MaskedMerge(src_wkn, src_amz, src_mask1)

exm1 = kgf.squaremask(src_wkn, 400, 80, 465, 30)
src_ex1 = core.std.MaskedMerge(src_dialog, src_amz, exm1)

src = fvf.rfs(src_dialog, src_ex1, "[10853 10906]")

src = nao.rapplym(src, src_amz, "masks/03")
src_ex2 = nao.rapplym(src, src_amz, "masks/03")

src = fvf.rfs(src, src_ex2, "[5966 6003] [16381 16427]")

exm2 = kgf.squaremask(src_wkn, 500, 80, 780, 560)
src_ex3 = core.std.MaskedMerge(src, src_amz, exm2)

src = fvf.rfs(src, src_ex3, "[15656 15782]")

exm3 = kgf.squaremask(src_wkn, 580, 60, 60, 900)
src_ex4 = core.std.MaskedMerge(src, src_amz, exm3)

src = fvf.rfs(src, src_ex4, "[33325 33443]")

# stacked1 = core.std.StackVertical([src_wkn, src_amz])
# stacked2 = core.std.StackVertical([get_y(src_ex4), exm3])
# stackstack = core.std.StackHorizontal([get_y(stacked1), stacked2]).resize.Bicubic(1280, 720)

sobel_edge = iterate(core.std.Sobel(get_y(src)), core.std.Inflate, 2).std.Invert()
normal_deband = core.neo_f3kdb.Deband(src, 15, 80, 75, 75, 25, 0, output_depth=16)

vid = core.std.MaskedMerge(src, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()
