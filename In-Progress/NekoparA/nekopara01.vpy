import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
from vsutil import get_y, iterate

core.max_cache_size = 11 * 1024

src_wkn = nao.src("01/Nekopara_01_EN_HD.mp4", depth=16)
src_amz = nao.src("01/NekoparA - 01 (Amazon dAnime VBR 1080p).mkv", depth=16, trims=[1, 0])

src_amz_new = nao.src("01/NekoparA - 01v2 (Amazon dAnime VBR 1080p).mkv", depth=16)
# Dehardsub
src_mask1 = kgf.hardsubmask(src_wkn, src_amz)
src_dialog = core.std.MaskedMerge(src_wkn, src_amz, src_mask1)
src = nao.rapplym(src_dialog, src_amz, "masks/01")

#aaed = taa.TAAmbk(src, "Nnedi3", cycle=1, mtype=3, sharp=-1)

src = src[:958+1] + src_amz_new[960:3116+1] + src[959:]

# Comparision
#stacked1 = core.std.StackVertical([src, aaed])
#stacked2 = core.std.StackVertical([get_y(src), src_mask1])
#stackstack = core.std.StackHorizontal([stacked1, stacked2])

sobel_edge = iterate(core.std.Sobel(get_y(src)), core.std.Inflate, 2).std.Invert()
normal_deband = core.f3kdb.Deband(src, 15, 80, 75, 75, 25, 0, output_depth=16)

vid = core.std.MaskedMerge(src, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()
