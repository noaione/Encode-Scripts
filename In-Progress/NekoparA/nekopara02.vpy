import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
from vsutil import insert_clip, get_y, iterate

core.max_cache_size = 12 * 1024

src_wkn = nao.src("02/NekoparA - 02 (Wakanim SC 1080p).mkv", depth=16)
src_cr = nao.src("02/[HorribleSubs] Nekopara - 02 [1080p].mkv", depth=16, trims=[169, 0])

src_mask1 = kgf.hardsubmask(src_wkn, src_cr)
src_dialog = core.std.MaskedMerge(src_wkn, src_cr, src_mask1)

src = nao.rapplym(src_dialog, src_cr, "masks/02")
src = nao.rapplym(src, src_cr, "masks/02")

#stacked1 = core.std.StackVertical([get_y(src_wkn), get_y(src_cr)])
#stacked2 = core.std.StackVertical([get_y(src), src_mask1])
#stackstack = core.std.StackHorizontal([stacked1, stacked2])

sobel_edge = iterate(core.std.Sobel(get_y(src)), core.std.Inflate, 2).std.Invert()
normal_deband = core.f3kdb.Deband(src, 15, 80, 75, 75, 25, 0, output_depth=16)

vid = core.std.MaskedMerge(src, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()