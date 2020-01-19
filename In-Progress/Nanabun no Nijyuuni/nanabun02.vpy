import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
from vsutil import insert_clip, get_y, iterate

core.max_cache_size = 12 * 1024

wkn = nao.src("02/227 - 02 (Wakanim SC 1080p).mkv", depth=16)
funi = nao.src("02/[HorribleSubs] Nanabun no Nijyuuni - 02 [1080p].mkv", depth=16, trims=[290, 0])

funi = funi[:15254] + funi[15264:]
funi = funi[:34392] + funi[34396:] + (funi[-1] * 72)

src_mask1 = kgf.hardsubmask(wkn, funi)
src_dialog = core.std.MaskedMerge(wkn, funi, src_mask1)
src_mask2 = kgf.hardsubmask_fades(src_dialog, funi)
src_full = core.std.MaskedMerge(src_dialog, funi, src_mask2)

src = fvf.rfs(src_dialog, src_full, "[4051 4152] [7543 7785] [9015 9055] [15118 15176] [19140 19211] [19388 19438] [27191 27234] [31855 31881] [32095 32158]")

src = nao.rapplym(src, funi, "masks/02")
src_extra = nao.rapplym(src, funi, "masks/02")

src = fvf.rfs(src, src_extra, "[7544 7546]")

#stacked1 = core.std.StackVertical([wkn, funi])
#stacked2 = core.std.StackVertical([get_y(src), src_mask2])
#stackstack = core.std.StackHorizontal([get_y(stacked1), stacked2])

sobel_edge = iterate(core.std.Sobel(get_y(src)), core.std.Inflate, 2).std.Invert()
normal_deband = core.f3kdb.Deband(src, 12, 60, 55, 55, 15, 0, preset="medium/nograin", output_depth=16)

vid = core.std.MaskedMerge(src, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()