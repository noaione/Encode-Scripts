import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import insert_clip, get_y, iterate

core.max_cache_size = 10 * 1024

wkn = nao.src("01/227_01_EN_HD.mp4", depth=16)
funi = nao.src("01/[HorribleSubs] Nanabun no Nijyuuni - 01 [1080p].mkv", depth=16, trims=[290, 0])

funi = funi[:16570] + funi[16668:]
funi = funi[:34287] + funi[34307:]
funi = funi + (funi[-1] * 72)

src_mask1 = kgf.hardsubmask(wkn, funi)
src_dialog = core.std.MaskedMerge(wkn, funi, src_mask1)
src_mask2 = kgf.hardsubmask_fades(src_dialog, funi)
src_full = core.std.MaskedMerge(src_dialog, funi, src_mask2)

# Neat-pick :^)
src = insert_clip(src_dialog, src_full[21018:21074], 21018)
src = insert_clip(src_dialog, src_full[13853:13996], 13853)
src = insert_clip(src_dialog, src_full[13026:13142], 13026)
src = insert_clip(src_dialog, src_full[11768:11850], 11768)
src = insert_clip(src_dialog, src_full[11355:11439], 11355)
src = insert_clip(src_dialog, src_full[10316:10396], 10316)
src = insert_clip(src_dialog, src_full[9584:9656], 9584)
src = insert_clip(src_dialog, src_full[8160:8300], 8160)
src = insert_clip(src_dialog, src_full[5460:5532], 5460)

src = nao.rapplym(src, funi, "masks/01")

#stacked1 = core.std.StackVertical([get_y(wkn), get_y(funi)])
#stacked2 = core.std.StackVertical([get_y(src), src_mask2])
#stackstack = core.std.StackHorizontal([stacked1, stacked2])

sobel_edge = iterate(core.std.Sobel(get_y(src)), core.std.Inflate, 2).std.Invert()
normal_deband = core.f3kdb.Deband(src, 12, 60, 55, 55, 15, 0, preset="medium/nograin", output_depth=16)

vid = core.std.MaskedMerge(src, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()