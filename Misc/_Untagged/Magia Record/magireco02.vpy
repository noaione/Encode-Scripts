import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import get_y, iterate

core.max_cache_size = 10 * 1024

wkn = nao.src("02/Magia_Record_02_EN_HD.mp4", depth=16)
funi = nao.src("02/[HorribleSubs] Magia Record - 02 [1080p].mkv", depth=16, trims=[290, 0])

funi = funi[:4456] + funi[4458:]
funi = funi[:4496] + funi[4498:]
funi = funi[:16875] + (funi[16874] * 3) + funi[16875:]
funi = funi[:16916] + funi[16926:]
funi = funi[:34013]

src_mask1 = kgf.hardsubmask(wkn, funi)
src_dialog = core.std.MaskedMerge(wkn, funi, src_mask1)
src_mask2 = kgf.hardsubmask_fades(src_dialog, funi)
src = core.std.MaskedMerge(src_dialog, funi, src_mask2)

# Nitpick
src = fvf.rfs(src, wkn, "[0 4455] [6849 10970] [24208 25274] [28514 31237] [31756 33913]")
src = fvf.rfs(src, src_dialog, "[0 4455] [6849 10970] [24208 25274] [28514 31237] [31756 33913]")

# stacked1 = core.std.StackVertical([get_y(wkn), get_y(funi)])
# stacked2 = core.std.StackVertical([get_y(src), src_mask2])
# stackstack = core.std.StackHorizontal([stacked1, stacked2])

sobel_edge = iterate(core.std.Sobel(get_y(src)), core.std.Inflate, 2).std.Invert()
normal_deband = core.neo_f3kdb.Deband(src, 15, 80, 75, 75, 20, 0, preset="medium", output_depth=16)

vid = core.std.MaskedMerge(src, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()
