import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import get_y, iterate

core.max_cache_size = 10 * 1024

wkn = nao.src("03/Magia Record - 03 (Wakanim SC 1080p).mkv", depth=16)
funi = nao.src("03/[HorribleSubs] Magia Record - 03 [1080p].mkv", depth=16, trims=[290, 0])

funi = funi.std.DeleteFrames(3643)
funi = funi[:3685] + funi[3697:]
funi = funi[:18009] + funi[18030:]
funi = funi[:33890] + funi[33892:]
funi = funi[:34010] + (funi[34009] * 10)

src_mask1 = kgf.hardsubmask(wkn, funi)
src_dialog = core.std.MaskedMerge(wkn, funi, src_mask1)
src_mask2 = kgf.hardsubmask_fades(src_dialog, funi)
src = core.std.MaskedMerge(src_dialog, funi, src_mask2)

# Nitpick
src = fvf.rfs(
    src, wkn, "[0 3532] [4544 6016] [8217 8338] [8387 10206] [10609 17210] [17229 29643] [29817 33889]"
)
src = fvf.rfs(
    src, src_dialog, "[0 3532] [4544 6016] [8217 8338] [8387 10206] [10609 17210] [17229 29643] [29817 33889]"
)

# stacked1 = core.std.StackVertical([wkn, funi])
# stacked23 = core.std.StackVertical([src_mask2, get_y(wkn.std.BlankClip())])
# stacked2 = core.std.StackVertical([get_y(src), src_mask2])
# stackstack = core.std.StackHorizontal([get_y(stacked1), stacked2])
# stackstack2 = core.std.StackHorizontal([get_y(stacked1), stacked23])

sobel_edge = iterate(core.std.Sobel(get_y(src)), core.std.Inflate, 2).std.Invert()
normal_deband = core.neo_f3kdb.Deband(src, 15, 80, 75, 75, 20, 0, preset="medium", output_depth=16)

vid = core.std.MaskedMerge(src, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()
