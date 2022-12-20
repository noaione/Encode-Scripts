import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import get_y, iterate

core.max_cache_size = 8192

wkn = nao.src("01/Magia_Record_01_EN_HD.mp4", depth=16)
funi = nao.src("01/Magia Record - 01 (Funimation 1080p).mkv", depth=16, trims=[290, 0])

# Sync to wakanim
funi_clip1 = funi[18149] * 4
funi = funi[:18149] + funi_clip1 + funi[18149:]

funi_clip2 = funi[33929] * 18
funi = funi[:33929] + funi_clip2 + funi[33929:]
funi = funi[:34055]

# Dehardsub
src_mask1 = kgf.hardsubmask(wkn, funi)
src_dialog = core.std.MaskedMerge(wkn, funi, src_mask1)
src_mask2 = kgf.hardsubmask_fades(src_dialog, funi)
src = core.std.MaskedMerge(src_dialog, funi, src_mask2)
src = nao.rapplym(src, funi, "masks/01")

# Cherry-pick
# Use wakanim frame for all action scene since it get caught on dehardsub mask

# "[0 634] [668 669] [707 711] [769 816] [868 869] [943 1235] [1291 1292] [1397 1402] [1504 2060] [2114 2115] [2168 2172] [2269 2395] [2443 2451] [2539 2641] [2742  2744] [2837 2840] [2887 2890] [2908 3644] [3692 3697] [3755 3771] [3829 3833] [3894 4021] [4097 4361]"

src = fvf.rfs(src, wkn, "[0 4361] [18154 24476] [30457 30594] [31652 33947]")
src = fvf.rfs(src, src_dialog, "[0 4361] [18154 24476] [30457 30594] [31652 33947]")

###############
#             #
# COMPARISION #
#             #
###############

# stacked1 = core.std.StackVertical([get_y(wkn), get_y(funi)])
# stacked2 = core.std.StackVertical([get_y(src), src_mask2])
# stackstack = core.std.StackHorizontal([stacked1, stacked2])
# vsbookmark.generate(funi, 'magireco01')
# nao.check_diff(wkn, funi, 0.09)

###############
###############

# Filter

sobel_edge = iterate(core.std.Sobel(get_y(src)), core.std.Inflate, 2).std.Invert()
normal_deband = core.neo_f3kdb.Deband(src, 15, 80, 75, 75, 25, 0, output_depth=16)

vid = core.std.MaskedMerge(src, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()
