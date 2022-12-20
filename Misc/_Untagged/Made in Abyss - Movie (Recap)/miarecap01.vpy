import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import get_y, iterate

core.max_cache_size = 12500


def compmask4(srcs, mask):
    src_l = core.std.StackVertical(srcs[:2])
    src_l = get_y(src_l)
    src_r = core.std.StackVertical([get_y(srcs[-1]), mask])
    return core.std.StackHorizontal([src_l, src_r])


wkn_sc = nao.src("01/Made in Abyss - Movie 01 (Wakanim SC 1080p).mkv", depth=16, trims=[0, -360])
wkn_ru = nao.src(
    "01/Gekijouban Soushuuhen Made in Abyss - 01 (WEB-DL 1920x1080 x264 AAC Rus).mkv", depth=16, trims=[0, -360]
)
amz_720p = nao.src("01/Made in Abyss - Movie 01 (Amazon dAnime VBR 720p).mkv", depth=16, trims=[0, -1])
amz_1080p = core.resize.Spline36(amz_720p, 1920, 1080)

# Remove Dialog
undialog_mask = kgf.hardsubmask(wkn_sc, amz_1080p)
src_undialog = core.std.MaskedMerge(wkn_sc, amz_1080p, undialog_mask)
undialog_mask_fade = kgf.hardsubmask_fades(src_undialog, amz_1080p)
undialog_mask_fade = nao.sfr(undialog_mask_fade, 46298, "46297")
src_undialog_fade = core.std.MaskedMerge(src_undialog, amz_1080p, undialog_mask_fade)

src_undialog = fvf.rfs(src_undialog, src_undialog_fade, "[46297 46306] [152856 152874] [167472 167490]")
src_undialog_extra = kgf.squaremask(src_undialog, 530, 150, 720, 80)
src_undialog = core.std.MaskedMerge(src_undialog, wkn_ru, src_undialog_extra)

# Remove TS from RU
untsed_mask = kgf.hardsubmask_fades(wkn_ru, src_undialog)
untsed_mask = nao.sfr(untsed_mask, 5836, "5824-5833")
untsed_mask = nao.sfr(untsed_mask, 46309, "46296-46300")
untsed_mask = nao.sfr(untsed_mask, 73899, "73898")
untsed_mask = nao.sfr(untsed_mask, 89589, "89588")
untsed_mask = nao.sfr(untsed_mask, 152859, "152856-152858")
untsed_mask = nao.sfr(untsed_mask, 167491, "167472-167602")
untsed_mask = nao.sfr(untsed_mask, 170606, "170582-170711")
src_untsed = core.std.MaskedMerge(wkn_ru, src_undialog, untsed_mask)

# stackeded = compmask4([wkn_ru, src_undialog_fade, src_untsed], untsed_mask)

rescaled = nao.adaptive_rescale(src_untsed, 1920, 1080, [846, 849])

sobel_edge = iterate(core.std.Sobel(get_y(rescaled)), core.std.Inflate, 2)
normal_deband = core.neo_f3kdb.Deband(rescaled, 14, 60, 45, 35, 15, 0, output_depth=16)

vid = core.std.MaskedMerge(rescaled, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=190, thSADC=150)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=16)
# vid = taa.TAAmbk(vid, "Nnedi3", cycle=2)
vid = fvf.Depth(vid, 10)

vid.set_output()
