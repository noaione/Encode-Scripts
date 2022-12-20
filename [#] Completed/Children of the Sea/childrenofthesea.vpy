import vapoursynth as vs
from vapoursynth import core
import n4ofunc as nao
import fvsfunc as fvf
import kagefunc as kgf
from vsutil import insert_clip, iterate, get_y

core.max_cache_size = 12 * 1024

src = nao.src(r"./TDMPXA_105/BDMV/PLAYLIST/00001.mkv", crop_r=2, crop_l=2, crop_t=132, crop_b=132)

vid = core.edgefixer.Continuity(src, 2, 2, 2, 2, 4)
vid = fvf.Depth(vid, 16)

# ED Part #
edc = vid[150623:156267]
edc = core.sub.Subtitle(
    edc, r"{\an7\fad(250,250)\pos(0,0)\c&H010001&\blur5\fscx100\fscy100\p1}m 1735 25 l 1903 25 1903 526 1735 526"
)

vid = insert_clip(vid, edc, 150623)

sobel_edge = iterate(core.std.Sobel(get_y(vid)), core.std.Inflate, 2).std.Invert()
normal_deband = core.neo_f3kdb.Deband(vid, 15, 80, 75, 75, 25, 0, output_depth=16)

vid = core.std.MaskedMerge(vid, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=2, thSAD=160, thSADC=140)
vid = kgf.adaptive_grain(vid, 0.45, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()
