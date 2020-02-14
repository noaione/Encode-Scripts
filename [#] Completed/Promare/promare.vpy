import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
from vsutil import insert_clip, get_y, iterate, get_w
from nnedi3_rpow2 import *

core.max_cache_size = 12 * 1024

src = nao.src(r"./BDROM/BDMV/STREAM/PROMARE_BDRemux.mkv", trims=[0, -26])

native_mask = nao.simple_native_mask(src, get_w(892), 892, 1, 1).resize.Spline36(1920, 1080)
descaled = nao.descale(src, get_w(892), 892, expandN=3, b=1, c=0)
rescaled = nnedi3_rpow2(descaled, nns=4, correct_shift=True, width=1920, height=1080)

#stacked2 = core.std.StackVertical([get_y(src), native_mask])

masked_src = core.std.MaskedMerge(rescaled, src, native_mask)

src_final = fvf.rfs(masked_src, src, "[0 720] [5945 6053] [6696 6790] [130756 131215] [151321 151509] [151616 151693] [151799 151876] [151913 151966] [152006 152060] [152086 152147] [152188 152240] [152271 152335] [152396 152428] [152469 152522] [152930 152980] [153082 159814]")
src_final = fvf.Depth(src, 16)

sobel_edge = iterate(core.std.Sobel(get_y(src_final)), core.std.Inflate, 2).std.Invert()
normal_deband = core.f3kdb.Deband(src_final, 12, 60, 55, 55, 15, 0, preset="medium/nograin", output_depth=16)

vid = core.std.MaskedMerge(src_final, normal_deband, sobel_edge)
vid = nao.adaptive_smdegrain(vid, luma_scaling=10, iter_edge=1, thSAD=100, thSADC=80)
vid = kgf.adaptive_grain(vid, 0.35, luma_scaling=20)
vid = fvf.Depth(vid, 10)

vid.set_output()