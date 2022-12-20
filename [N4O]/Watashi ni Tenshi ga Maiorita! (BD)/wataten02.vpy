import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
import mvsfunc as mvf

src = nao.src(r".\BDROM\Wataten02_BD.mkv", depth=16)
src = src.std.Trim(0, 34022)

v = nao.masked_descale(src, 1280, 720, kernel="lanczos", yuv444=True, taps=3, expandN=3, masked=True)
v = core.neo_f3kdb.Deband(v, 12, 40, 20, 20, 15, 0, output_depth=16)
v = nao.adaptive_degrain2(v, thSAD=60, thSADC=50, luma_scaling=6, iter_edge=1)
v = fvf.Depth(v, 10)

# src_tv = nao.src(r'X:\NEW\[4] Winter 2018\Watashi ni Tenshi ga Maiorita!\[blank+] Watashi ni Tenshi ga Maiorita! - 02 [3AC4F314].mkv', depth=False)#.std.DeleteFrames(33686)
# diff = nao.save_difference(src_tv, src, 0.09)

v.set_output()
