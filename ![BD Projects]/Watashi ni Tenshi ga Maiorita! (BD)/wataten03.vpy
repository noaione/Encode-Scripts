import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa

core.max_cache_size = 12500

src = nao.src(r'.\BDROM\Wataten03_BD.mkv', depth=16)
src = src.std.Trim(0, 34022)

v = nao.descale(src, 1280, 720, kernel='lanczos', yuv444=True, taps=3, expandN=3, inflateN=1, masked=True)
v = core.f3kdb.Deband(v, 12, 40, 20, 20, 15, 0, output_depth=16)
v = nao.adaptive_degrain(v, 60, 50, 6, iter_edge=1)
v = fvf.Depth(v, 10)

v.set_output()