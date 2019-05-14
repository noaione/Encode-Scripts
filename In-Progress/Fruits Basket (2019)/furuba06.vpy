import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import get_y, iterate

core.max_cache_size = 12555

cr = nao.src('Fruits Basket (2019) - 06 (CR 1080p).mkv', depth=16, trims=[2, 'last'])
wkn = nao.src(r'.\Fruits Basket 2019 [2019] [WEB-DL] [1080p] [RUS + JAP]\Fruits Basket (2019) - 06 (Wakanim-DL RUS 1080p).mkv', depth=16)

# Trimming

cr_pre = cr[0:2875]
cr_mid = cr[2878:17165]
cr_post = cr[17168:34283]
cr = cr_pre + cr_mid + cr_post

wkn = nao.sfr(wkn, 34161, '34160') # n4ofunc.SimpleFrameReplace

# Dehardsub

hardmask = kgf.hardsubmask(wkn, cr, 8)
src = core.std.MaskedMerge(wkn, cr, hardmask)

hardmask_fade = kgf.hardsubmask_fades(src, cr)
src = core.std.MaskedMerge(src, cr, hardmask_fade)

assmask_clip = core.std.BlankClip(src)
assmask = core.sub.TextFile(assmask_clip, 'furuba06.ass')
assmask = core.std.BoxBlur(get_y(assmask), hradius=3, vradius=3)
src = core.std.MaskedMerge(src, cr, assmask)

# Filtering

descaled = nao.descale(src, 1280, 720, b=0, c=1, expandN=5, inflateN=2, yuv444=True, iter_max=2)

sobel_edge = iterate(core.std.Sobel(get_y(descaled)), core.std.Inflate, 2)

adaptmask_dark = kgf.adaptive_grain(descaled, luma_scaling=30, show_mask=True)
adaptmask_light = kgf.adaptive_grain(descaled, luma_scaling=5, show_mask=True).std.Invert()
adaptmask_dark = core.std.Expr([adaptmask_dark, sobel_edge], 'x y -')
adaptmask_light = core.std.Expr([adaptmask_light, sobel_edge], 'x y -')

normal_deband = core.f3kdb.Deband(descaled, 12, 45, 35, 35, 15, 0, output_depth=16)
much_harder_deband = core.f3kdb.Deband(descaled, 16, 75, 60, 60, 20, 0, output_depth=16)

vid = core.std.MaskedMerge(descaled, normal_deband, adaptmask_light)
vid = core.std.MaskedMerge(descaled, much_harder_deband, adaptmask_dark)
vid = kgf.adaptive_grain(vid, 0.25)
vid = fvf.Depth(vid, 10)

wkn.set_output()