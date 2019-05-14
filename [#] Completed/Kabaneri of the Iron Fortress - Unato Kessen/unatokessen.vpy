import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
from vsutil import insert_clip, iterate, get_y

core.max_cache_size = 12555

# AMAZON COUNTERPART

# Frame matching for Netflix 5.1 and Amazon 2.0 audio (Synced to Amazon)

## Episode 1
amz01 = nao.src('Koutetsujou no Kabaneri Unato Kessen - 01 (AMZN H264 1080p).mkv', depth=16)
am01_intro = amz01[0:650+1]
am01_main = amz01[819:]
amz01 = am01_intro + am01_main

amz01 = amz01[0:34636+1]
amz02 = nao.src('Koutetsujou no Kabaneri Unato Kessen - 02 (AMZN H264 1080p).mkv', depth=16, trims=[2899, 34487])
amz03 = nao.src('Koutetsujou no Kabaneri Unato Kessen - 03 (AMZN H264 1080p).mkv', depth=16, trims=[4064, 'black24'])

# NETFLIX

#nf01 = nao.src('Koutetsujou no Kabaneri Unato Kessen - 01 (NF 1080p).mkv', depth=16, trims=[23, 'last']) #34660
#nf02 = nao.src('Koutetsujou no Kabaneri Unato Kessen - 02 (NF 1080p).mkv', depth=16, trims=[2850, 34438])
#nf03 = nao.src('Koutetsujou no Kabaneri Unato Kessen - 03 (NF 1080p).mkv', depth=16, trims=[4015, -1])

#nf01 = nf01[0:34636+1]

#srcNF = nf01 + nf02 + nf03
srcAMZ = amz01 + amz02 + amz03

rescaled = nao.adaptive_rescale(srcAMZ, 1920, 1080, [914, 920], 'lanczos', taps=5, show_mask=False)

sobel_edge = iterate(core.std.Sobel(get_y(rescaled)), core.std.Inflate, 2)

adaptmask_dark = kgf.adaptive_grain(rescaled, luma_scaling=30, show_mask=True)
adaptmask_light = kgf.adaptive_grain(rescaled, luma_scaling=5, show_mask=True).std.Invert()
adaptmask_dark = core.std.Expr([adaptmask_dark, sobel_edge], 'x y -')
adaptmask_light = core.std.Expr([adaptmask_light, sobel_edge], 'x y -')

normal_deband = core.f3kdb.Deband(rescaled, 12, 45, 35, 35, 15, 0, output_depth=16)
much_harder_deband = core.f3kdb.Deband(rescaled, 16, 75, 60, 60, 20, 0, output_depth=16)

vid = core.std.MaskedMerge(rescaled, normal_deband, adaptmask_light)
vid = core.std.MaskedMerge(rescaled, much_harder_deband, adaptmask_dark)
vid = nao.adaptive_degrain(vid, 150, 150, 55, 'dark')
vid = kgf.adaptive_grain(vid, 0.35)
vid = fvf.Depth(vid, 10)

vid.set_output()

