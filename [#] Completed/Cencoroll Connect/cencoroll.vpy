import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
from vsutil import get_y, iterate, get_w
from nnedi3_rpow2 import *

core.max_cache_size = 12000

src = nao.src("00001.mkv", depth=16)

"""
Cencoroll 1 Filter
"""
cen1_src = src[528:36609]
cen1 = core.edgefixer.ContinuityFixer(cen1_src, 4, 4, 4, 4, 5)
cen1 = nao.descale(cen1, get_w(680), 680, 'bicubic', b=0.2, c=0.5, expandN=3)

cen1_edge = iterate(core.std.Sobel(get_y(cen1)), core.std.Inflate, 6)
cen1_linedarken = haf.FastLineDarkenMOD(cen1, 56, 6, 180, 8)

cen1 = core.std.MaskedMerge(cen1, cen1_linedarken, cen1_edge)
cen1 = taa.TAAmbk(cen1, "Nnedi3UpscaleSangNom", cycle=1, mtype=3, sharp=-1)
cen1 = nnedi3_rpow2(cen1, width=1280, height=720, nsize=5, nns=2) # First upscaling
cen1 = nnedi3_rpow2(cen1, width=1920, height=1080, nsize=5, nns=2) # Second Upscaling

"""
Cencoroll 2 (Connect) Filter
"""
cen2_src = src[36788:99705]

cen2 = haf.HQDeringmod(cen2_src, mrad=1, msmooth=3, thr=8.0, mthr=60)

cen2_edge = iterate(core.std.Sobel(get_y(cen2)), core.std.Inflate, 3)
cen2_edge = iterate(cen2_edge, core.std.Deflate, 2).std.BoxBlur(hradius=1, vradius=1)
cen2_blur = core.std.BoxBlur(cen2, hradius=1.5, vradius=1.5)

cen2 = core.std.MaskedMerge(cen2, cen2_blur, cen2_edge)

"""
Cencoroll 2 (Connect) After Credits Scene Filter
"""
afcred_src = src[105950:106426]

afcred = haf.HQDeringmod(afcred_src, mrad=1, msmooth=3, thr=8.0, mthr=60)

afcred_edge = iterate(core.std.Sobel(get_y(afcred)), core.std.Inflate, 3)
afcred_edge = iterate(afcred_edge, core.std.Deflate, 2).std.BoxBlur(hradius=1, vradius=1)
afcred_blur = core.std.BoxBlur(afcred, hradius=1.5, vradius=1.5)

afcred = core.std.MaskedMerge(afcred, afcred_blur, afcred_edge)

"""
Merge back all clip together and do final filtering
"""
merged_clip = src[0:528] + cen1 + src[36609:36788] + cen2 + src[99705:105950] + afcred + src[106426:]

sobel_edge = iterate(core.std.Sobel(get_y(merged_clip)), core.std.Inflate, 2)
normal_deband = core.f3kdb.Deband(merged_clip, 12, 60, 15, 15, 15, 0, output_depth=16)
vid = core.std.MaskedMerge(merged_clip, normal_deband, sobel_edge)

vid = nao.adaptive_degrain2(vid, 10, 'knlm', 'light', 1, d=2, a=2, s=8, h=1.2)
vid = nao.adaptive_degrain2(vid, 24, 'knlm', 'dark', 1, d=3, a=3, s=8, h=1.8)
vid = kgf.adaptive_grain(vid, 0.45)
vid = fvf.Depth(vid, 10)

vid.set_output()