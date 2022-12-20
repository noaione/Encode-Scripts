import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import iterate, get_y, insert_clip, frame2clip

core.max_cache_size = 12555

cr = nao.src("Fruits Basket (2019) - 05 (CR 1080p).mkv", depth=16, trims=[2, "last"])
wkn = nao.src(
    r".\Fruits Basket 2019 [2019] [WEB-DL] [1080p] [RUS + JAP]\Fruits Basket (2019) - 05 (Wakanim-DL RUS 1080p).mkv",
    depth=16,
)

# CR Trim & Wakanim Replace

cr_pre = cr[0:4744]
cr_mid = cr[4747:17979]
cr_post = cr[17980:-3]
cr = cr_pre + cr_mid + cr_post

cr_x = frame2clip(cr.get_frame(17905)).std.AssumeFPS(fpsnum=24000, fpsden=1001)

wkn = insert_clip(wkn, cr[17906], 17905)

wkn_cut1 = wkn[17905:17976]

# Dehardsub

hardmask = kgf.hardsubmask(wkn, cr, 8)
src = core.std.MaskedMerge(wkn, cr, hardmask)

hardmask_fade = kgf.hardsubmask_fades(src, cr, highpass=4000)
src = core.std.MaskedMerge(src, cr, hardmask_fade)

assmask_clip = core.std.BlankClip(src)
assmask = core.sub.TextFile(assmask_clip, "furuba05.ass")
assmask = core.std.BoxBlur(get_y(assmask), hradius=3, vradius=3).std.Inflate()
src = core.std.MaskedMerge(src, cr, assmask)

src2 = core.std.MaskedMerge(src, cr, assmask)[4668:4675]  # Residue
src = insert_clip(src, src2, 4668)

src = insert_clip(src, wkn_cut1, 17905)  # Replace with full wakanim since the dehardsub are borked

# Filtering

descaled = nao.masked_descale(src, 1280, 720, b=0, c=1, expandN=5, yuv444=True)

sobel_edge = iterate(core.std.Sobel(get_y(descaled)), core.std.Inflate, 2)

adaptmask_dark = kgf.adaptive_grain(descaled, luma_scaling=30, show_mask=True)
adaptmask_light = kgf.adaptive_grain(descaled, luma_scaling=5, show_mask=True).std.Invert()
adaptmask_dark = core.std.Expr([adaptmask_dark, sobel_edge], "x y -")
adaptmask_light = core.std.Expr([adaptmask_light, sobel_edge], "x y -")

normal_deband = core.neo_f3kdb.Deband(descaled, 12, 45, 35, 35, 15, 0, output_depth=16)
much_harder_deband = core.neo_f3kdb.Deband(descaled, 16, 75, 60, 60, 20, 0, output_depth=16)

vid = core.std.MaskedMerge(descaled, normal_deband, adaptmask_light)
vid = core.std.MaskedMerge(descaled, much_harder_deband, adaptmask_dark)
vid = kgf.adaptive_grain(vid, 0.25)
vid = fvf.Depth(vid, 10)

vid.set_output()
