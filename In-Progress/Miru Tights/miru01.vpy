import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import get_y, get_w, insert_clip, iterate

core.max_cache_size = 12555

src = nao.src("Miru Tights - 01 (VM 1080p).mkv", depth=16)

# Splicing

main = src[:5880]
end_card = src[5880:]

# Descaling

main_descale = nao.masked_descale(main, get_w(862), 862, "lanczos", yuv444=True, taps=5, expandN=5)
end_card_resize = core.resize.Bicubic(end_card, get_w(862), 862, nao.register_format(main_descale, True).id)
end_song = main_descale[4680:5880]

# Filtering

sobel_edge = iterate(core.std.Sobel(get_y(main_descale)), core.std.Inflate, 2)

adaptmask_dark = kgf.adaptive_grain(main_descale, luma_scaling=30, show_mask=True)
adaptmask_light = kgf.adaptive_grain(main_descale, luma_scaling=5, show_mask=True).std.Invert()
adaptmask_dark = core.std.Expr([adaptmask_dark, sobel_edge], "x y -")
adaptmask_light = core.std.Expr([adaptmask_light, sobel_edge], "x y -")

normal_deband = core.neo_f3kdb.Deband(main_descale, 12, 45, 35, 35, 15, 0, output_depth=16)
much_harder_deband = core.neo_f3kdb.Deband(main_descale, 16, 75, 60, 60, 20, 0, output_depth=16)
end_song = core.neo_f3kdb.Deband(end_song, 12, 45, 35, 35, 15, 0, output_depth=16)

main_vid = core.std.MaskedMerge(main_descale, normal_deband, adaptmask_light)
main_vid = core.std.MaskedMerge(main_descale, much_harder_deband, adaptmask_dark)
main_vid = insert_clip(main_vid, end_song, 4680)
main_vid = kgf.adaptive_grain(main_vid, 0.4)

# Reinsert back everything and final resize

vid = main_vid + end_card_resize
vid = core.resize.Bicubic(vid, 1440, 810)
vid = fvf.Depth(vid, 10)

vid.set_output()
