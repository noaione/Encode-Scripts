import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import n4ofunc as nao
import vsTAAmbk as taa
from vsutil import insert_clip, get_y, iterate

core.max_cache_size = 12555

src = nao.src("Sewayaki Kitsune no Senko-san - 01 (FUNimation 1080p).mkv", depth=16)
src = core.std.Trim(src, 240, 34163).std.DuplicateFrames(15582).std.DuplicateFrames(15582)
# src3 = nao.src('Sewayaki no Kitsune Senko-san - 01 (Abema 1080p).mkv', depth=16)
# src = core.fftspectrum.FFTSpectrum(src, True)

descaled = nao.masked_descale(
    src, 1440, 810, "spline16", yuv444=True, expandN=4, masked=True, show_mask=False
)

# @# OP #@#
descale_format = core.register_format(
    descaled.format.color_family, descaled.format.sample_type, descaled.format.bits_per_sample, 0, 0
)

title_card = src[3427 : 3486 + 1]
title_card = core.resize.Bicubic(title_card, 1440, 810, format=descale_format.id)

v = insert_clip(descaled, title_card, 3427)

#################

v = core.neo_f3kdb.Deband(v, 15, 40, 25, 25, 15, 0, output_depth=16)

# @# Dark Area #@#

adapt_mask = kgf.adaptive_grain(v, luma_scaling=24, show_mask=True)
edgemask = iterate(core.std.Sobel(get_y(v)), core.std.Maximum, 1)
adapt_mask = core.std.Expr([adapt_mask, edgemask], "x y -")

more_deband = core.neo_f3kdb.Deband(v, 16, 60, 40, 40, 15, 0, output_depth=16)

v = core.std.MaskedMerge(more_deband, v, adapt_mask)

#################

v = core.resize.Spline36(v, 1280, 720)
v = kgf.adaptive_grain(v, 0.15)

v.set_output()
