from pathlib import Path

import n4ofunc as nao
import vapoursynth as vs
from vapoursynth import core
from havsfunc import FineDehalo, SMDegrain
from stgfunc import adaptive_grain
from finedehalo import fine_dehalo
from encode_common import open_source, deband_texmask, create_img_mask, start_encode
from vsaa import Znedi3SR, transpose_aa
from vstools import depth, insert_clip, iterate

current_file = Path(__file__)

source = open_source("00003.m2ts", "1", current_file, trims=(None, -26))
src = depth(source.clip_cut, 16)

# First num is the fade start
# end num is before the next frame
op_titles = (6237, 6302)

# apply title mask
optitle_dur = (op_titles[1] - op_titles[0]) + 1
optitle_ref = src[op_titles[0]:op_titles[1] + 1]
optitle_mask = create_img_mask("bocchi01_title.png", src, optitle_dur).std.Binarize().std.BoxBlur(hradius=1, vradius=1)
opfreeze_title = src[op_titles[0] - 1] * optitle_dur
opfreeze_titlemask = core.std.MaskedMerge(optitle_ref, opfreeze_title, optitle_mask)
src = insert_clip(src, opfreeze_titlemask, op_titles[0])

# light AA
filt_aa = transpose_aa(src, Znedi3SR(4, 1))

# dehalo
filt_dh = FineDehalo(filt_aa, rx=2, darkstr=0.0, brightstr=1.0)
# medium denoise
texmask = deband_texmask(src, rady=1, edge=28, edge_dilate=5)
filt_dn1 = SMDegrain(filt_dh, tr=2, thSAD=180, thSADC=0, RefineMotion=True, prefilter=2)
# filt_dn = core.std.MaskedMerge(filt_dn_1, filt_dh, texmask)
filt_dn_s = core.bm3dcuda.BM3D(depth(filt_dh, 32), ref=depth(filt_dn1, 32), sigma=0.88, radius=2).bm3d.VAggregate(radius=2)
filt_dn = core.std.MaskedMerge(filt_dn_s, filt_dh, texmask)

# deband
filt_db_m = core.neo_f3kdb.Deband(filt_dn, range=18, y=36, cb=30, cr=30, grainy=0, grainc=0)
filt_db_m = adaptive_grain(filt_db_m, strength=0.07, static=True, luma_scaling=7, sharp=75, size=1.2)
filt_db_m = adaptive_grain(filt_db_m, strength=0.04, static=False, luma_scaling=7, sharp=75, size=1.2)
filt_db = core.std.MaskedMerge(filt_db_m, filt_dn, texmask)

# more grain
filt_gg = adaptive_grain(filt_db, strength=0.22, luma_scaling=10, sharp=75, size=1.2)

if __name__ == "__main__":
    start_encode(source, filt_gg)
else:
    src.set_output(0)
    filt_gg.set_output(1)