from pathlib import Path

import n4ofunc as nao
import vapoursynth as vs
from havsfunc import FineDehalo, SMDegrain
from stgfunc import adaptive_grain
from vapoursynth import core
from vsaa import transpose_aa
from vsaa.antialiasers.znedi3 import Znedi3SR
from vstools import depth, insert_clip

from encode_common import (
    create_img_mask,
    deband_texmask,
    open_endcard,
    open_source,
    start_encode,
)

current_file = Path(__file__)

source = open_source("00004.m2ts", "2", current_file, trims=(None, -24))
source_ncop = open_source("00004.m2ts", "0", current_file, trims=(24, -27))
src = depth(source.clip_cut, 16)
ncop = depth(source_ncop.clip_cut, 16)

# first num is the first frame where the title will show up
# end num is before the next frame
op_titles = (4559, 4648)
endcard_start = 33924

# apply title mask
optitle_dur = (op_titles[1] - op_titles[0]) + 1
optitle_ref = src[op_titles[0] : op_titles[1] + 1]
optitle_ncopref = ncop[2040:2130]
optitle_mask = create_img_mask("bocchi04_title.png", src, optitle_dur).std.Binarize().std.BoxBlur(hradius=1, vradius=1)
opfreeze_titlemask = core.std.MaskedMerge(optitle_ref, optitle_ncopref, optitle_mask)
src = insert_clip(src, opfreeze_titlemask, op_titles[0])

# swap out the original endcard with a custom endcard
# the card is made in Aegisub, then exported as PNG
# after that it got slapped to a *close enough* background like the original
# that are taken from next episode which are cleaned out and desaturated.
# then I use premiere pro to make a lossless spinning animation because
# i'm too stupid to do it in Aegisub and I hate math.
src_endcard = open_endcard(5, src, src.num_frames - endcard_start, ext_fmt="mp4")
src = insert_clip(src, src_endcard, endcard_start)

# light AA
filt_aa = transpose_aa(src, Znedi3SR(4, 1))

# dehalo
filt_dh = FineDehalo(filt_aa, rx=2, darkstr=0.0, brightstr=1.0)
# medium denoise
texmask = deband_texmask(src, rady=1, edge=28, edge_dilate=5)
filt_dn1 = SMDegrain(filt_dh, tr=2, thSAD=180, thSADC=0, RefineMotion=True, prefilter=2)
# filt_dn = core.std.MaskedMerge(filt_dn_1, filt_dh, texmask)
filt_dn_s = core.bm3dcuda.BM3D(depth(filt_dh, 32), ref=depth(filt_dn1, 32), sigma=0.88, radius=2).bm3d.VAggregate(
    radius=2
)
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
    # ncop.set_output(0)
    # ncop_credmask.set_output(1)
    _DEFAULT_DEBUG_FORMAT = (
        "Frame {n} of {total_abs} ({total})\n"
        "Picture Type: {fp_pict_type}\n"
        "Resolution: {width}/{height} ({ar})\n"
        "FPS: {fps_num}/{fps_den} ({fps_frac})"
    )
    nao.debug_clip(src, f"Source\n{_DEFAULT_DEBUG_FORMAT}").set_output()
    nao.debug_clip(source.clip_cut, f"Original\n{_DEFAULT_DEBUG_FORMAT}").set_output(1)
    nao.debug_clip(filt_gg).set_output(2)
    # filt_gg.set_output(1)
