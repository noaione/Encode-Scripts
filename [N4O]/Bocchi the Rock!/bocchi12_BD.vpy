from pathlib import Path

import n4ofunc as nao
import vapoursynth as vs
from havsfunc import FineDehalo, SMDegrain
from stgfunc import adaptive_grain
from vapoursynth import core
from vardautomation import FileInfo
from vsaa import transpose_aa
from vsaa.antialiasers.znedi3 import Znedi3SR
from vstools import depth, insert_clip

from encode_common import (
    create_img_mask,
    deband_texmask,
    do_boccher,
    open_image,
    open_source,
    open_web_source,
)

current_file = Path(__file__)
CURRENT_DIR = current_file.absolute().parent
EPISODE_NO = 12

source = open_source("00004.m2ts", "6", current_file, trims=(None, -24))
source_web = open_web_source("[SubsPlease] Bocchi the Rock! - 12 (1080p) [CA5333CB].mkv", current_file)


def _handle_op_title(src: vs.VideoNode, op_titles: tuple[tuple[int, int], tuple[int, int]]):
    crowd_timing, blue_sky_timing = op_titles
    optitle_mask = (
        create_img_mask(f"bocchi{EPISODE_NO:02d}_title.png", src)
        .std.Binarize()
        .std.BoxBlur(hradius=1, vradius=1)
    )

    # --> crowd handler
    crowd_dur = (crowd_timing[1] - crowd_timing[0]) + 1
    crowd_ref = src[crowd_timing[0] : crowd_timing[1] + 1]
    crowd_ff = src[crowd_timing[0]] * crowd_dur
    crowd_mask = optitle_mask * crowd_dur
    crowd_titlemask = core.std.MaskedMerge(crowd_ref, crowd_ff, crowd_mask)
    out_masked = insert_clip(src, crowd_titlemask, crowd_timing[0])
    # <--

    # --> blue sky handler
    bsky_dur = (blue_sky_timing[1] - blue_sky_timing[0]) + 1
    bsky_ref = src[blue_sky_timing[0] : blue_sky_timing[1] + 1]
    bsky_swap = open_image(CURRENT_DIR / "masks" / "bocchi12_titlesky.png", src) * bsky_dur
    bsky_mask = optitle_mask * bsky_dur
    bsky_titlemask = core.std.MaskedMerge(bsky_ref, bsky_swap, bsky_mask)
    out_masked = insert_clip(out_masked, bsky_titlemask, blue_sky_timing[0])
    # <-- blue sky handler

    return out_masked


def boccher_chain(s0rs: vs.VideoNode, _: FileInfo, test_only: bool = False):
    src = depth(s0rs, 16)

    # first num is the first frame where the title will show up
    # end num is before the next frame
    op_titles = ((3296, 3391), (3392, 3465))

    # apply title mask
    src = _handle_op_title(src, op_titles)

    if test_only:
        # Return only the masked title and spliced endcard
        return src

    # light AA
    filt_aa = transpose_aa(src, Znedi3SR(4, 1))

    # dehalo
    filt_dh = FineDehalo(filt_aa, rx=2, darkstr=0.0, brightstr=1.0)
    # medium denoise
    texmask = deband_texmask(src, rady=1, edge=28, edge_dilate=5)
    filt_dn1 = SMDegrain(filt_dh, tr=2, thSAD=180, thSADC=0, RefineMotion=True, prefilter=2)
    # filt_dn = core.std.MaskedMerge(filt_dn_1, filt_dh, texmask)
    filt_dn_s = core.bm3dcuda.BM3D(
        depth(filt_dh, 32), ref=depth(filt_dn1, 32), sigma=0.88, radius=2
    ).bm3d.VAggregate(radius=2)
    filt_dn = core.std.MaskedMerge(filt_dn_s, filt_dh, texmask)

    # deband
    filt_db_m = core.neo_f3kdb.Deband(filt_dn, range=18, y=36, cb=30, cr=30, grainy=0, grainc=0)
    filt_db_m = adaptive_grain(filt_db_m, strength=0.07, static=True, luma_scaling=7, sharp=75, size=1.2)
    filt_db_m = adaptive_grain(filt_db_m, strength=0.04, static=False, luma_scaling=7, sharp=75, size=1.2)
    filt_db = core.std.MaskedMerge(filt_db_m, filt_dn, texmask)

    # more grain
    filt_gg = adaptive_grain(filt_db, strength=0.22, luma_scaling=10, sharp=75, size=1.2)
    return filt_gg


if __name__ == "__main__":
    do_boccher(source, boccher_chain)
else:
    # ncop.set_output(0)
    # ncop_credmask.set_output(1)
    _DEFAULT_DEBUG_FORMAT = (
        "Frame {n} of {total_abs} ({total})\n"
        "Picture Type: {fp_pict_type}\n"
        "Resolution: {width}/{height} ({ar})\n"
        "FPS: {fps_num}/{fps_den} ({fps_frac})"
    )
    clippy = [
        ("Source", source.clip_cut),
        ("Source (WEB)", source_web.clip_cut),
        ("Pre-Filtered", boccher_chain(source.clip_cut, source, True)),
    ]  # , ("Original", source.clip_cut)]
    for i, (name, clip) in enumerate(clippy):
        nao.debug_clip(clip, f"{name}\n{_DEFAULT_DEBUG_FORMAT}").set_output(i)
    # filt_gg.set_output(1)
