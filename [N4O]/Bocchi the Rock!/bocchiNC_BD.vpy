from pathlib import Path

import n4ofunc as nao
import vapoursynth as vs
from havsfunc import FineDehalo, SMDegrain
from stgfunc import adaptive_grain
from vapoursynth import core
from vardautomation import FileInfo
from vsaa import Znedi3SR, transpose_aa
from vstools import depth, get_y, join, split

from encode_common import (
    deband_texmask,
    open_source,
    start_encode,
)

current_file = Path(__file__)
source_ncop = open_source("00004.m2ts", "0", current_file.with_stem("bocchiNCOP_BD"), trims=(24, -24))
source_nced1 = open_source("00007.m2ts", "0", current_file.with_stem("bocchiNCED1_BD"), trims=(24, -24))
source_nced2 = open_source("00008.m2ts", "0", current_file.with_stem("bocchiNCED2_BD"), trims=(24, -24))
source_nced3 = open_source("00003.m2ts", "0", current_file.with_stem("bocchiNCED3_BD"), trims=(24, -24))


def filterchain(source: FileInfo) -> vs.VideoNode:
    src = depth(source.clip_cut, 16)

    # light AA
    filt_aa = transpose_aa(src, Znedi3SR(4, 1))

    # dehalo
    filt_dh = FineDehalo(filt_aa, rx=2, darkstr=0.0, brightstr=1.0)
    # medium denoise
    texmask = deband_texmask(src, rady=1, edge=28, edge_dilate=5)
    filt_yy, filt_uu, filt_vv = split(filt_dh)
    filt_dny1 = SMDegrain(filt_yy, tr=2, thSAD=180, thSADC=0, RefineMotion=True, prefilter=2)
    # filt_dn = core.std.MaskedMerge(filt_dn_1, filt_dh, texmask)
    filt_dny_s = core.bm3dcuda.BM3D(
        depth(filt_yy, 32), ref=depth(filt_dny1, 32), sigma=0.88, radius=2
    ).bm3d.VAggregate(radius=2)
    filt_dny = core.std.MaskedMerge(filt_dny_s, get_y(filt_dh), texmask)
    # Weakly denoise chroma
    filt_dnuv: list[vs.VideoNode] = []
    for filt_uv in (filt_uu, filt_vv):
        filt_dnuv.append(SMDegrain(filt_uv, tr=2, thSAD=80, thSADC=0, RefineMotion=True, prefilter=2))
    # Merge back luma and chroma
    filt_dn = join([filt_dny, *filt_dnuv], family=filt_dh.format.color_family)

    # deband
    filt_db_m = core.neo_f3kdb.Deband(filt_dn, range=18, y=36, cb=30, cr=30, grainy=0, grainc=0)
    filt_db_m = adaptive_grain(filt_db_m, strength=0.07, static=True, luma_scaling=7, sharp=75, size=1.2)
    filt_db_m = adaptive_grain(filt_db_m, strength=0.04, static=False, luma_scaling=7, sharp=75, size=1.2)
    filt_db = core.std.MaskedMerge(filt_db_m, filt_dn, texmask)

    # more grain
    filt_gg = adaptive_grain(filt_db, strength=0.22, luma_scaling=10, sharp=75, size=1.2)
    return filt_gg


if __name__ == "__main__":
    import sys

    try:
        encode_thing = sys.argv[1]
    except IndexError:
        print("Usage: python bocchiNC_BD.vpy [op|ed1|ed2|ed3]")
        sys.exit(1)

    encode_thing = encode_thing.lower()
    match encode_thing:
        case "op":
            pref = source_ncop
        case "ed1":
            pref = source_nced1
        case "ed2":
            pref = source_nced2
        case "ed3":
            pref = source_nced3
        case _:
            print("Usage: python bocchiNC_BD.vpy [op|ed1|ed2|ed3]")
            sys.exit(1)

    start_encode(pref, filterchain(pref))
else:
    # ncop.set_output(0)
    # ncop_credmask.set_output(1)
    DEFAULT_DEBUG = (
        "Frame {n} of {total_abs} ({total})\n"
        "Picture Type: {fp_pict_type}\n"
        "Resolution: {width}/{height} ({ar})\n"
        "FPS: {fps_num}/{fps_den} ({fps_frac})"
    )
    nao.debug_clip(source_ncop.clip_cut, f"NCOP\n{DEFAULT_DEBUG}").set_output()
    nao.debug_clip(source_nced1.clip_cut, f"NCED1\n{DEFAULT_DEBUG}").set_output(1)
    nao.debug_clip(source_nced2.clip_cut, f"NCED2\n{DEFAULT_DEBUG}").set_output(2)
    nao.debug_clip(source_nced3.clip_cut, f"NCED3\n{DEFAULT_DEBUG}").set_output(3)
    # filt_gg.set_output(1)
