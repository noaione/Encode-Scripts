from pathlib import Path
from typing import overload

import n4ofunc as nao
import vapoursynth as vs
from encode_common import deband_texmask, open_source, start_encode
from finedehalo import fine_dehalo
from stgfunc import adaptive_grain
from vapoursynth import core
from vsaa import fine_aa
from vsaa.antialiasers.nnedi3 import Nnedi3SR
from vstools import depth

CURRENT_DIR = Path(__file__).absolute().parent
CURRENT_FILE = Path(__file__)

source = open_source("00000.m2ts", "6", CURRENT_FILE, (24, -24))

RANGES = {
    "ED": [32692, None],
}


@overload
def filterchain(show_mask: bool = False) -> vs.VideoNode:
    ...


@overload
def filterchain(show_mask: bool = False) -> tuple[vs.VideoNode, vs.VideoNode]:
    ...


def filterchain(show_mask: bool = False) -> vs.VideoNode | tuple[vs.VideoNode, vs.VideoNode]:
    # ED = RANGES["ED"]

    # working depth
    src = depth(source.clip_cut, 16)

    # quick and dirty AA
    filt_aa = fine_aa(src, taa=True, singlerater=Nnedi3SR(4, 0, opencl=True))

    # dehalo
    filt_dehalo = fine_dehalo(filt_aa, rx=2, ry=1, darkstr=0.0)

    # medium degrain
    filt_degrain0 = nao.adaptive_smdegrain(
        filt_dehalo, iter_edge=1, thSAD=95, thSADC=0, tr=2, RefineMotion=True
    )
    filt_degrain = nao.adaptive_smdegrain(
        filt_degrain0, iter_edge=1, thSAD=75, thSADC=0, tr=2, area="dark", RefineMotion=True
    )

    # adaptive deband (without fucking up edge and textures)
    tex_mask = deband_texmask(filt_degrain, 1)
    adaptmask_area = core.adg.Mask(filt_degrain.std.PlaneStats(), luma_scaling=5)
    adaptmask_light = core.std.Expr([adaptmask_area.std.Invert(), tex_mask], "x y -")
    adaptmask_dark = core.std.Expr([adaptmask_area, tex_mask], "x y -")
    if show_mask:
        return adaptmask_light, adaptmask_dark
    filt_deband_lite = core.neo_f3kdb.Deband(filt_degrain, 15, 40, 15, 15, 5, 0, output_depth=16)
    filt_deband_dark = core.neo_f3kdb.Deband(filt_degrain, 15, 60, 40, 40, 12, 0, output_depth=16)

    filt_deband = core.std.MaskedMerge(filt_degrain, filt_deband_lite, adaptmask_light)
    filt_deband = core.std.MaskedMerge(filt_deband, filt_deband_dark, adaptmask_dark)

    # regrain
    filt_adgrain = adaptive_grain(filt_deband, strength=0.24, luma_scaling=15)

    return filt_adgrain


if __name__ == "__main__":
    start_encode(source, filterchain())
else:
    # filterchain().text.Text("Filtered").set_output(0)
    nao.debug_clip(source.clip_cut, "Source\nFrame {n} of {total_abs}").set_output(0)
    nao.debug_clip(filterchain(), "Filtered\nFrame {n} of {total_abs}").set_output(1)
    mask_l, mask_d = filterchain(show_mask=True)
    nao.debug_clip(mask_l, "Light Mask\nFrame {n} of {total_abs}").set_output(2)
    nao.debug_clip(mask_d, "Dark Mask\nFrame {n} of {total_abs}").set_output(3)
