# Script version 2

from pathlib import Path

import n4ofunc as nao
import vapoursynth as vs
from vapoursynth import core
from vardautomation import X265, FileInfo, PresetBD, PresetOpus
from vardefunc import AddGrain, Graigasm
from vsaa import Eedi3SR, transpose_aa
from vsdehalo import fine_dehalo
from vstools import depth, get_y, iterate

CURRENT_DIR = Path(__file__).absolute().parent


source = FileInfo(CURRENT_DIR / "BDMV" / "Vol.1" / "00005.m2ts", trims_or_dfs=[(0, -26)], preset=[PresetBD, PresetOpus])


def dither_down(clip: vs.VideoNode) -> vs.VideoNode:
    """Output video node"""
    return depth(clip, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def filterchain():
    src = depth(source.clip_cut, 16)

    # weak AA
    filt_aa = transpose_aa(clip=src, aafunc=Eedi3SR(0.2, 0.25, 100, 2, 20))

    # dehalo
    filt_dehalo = fine_dehalo(filt_aa, rx=4)

    # medium? degrain
    filt_degrain = nao.adaptive_smdegrain(filt_dehalo, iter_edge=1, thSAD=60, thSADC=0, tr=2)

    # deband
    sobel_edge = iterate(core.std.Sobel(get_y(filt_degrain)), core.std.Inflate, 2)
    adaptmask_area = core.adg.Mask(filt_degrain.std.PlaneStats(), luma_scaling=5)
    adaptmask_light = core.std.Expr([adaptmask_area.std.Invert(), sobel_edge], "x y -")
    adaptmask_dark = core.std.Expr([adaptmask_area, sobel_edge], "x y -")
    filt_deband_lite = core.neo_f3kdb.Deband(filt_degrain, 15, 40, 15, 15, 5, 0, output_depth=16)
    filt_deband_dark = core.neo_f3kdb.Deband(filt_degrain, 15, 60, 40, 40, 12, 0, output_depth=16)

    filt_deband = core.std.MaskedMerge(filt_degrain, filt_deband_lite, adaptmask_light)
    filt_deband = core.std.MaskedMerge(filt_deband, filt_deband_dark, adaptmask_dark)

    # who loves grain? (stolen from Light's Yuru Camp S2 encode)
    filt_grain = Graigasm(
        thrs=[x << 8 for x in (32, 80, 128, 176)],
        strengths=[(0.25, 0.0), (0.20, 0.0), (0.15, 0.0), (0.0, 0.0)],
        sizes=(1.20, 1.15, 1.10, 1),
        sharps=(80, 70, 60, 50),
        grainers=[
            AddGrain(seed=69420, constant=True),
            AddGrain(seed=69420, constant=False),
            AddGrain(seed=69420, constant=False)
        ]).graining(filt_deband)

    return filt_grain



if __name__ == "__main__":
    FILTER = filterchain()
    X265(CURRENT_DIR / "_settings.ini").run_enc(dither_down(FILTER), source)
    # Handle audio later, me lazy
else:
    filterchain().set_output(0)
    source.clip_cut.set_output(1)
