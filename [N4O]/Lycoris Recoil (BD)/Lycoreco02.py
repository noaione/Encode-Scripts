from pathlib import Path

import n4ofunc as nao
import vapoursynth as vs
from stgfunc import adaptive_grain
from vapoursynth import core
from vardautomation import (
    X265,
    BitrateMode,
    EztrimCutter,
    FFmpegAudioExtracter,
    FileInfo,
    OpusEncoder,
    PresetBD,
    PresetOpus,
    RunnerConfig,
    SelfRunner,
    VPath,
)
from vsaa import Znedi3SR, fine_aa
from vsdehalo import fine_dehalo
from vstools import depth, get_y, iterate

CURRENT_DIR = Path(__file__).absolute().parent
CURRENT_FILE = VPath(__file__)

source = FileInfo(
    CURRENT_DIR / "BDMV" / "Vol.1" / "00001.m2ts", trims_or_dfs=[(24, -24)], preset=[PresetBD, PresetOpus]
)  # noqa
source.name_clip_output = VPath(CURRENT_DIR / CURRENT_FILE.stem)
source.set_name_clip_output_ext(".265")

RANGES = {
    "ED": [32692, None],
}


def dither_down(clip: vs.VideoNode) -> vs.VideoNode:
    """Output video node"""
    return depth(clip, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def filterchain() -> vs.VideoNode:
    # ED = RANGES["ED"]

    # working depth
    src = depth(source.clip_cut, 16)

    # quick and dirty AA
    filt_aa = fine_aa(src, taa=True, singlerater=Znedi3SR(4, 0))

    # dehalo
    filt_dehalo = fine_dehalo(filt_aa, rx=2, ry=1)

    # medium degrain
    filt_degrain0 = nao.adaptive_smdegrain(
        filt_dehalo, iter_edge=1, thSAD=80, thSADC=0, tr=2, RefineMotion=True
    )
    filt_degrain = nao.adaptive_smdegrain(
        filt_degrain0, iter_edge=1, thSAD=60, thSADC=0, tr=2, area="dark", RefineMotion=True
    )

    # adaptive deband (without fucking up edge)
    sobel_edge = iterate(core.std.Sobel(get_y(filt_degrain)), core.std.Inflate, 2)
    adaptmask_area = core.adg.Mask(filt_degrain.std.PlaneStats(), luma_scaling=5)
    adaptmask_light = core.std.Expr([adaptmask_area.std.Invert(), sobel_edge], "x y -")
    adaptmask_dark = core.std.Expr([adaptmask_area, sobel_edge], "x y -")
    filt_deband_lite = core.neo_f3kdb.Deband(filt_degrain, 15, 40, 15, 15, 5, 0, output_depth=16)
    filt_deband_dark = core.neo_f3kdb.Deband(filt_degrain, 15, 60, 40, 40, 12, 0, output_depth=16)

    filt_deband = core.std.MaskedMerge(filt_degrain, filt_deband_lite, adaptmask_light)
    filt_deband = core.std.MaskedMerge(filt_deband, filt_deband_dark, adaptmask_dark)

    # regrain
    filt_adgrain = adaptive_grain(filt_deband, strength=0.22, luma_scaling=15)

    return filt_adgrain


if __name__ == "__main__":
    config = RunnerConfig(
        X265(CURRENT_DIR / "_settings.ini"),
        a_extracters=FFmpegAudioExtracter(source, track_in=1, track_out=1),
        a_cutters=EztrimCutter(source, track=1),
        a_encoders=OpusEncoder(source, track=1, mode=BitrateMode.VBR, bitrate=224, use_ffmpeg=False),
    )
    SelfRunner(dither_down(filterchain()), source, config).run()
else:
    filterchain().text.Text("Filtered").set_output(0)
    source.clip_cut.text.Text("Source").set_output(1)
