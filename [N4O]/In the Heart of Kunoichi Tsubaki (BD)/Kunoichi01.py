from pathlib import Path
from typing import List

import lvsfunc as lvf
import n4ofunc as nao
import vapoursynth as vs
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
from vardefunc import AddGrain, Graigasm
from vsaa import Eedi3SR, transpose_aa
from vsdehalo import fine_dehalo
from vstools import depth, get_y, iterate

CURRENT_DIR = Path(__file__).absolute().parent
CURRENT_FILE = VPath(__file__)

source = FileInfo(
    CURRENT_DIR / "BDMV" / "Vol.1" / "00005.m2ts", trims_or_dfs=[(0, -26)], preset=[PresetBD, PresetOpus]
)  # noqa
source_nced = FileInfo(
    CURRENT_DIR / "BDMV" / "Vol.1" / "00017.m2ts", trims_or_dfs=[(24, -26)], preset=[PresetBD]
)  # noqa
source.name_clip_output = VPath(CURRENT_DIR / CURRENT_FILE.stem)
source.set_name_clip_output_ext(".265")

RANGES = {"ED": [30903, 32973]}  # 33060


def dither_down(clip: vs.VideoNode) -> vs.VideoNode:
    """Output video node"""
    return depth(clip, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def credit_mask(reference: vs.VideoNode, replace: vs.VideoNode, ranges: List[int]):
    clipped_source = reference[ranges[0] : ranges[1] + 1]
    mask_data = lvf.hardsub_mask(clipped_source, replace, expand=10, inflate=4)
    return mask_data


def splice_credit(ref: vs.VideoNode, nc: vs.VideoNode, ranges: List[int]):
    if nc.format.bits_per_sample != ref.format.bits_per_sample:
        nc = depth(nc, ref.format.bits_per_sample)
    total_l = ranges[1] - ranges[0]
    return ref[: ranges[0]] + nc[: total_l + 1] + ref[ranges[1] + 1 :]


def replace_back_credit(filtered: vs.VideoNode, reference: vs.VideoNode, ranges: List[int]):
    filt_part = filtered[ranges[0] : ranges[1] + 1]
    ref_part = reference[ranges[0] : ranges[1] + 1]
    masking = credit_mask(reference, filt_part, ranges)
    masked = core.std.MaskedMerge(filt_part, ref_part, masking)
    return filtered[: ranges[0]] + masked + filtered[ranges[1] + 1 :]


def filterchain():
    ED = RANGES["ED"]

    src_main = depth(source.clip_cut, 16)
    src_nced = depth(source_nced.clip_cut, 16)
    src = splice_credit(src_main, src_nced, ED)

    # dehalo
    filt_dehalo_bf = fine_dehalo(src, rx=2, brightstr=1.2)

    # weak AA?
    filt_aa = transpose_aa(clip=filt_dehalo_bf, aafunc=Eedi3SR(0.2, 0.25, 100, 2, 20))

    # medium degrain
    filt_degrain = nao.adaptive_smdegrain(filt_aa, iter_edge=1, thSAD=60, thSADC=0, tr=2)

    # adaptive deband (without fucking up edge)
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
            AddGrain(seed=69420, constant=False),
        ],
    ).graining(filt_deband)

    # replace back the credits (maybe I should use rfs?)
    filt_final = replace_back_credit(filt_grain, src_main, ED)

    return filt_final


if __name__ == "__main__":
    config = RunnerConfig(
        X265(CURRENT_DIR / "_settings.ini"),
        a_extracters=FFmpegAudioExtracter(source, track_in=1, track_out=1),
        a_cutters=EztrimCutter(source, track=1),
        a_encoders=OpusEncoder(source, track=1, mode=BitrateMode.VBR, bitrate=192, use_ffmpeg=False),
    )

    SelfRunner(dither_down(filterchain()), source, config).run()
else:
    filterchain().set_output(0)
    # source.clip_cut[RANGES["ED"][1]:].set_output(0)
    # source_nced.clip_cut.set_output(1)
    source.clip_cut.set_output(1)
    # credit_mask(source.clip_cut, source_nced.clip_cut, RANGES["ED"]).set_output(2)
