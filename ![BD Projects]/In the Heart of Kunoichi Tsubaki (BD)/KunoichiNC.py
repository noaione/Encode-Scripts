from pathlib import Path
from typing import List

import lvsfunc as lvf
import n4ofunc as nao
import vapoursynth as vs
from vapoursynth import core
from vardautomation import X265, BitrateMode, EztrimCutter, FFmpegAudioExtracter, FileInfo, FlacCompressionLevel, FlacEncoder, OpusEncoder, PresetBD, RunnerConfig, SelfRunner, VPath
from vardefunc import AddGrain, Graigasm
from vsaa import Eedi3SR, transpose_aa
from vsdehalo import fine_dehalo
from vstools import depth, get_y, iterate

CURRENT_DIR = Path(__file__).absolute().parent
CURRENT_FILE = VPath(__file__)

source_ncop = FileInfo(CURRENT_DIR / "BDMV"  / "Vol.1" / "00016.m2ts", trims_or_dfs=[(0, -24)], preset=[PresetBD])
source_nced1 = FileInfo(CURRENT_DIR / "BDMV"  / "Vol.1" / "00017.m2ts", trims_or_dfs=[(24, -24)], preset=[PresetBD])
source_nced2 = FileInfo(CURRENT_DIR / "BDMV"  / "Vol.1" / "00018.m2ts", trims_or_dfs=[(24, -24)], preset=[PresetBD])
source_ncop.name_file_final = VPath(CURRENT_DIR / "KunoichiNCOP")
source_nced1.name_file_final = VPath(CURRENT_DIR / "KunoichiNCED1")
source_nced2.name_file_final = VPath(CURRENT_DIR / "KunoichiNCED2")
source_ncop.set_name_clip_output_ext(".265")
source_nced1.set_name_clip_output_ext(".265")
source_nced2.set_name_clip_output_ext(".265")


def dither_down(clip: vs.VideoNode) -> vs.VideoNode:
    """Output video node"""
    return depth(clip, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def credit_mask(reference: vs.VideoNode, replace: vs.VideoNode, ranges: List[int]):
    clipped_source = reference[ranges[0]:ranges[1] + 1]
    mask_data = lvf.hardsub_mask(clipped_source, replace, expand=10, inflate=4)
    return mask_data


def splice_credit(ref: vs.VideoNode, nc: vs.VideoNode, ranges: List[int]):
    if nc.format.bits_per_sample != ref.format.bits_per_sample:
        nc = depth(nc, ref.format.bits_per_sample)
    return ref[:ranges[0]] + nc + ref[ranges[1] + 1:]


def replace_back_credit(filtered: vs.VideoNode, reference: vs.VideoNode, ranges: List[int]):
    filt_part = filtered[ranges[0]:ranges[1] + 1]
    ref_part = reference[ranges[0]:ranges[1] + 1]
    masking = credit_mask(reference, filt_part, ranges)
    masked = core.std.MaskedMerge(filt_part, ref_part, masking)
    return filtered[:ranges[0]] + masked + filtered[ranges[1] + 1:]


def filterchain(source: FileInfo):
    # working depth
    src = depth(source.clip_cut, 16)
    # weak AA
    filt_aa = transpose_aa(clip=src, aafunc=Eedi3SR(0.2, 0.25, 100, 2, 20))

    # dehalo
    filt_dehalo = fine_dehalo(filt_aa, rx=4)

    # medium degrain
    filt_degrain = nao.adaptive_smdegrain(filt_dehalo, iter_edge=1, thSAD=60, thSADC=0, tr=2)

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
            AddGrain(seed=69420, constant=False)
        ]).graining(filt_deband)

    return filt_grain



def run_source(target: FileInfo):
    config = RunnerConfig(
        X265(CURRENT_DIR / "_settings.ini"),
        a_extracters=FFmpegAudioExtracter(target, track_in=1, track_out=1),
        a_cutters=EztrimCutter(target, track=1),
        a_encoders=[
            OpusEncoder(target, track=1, mode=BitrateMode.VBR, bitrate=192, use_ffmpeg=False),
            FlacEncoder(target, track=1, level=FlacCompressionLevel.VARDOU),
        ],
    )

    SelfRunner(dither_down(filterchain(target)), target, config).run()



if __name__ == "__main__":
    run_source(source_ncop)
    run_source(source_nced1)
    run_source(source_nced2)
else:
    source_ncop.clip_cut.set_output(0)
    source_nced1.clip_cut.set_output(1)
    source_nced2.clip_cut.set_output(2)
