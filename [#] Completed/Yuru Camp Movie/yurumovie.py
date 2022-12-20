from pathlib import Path

import havsfunc as hvf
import n4ofunc as nao
import vapoursynth as vs
from stgfunc import adaptive_grain
from vapoursynth import core
from vardautomation import (
    X265,
    FFmpegAudioExtracter,
    FileInfo,
    PresetEAC3,
    PresetWEB,
    RunnerConfig,
    SelfRunner,
    VPath,
)
from vsaa import Znedi3SR, Znedi3SS, clamp_aa, transpose_aa, upscaled_sraa
from vsdeband.f3kdb import F3kdb
from vsdehalo import fine_dehalo
from vsmask.edge import FreyChen, SobelStd
from vstools import depth, get_y

CURRENT_DIR = Path(__file__).absolute().parent
CURRENT_FILE = VPath(__file__)

VERSION = "v2"
source_amzn = FileInfo(
    CURRENT_DIR / "Yuru Camp Movie - 1080p WEB H.264 -NanDesuKa (AMZN).mkv", preset=[PresetWEB, PresetEAC3]
)  # noqa
source_cr = FileInfo(
    CURRENT_DIR / "Laid-Back Camp The Movie (2022) VOSTFR 1080p WEB x264 AAC -Tsundere-Raws (CR).mkv",
    preset=[PresetWEB],
)  # noqa
source_cr.name_clip_output = VPath(CURRENT_DIR / (CURRENT_FILE.stem + VERSION))
source_cr.set_name_clip_output_ext(".265")

do_not_sharpen = [(0, 368)]


def dither_down(clip: vs.VideoNode) -> vs.VideoNode:
    """Output video node"""
    return depth(clip, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def merge_source(show_mask: bool = False):
    src_amzn = depth(source_amzn.clip_cut, 16)
    src_cr = depth(source_cr.clip_cut, 16)

    # https://github.com/Moelancholy/Encode-Scripts/blob/master/Pizza/Do%20it%20Yourself/DIY_WEB05.vpy#L29
    mask = FreyChen().edgemask(get_y(src_cr)).std.Binarize(25 << 7).std.Convolution([1] * 9)
    if show_mask:
        return mask
    return core.std.MaskedMerge(src_amzn, src_cr, mask)


def swap_frames(srca: vs.VideoNode, srcb: vs.VideoNode, frames: tuple[int, int]):
    sf, ef = frames
    cut_b = srcb[sf : ef + 1]
    if sf == 0:
        return cut_b + srca[ef + 1 :]
    elif ef == (srca.num_frames - 1):
        return srca[: sf + 1] + cut_b
    else:
        return srca[:sf] + cut_b + srca[ef:]


def filterchain(skip_resharp: bool = False):
    src = merge_source()

    # fucked lineart destroyer
    filt_aa_weak = transpose_aa(src, Znedi3SR(nns=1))
    filt_aa_str = upscaled_sraa(src, ssfunc=Znedi3SS(nns=2), aafunc=Znedi3SR(nns=1, qual=1))
    filt_aa = clamp_aa(src, filt_aa_weak, filt_aa_str)

    # line resharpening (will not restore stuff that got AA'd too much)
    if not skip_resharp:
        asharp_mask = SobelStd().edgemask(get_y(src)).std.Binarize(25 << 7).std.Convolution([1] * 9)
        filt_asharp = core.asharp.ASharp(filt_aa, t=0.5, d=1.5, hqbf=True)
        filt_ldark_t = hvf.FastLineDarkenMOD(filt_asharp, strength=20, protection=24)
        filt_ldark = core.std.MaskedMerge(filt_aa, filt_ldark_t, asharp_mask)
        for frames in do_not_sharpen:
            filt_ldark = swap_frames(filt_ldark, filt_aa, frames)
    else:
        filt_ldark = filt_aa

    # dehaloing
    filt_dehalo = fine_dehalo(filt_ldark, darkstr=0.6)

    # degrain
    filt_degrain = hvf.SMDegrain(filt_dehalo, tr=2, thSAD=100, thSADC=0, RefineMotion=True, contrasharp=False)

    # deband
    debander = F3kdb(thr=45, grains=0, use_neo=True)
    filt_deband = debander.deband(filt_degrain)

    # graaaaain
    filt_regrain = adaptive_grain(filt_deband, strength=0.25, luma_scaling=12)
    return filt_regrain


def show_info(src: vs.VideoNode, text: str):
    return nao.debug_clip(src, text)


if __name__ == "__main__":
    config = RunnerConfig(
        X265(CURRENT_DIR / "_settings.ini"),
        a_extracters=FFmpegAudioExtracter(source_amzn, track_in=1, track_out=1),
        # a_cutters=EztrimCutter(source, track=1),
    )
    SelfRunner(dither_down(filterchain()), source_cr, config).run()
else:
    show_info(source_cr.clip_cut, "Source (CR)").set_output(0)
    show_info(source_amzn.clip_cut, "Source (AMZN)").set_output(1)
    show_info(merge_source(), "Source (Merged)").set_output(2)
    # show_info(merge_source(show_mask=True), "Source (Merged)").set_output(3)
    show_info(filterchain(True), "Filtered").set_output(4)
    show_info(filterchain(), "Filtered + Resharp").set_output(5)
