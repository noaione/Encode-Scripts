from pathlib import Path
from typing import List

import n4ofunc as nao
import havsfunc as hvf
import vapoursynth as vs
from debandshit import f3kpf
from stgfunc import adaptive_grain
from vapoursynth import core
from vardautomation import X265, FFmpegAudioExtracter, FileInfo, PresetWEB, PresetEAC3, RunnerConfig, SelfRunner, VPath
from vsaa import clamp_aa, transpose_aa, upscaled_sraa, Znedi3SR, Znedi3SS
from vsdehalo import fine_dehalo
from vstools import depth

CURRENT_DIR = Path(__file__).absolute().parent
CURRENT_FILE = VPath(__file__)

source = FileInfo(CURRENT_DIR / "Yuru Camp Movie - 1080p WEB H.264 -NanDesuKa (AMZN).mkv", preset=[PresetWEB, PresetEAC3])
source.name_clip_output = VPath(CURRENT_DIR / CURRENT_FILE.stem)
source.set_name_clip_output_ext(".265")


def dither_down(clip: vs.VideoNode) -> vs.VideoNode:
    """Output video node"""
    return depth(clip, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def filterchain():
    src = depth(source.clip_cut, 16)

    # fucked lineart destroyer
    filt_aa_weak = transpose_aa(src, Znedi3SR())
    filt_aa_str = upscaled_sraa(src, ssfunc=Znedi3SS(nns=2), aafunc=Znedi3SR(nns=1, qual=1))
    filt_aa = clamp_aa(src, filt_aa_weak, filt_aa_str)

    # dehaloing
    filt_dehalo = fine_dehalo(filt_aa)

    # degrain
    filt_degrain = hvf.SMDegrain(filt_dehalo, tr=2, thSAD=100, thSADC=0, RefineMotion=True, contrasharp=False)

    # deband
    filt_deband = f3kpf(filt_degrain, threshold=45, grain=0, f3kdb_args={"use_neo": True})

    # graaaaain
    filt_regrain = adaptive_grain(filt_deband, strength=0.25, luma_scaling=12)
    return filt_regrain


if __name__ == "__main__":
    config = RunnerConfig(
        X265(CURRENT_DIR / "_settings.ini"),
        a_extracters=FFmpegAudioExtracter(source, track_in=1, track_out=1),
        # a_cutters=EztrimCutter(source, track=1),
    )
    SelfRunner(dither_down(filterchain()), source, config).run()
else:
    filterchain().text.Text("Filtered").set_output(0)
    source.clip_cut.text.Text("Source").set_output(1)
