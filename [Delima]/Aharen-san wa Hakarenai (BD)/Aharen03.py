from pathlib import Path

import n4ofunc as nao
import vapoursynth as vs
from kagefunc import adaptive_grain
from vapoursynth import core
from vardautomation import (
    X265,
    BasicTool,
    BinaryPath,
    BitrateMode,
    EztrimCutter,
    FFmpegAudioExtracter,
    FileInfo,
    FlacCompressionLevel,
    FlacEncoder,
    MatroskaFile,
    OpusEncoder,
    PresetBD,
    PresetOpus,
    RunnerConfig,
    SelfRunner,
    VPath,
)
from vardautomation.vpathlib import CleanupSet
from vsaa import Eedi3SR, clamp_aa, transpose_aa, upscaled_sraa
from vstools import depth, get_y, iterate

CURRENT_DIR = Path(__file__).absolute().parent
CURRENT_FILE = VPath(__file__)

source = FileInfo(
    CURRENT_DIR / "BDMV" / "Vol.1" / "00002.m2ts", trims_or_dfs=[(0, -24)], preset=[PresetBD, PresetOpus]
)
source.name_clip_output = VPath(CURRENT_DIR / CURRENT_FILE.stem)
source.name_file_final = VPath(CURRENT_DIR / f"{CURRENT_FILE.stem}_premux.mp4")
source.set_name_clip_output_ext(".265")


def dither_down(clip: vs.VideoNode) -> vs.VideoNode:
    """Output video node"""
    return depth(clip, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def filterchain():
    src = depth(source.clip_cut, 16)

    # medium aa
    # filt_straa = upscaled_sraa(src, aafunc=Eedi3SR(mdis=40))
    filt_weakaa = transpose_aa(src, aafunc=Eedi3SR(nrad=2, gamma=60))
    # filt_aaclamp = clamp_aa(src, filt_weakaa, filt_straa)

    # medium degrain
    filt_degrain = nao.adaptive_smdegrain(filt_weakaa, iter_edge=2, thSAD=60, thSADC=0, tr=2)

    # adaptive deband (without fucking up edge)
    sobel_edge = iterate(core.std.Sobel(get_y(filt_degrain)), core.std.Inflate, 2)
    adaptmask_area = core.adg.Mask(filt_degrain.std.PlaneStats(), luma_scaling=5)
    adaptmask_light = core.std.Expr([adaptmask_area.std.Invert(), sobel_edge], "x y -")
    adaptmask_dark = core.std.Expr([adaptmask_area, sobel_edge], "x y -")
    filt_deband_lite = core.neo_f3kdb.Deband(filt_degrain, 15, 40, 15, 15, 5, 0, output_depth=16)
    filt_deband_dark = core.neo_f3kdb.Deband(filt_degrain, 15, 60, 40, 40, 12, 0, output_depth=16)

    filt_deband = core.std.MaskedMerge(filt_degrain, filt_deband_lite, adaptmask_light)
    filt_deband = core.std.MaskedMerge(filt_deband, filt_deband_dark, adaptmask_dark)

    # adaptive grain (me lazy)
    filt_adgrain = adaptive_grain(filt_deband, 0.2, luma_scaling=30)

    return filt_adgrain


class FFMPegMatroska(MatroskaFile):
    @property
    def command(self) -> list[str]:
        cmds: list[str] = []
        for track in self._tracks:
            cmds.extend(["-i", track.path.to_str()])
        cmds.extend(["-c", "copy"])
        cmds.append(self._output.to_str())
        return cmds

    def mux(self, return_workfiles: bool = True) -> CleanupSet | None:
        """
        Launch a merge command

        :return:        Return worksfiles if True
        """
        BasicTool(BinaryPath.ffmpeg, self.command).run()

        if return_workfiles:
            return CleanupSet(t.path for t in self._tracks)
        return None


if __name__ == "__main__":
    config = RunnerConfig(
        X265(CURRENT_DIR / "_settings.ini"),
        a_extracters=FFmpegAudioExtracter(source, track_in=1, track_out=1),
        a_cutters=EztrimCutter(source, track=1),
        a_encoders=OpusEncoder(source, track=1, mode=BitrateMode.VBR, bitrate=192, use_ffmpeg=False),
        mkv=FFMPegMatroska.autotrack(source),
    )

    SelfRunner(dither_down(filterchain()), source, config).run()
else:
    source.clip_cut.set_output(0)
    filterchain().set_output(1)
