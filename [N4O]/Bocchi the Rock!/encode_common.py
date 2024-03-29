from enum import Enum
import logging
import sys
import zlib
from pathlib import Path
from typing import Callable, Optional, overload

import vapoursynth as vs
from lvsfunc import clip_async_render, find_scene_changes
from mvsfunc import ToYUV
from vapoursynth import core
from vardautomation import (
    X265,
    AudioEncoder,
    AudioTrack,
    BitrateMode,
    EztrimCutter,
    FFmpegAudioExtracter,
    FileInfo,
    Lang,
    MatroskaFile,
    OpusEncoder,
    PresetBD,
    PresetOpus,
    RunnerConfig,
    SelfRunner,
    Track,
    VideoTrack,
    VPath,
)
from vsdeband import Placebo, PlaceboDither
from vskernels import Catrom, Kernel
from vsmask.edge import MinMax
from vstools import Matrix, depth, get_prop, get_y, iterate

CURRENT_DIR = VPath(__file__).absolute().parent
PREMUX_DIR = CURRENT_DIR / "Premux"
SOURCE_DIR = CURRENT_DIR / "Source"
PREMUX_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("encode_common")


def dither_10bit(clip: vs.VideoNode) -> vs.VideoNode:
    """The final output video node"""
    return depth(clip, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])


def _open_source_common(
    open_path: Path, output_file: VPath | Path, trims: tuple[Optional[int], Optional[int]] | None = None
) -> FileInfo:
    source = FileInfo(open_path, trims_or_dfs=trims, preset=[PresetBD, PresetOpus])
    source.name_clip_output = VPath(CURRENT_DIR / output_file.stem)
    source.set_name_clip_output_ext(".265")
    return source


def open_source(
    name: str, vol: str, output_file: VPath | Path, trims: tuple[Optional[int], Optional[int]] | None = None
) -> FileInfo:
    if name.endswith(".m2ts"):
        name = name[:-5]
    return _open_source_common(SOURCE_DIR / "BDMV" / f"Vol.{vol}" / f"{name}.m2ts", output_file, trims)


def open_web_source(
    filename: str, output_file: VPath | Path, trims: tuple[Optional[int], Optional[int]] | None = None
) -> FileInfo:
    return _open_source_common(SOURCE_DIR / "Web" / filename, output_file, trims)


def open_image(path: Path | VPath, ref: vs.VideoNode) -> vs.VideoNode:
    img_mask = core.imwri.Read(str(path))
    img_mask = ToYUV(img_mask)
    img_mask = core.resize.Bicubic(img_mask, format=vs.YUV420P16, matrix=1)
    img_mask = depth(img_mask, ref.format.bits_per_sample)
    return core.std.AssumeFPS(img_mask, ref)


def open_endcard(episode: int, ref: vs.VideoNode, endcard_length: int = -1, ext_fmt: str = "m2v"):
    ppath = CURRENT_DIR / "Endcard" / f"Endcard_{episode:02d}.{ext_fmt}"
    clip = core.lsmas.LWLibavSource(str(ppath))
    clip = depth(clip, ref.format.bits_per_sample)
    if endcard_length != -1:
        # If endcard_length is less than the actual length of the endcard, it will be trimmed
        if endcard_length < clip.num_frames:
            clip = clip[:endcard_length]
        elif endcard_length > clip.num_frames:
            ff_final = clip[-1]
            ff_freeze = ff_final * (endcard_length - clip.num_frames)
            clip = clip + ff_freeze
    return clip


def hash_file(file: Path | VPath):
    prev = 0
    for fp in file.open("rb"):
        prev = zlib.crc32(fp, prev)
    return ("%X" % (prev & 0xFFFFFFFF)).upper()


def _premux_exist(output_name: str, extension: str = ".mkv"):
    single_find = PREMUX_DIR / f"{output_name}_premux{extension}"
    if single_find.exists():
        return single_find
    multi_find = list(PREMUX_DIR.glob(f"{output_name}_premux [*{extension}"))
    if not multi_find:
        return None
    return multi_find[0]


def create_keyframes(final: Path, clip: vs.VideoNode | None = None):
    clip = clip or core.lsmas.LWLibavSource(str(final))
    keyframes = find_scene_changes(clip)

    kf_file = final.with_name(f"{final.stem}_keyframes.txt")
    print(f"Saving keyframes to {kf_file}")
    with kf_file.open("w") as f:
        f.write("# keyframe format v1\nfps 0\n")
        for i in keyframes:
            f.write(str(i) + "\n")

    # cleanup LWI
    cwd = Path.cwd()
    if clip is None:
        print(f"Cleaning up LWI files in {cwd} and {CURRENT_DIR}")
        for file in cwd.glob("*.lwi"):
            if final.stem in file.stem:
                file.unlink(missing_ok=True)

        for file in CURRENT_DIR.glob("*.lwi"):
            if final.stem in file.stem:
                file.unlink(missing_ok=True)


def start_encode(source: FileInfo, clip: vs.VideoNode, *, audio_encoder: AudioEncoder | None = None):
    file_final = f"{source.name_clip_output.stem.lower()}_premux.mkv"
    output_final = PREMUX_DIR / file_final
    actual_premux = _premux_exist(source.name_clip_output.stem)
    if actual_premux is not None:
        print(f"Premux file already exists: {actual_premux.name}")
        kf_exist = _premux_exist(source.name_clip_output.stem, ".txt")
        if kf_exist is not None:
            print(f"Keyframes file already exists: {kf_exist.name}")
            return
        create_keyframes(actual_premux)
        return
    mkv_tracks: list[Track] = [
        VideoTrack(
            source.name_clip_output, "Video diolah oleh Suami Nijika-chan | x265 10bit", Lang.make("ja")
        )
    ]
    if source.a_enc_cut is not None:
        mkv_tracks.append(AudioTrack(source.a_enc_cut.set_track(1), "Japanese OPUS 2.0", Lang.make("ja")))
    mkv_meta = MatroskaFile(output_final, mkv_tracks)
    logger.info("Preparing runner...")
    logger.info(f"Output file: {output_final}")
    ENCODER = X265(CURRENT_DIR / "_settings.ini")
    ENCODER.resumable = True
    config = RunnerConfig(
        v_encoder=ENCODER,
        a_extracters=FFmpegAudioExtracter(source, track_in=1, track_out=1),
        a_cutters=EztrimCutter(source, track=1),
        a_encoders=audio_encoder or OpusEncoder(source, track=1, mode=BitrateMode.VBR, bitrate=224),
        mkv=mkv_meta,
    )
    if clip.format.bits_per_sample < 10:
        # Convert to 10 bit
        logger.info("Upsampling to 10 bit")
        clip = depth(clip, 10)
    elif clip.format.bits_per_sample > 10:
        # Convert to 10 bit
        logger.info("Downsampling to 10 bit")
        clip = dither_10bit(clip)
    logger.info("Starting runner...")
    runner = SelfRunner(clip, source, config)
    runner.run()
    logger.info("Runner finished, hashing file...")
    file_hash = hash_file(output_final)
    logger.info(f"Hash: {file_hash}")
    stat_file = output_final.stat()
    file_size = stat_file.st_size
    logger.info(f"Size: {file_size}")
    output_final_f = output_final.with_name(f"{output_final.stem} [{file_hash}].mkv")
    output_final.rename(output_final_f)
    logger.info(f"Runner completed, final file saved to {output_final_f.name}")
    create_keyframes(output_final_f)
    logger.info("Cleaning up work files...")
    runner.work_files.clear()


def generate_cambi_file(source: FileInfo, clip: vs.VideoNode | None = None):
    if not hasattr(sys, "argv"):
        raise RuntimeError("generate_cambi_file: please run this script from a command line")
    stem = source.name_clip_output.stem
    cambi_file = source.name_clip_output.with_name(f"{stem}_cambi.txt")
    if clip is None:
        clip = depth(source.clip_cut, 10)
    if clip.format.color_family != vs.GRAY:
        clip = get_y(clip)
    f = clip[0]
    if get_prop(f, "CAMBI", float | None, default=None) is None:
        clip = core.akarin.Cambi(clip, scores=True)

    cambi_scores: list[tuple[int, float]] = []

    def _cambi_cb(n: int, f: vs.VideoFrame):
        if (cam := get_prop(f, "CAMBI", float | None, default=None)) is not None and cam > 5:
            cambi_scores.append((n, cam))

    clip_async_render(clip, progress="Generating CAMBI scores", callback=_cambi_cb)
    cambi_scores.sort(key=lambda x: x[0])
    with cambi_file.open("w") as f:
        for n, score in cambi_scores:
            f.write(f"{n}: {score}\n")


@overload
def LevelsM(
    clip: vs.VideoNode, points: list[float | int], levels: list[int], xpass=[0, "peak"], return_expr=False
) -> vs.VideoNode:
    ...


@overload
def LevelsM(
    clip: vs.VideoNode, points: list[float | int], levels: list[int], xpass=[0, "peak"], return_expr=True
) -> str:
    ...


def LevelsM(
    clip: vs.VideoNode, points: list[float | int], levels: list[int], xpass=[0, "peak"], return_expr=False
):
    # https://github.com/Moelancholy/Encode-Scripts/blob/master/Pizza/Urusei%20Yatsura%20(2022)/UY22_09.vpy#L74
    qm = len(points)
    peak = [(1 << clip.format.bits_per_sample) - 1, 1][clip.format.sample_type]

    if len(set(xpass)) == 1:
        expr = f"x {points[0]} < x {points[-1]} > or {xpass[0]} "
        qm -= 1
    else:
        expr = f"x {points[0]} < {xpass[0]} x {points[-1]} > {xpass[-1]} "

    for x in range(len(points) - 1):
        if points[x + 1] < points[-1]:
            expr += f" x {points[x+1]} <= "
        if levels[x] == levels[x + 1]:
            expr += f" {peak * levels[x]} "
        else:
            expr += f" x {points[x]} - {peak * (levels[x+1] - levels[x])/(points[x+1] - points[x])} * {peak * levels[x]} + "  # noqa

    for _ in range(qm):
        expr += " ? "

    expr = expr.replace("  ", " ").replace("peak", f"{peak}")

    if return_expr:
        return expr

    return clip.std.Expr(expr)


def texture_mask(clip: vs.VideoNode, range: int) -> vs.VideoNode:
    # https://github.com/Moelancholy/Encode-Scripts/blob/master/Pizza/Urusei%20Yatsura%20(2022)/UY22_09.vpy#L104
    ed_gray = vs.core.std.ShufflePlanes(clip, 0, vs.GRAY)
    rmask = MinMax(range).edgemask(ed_gray, lthr=0, multi=1.00)
    emask = ed_gray.std.Prewitt()
    em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
    em_hi = iterate(em_hi, vs.core.std.Minimum, 5)
    em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
    em_me = iterate(em_me, vs.core.std.Minimum, 4)
    em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
    em_lo = iterate(em_lo, vs.core.std.Minimum, 2)
    rm_txt = vs.core.std.Expr([rmask, em_hi, em_me, em_lo], "x y z a min min min")
    weighted = LevelsM(
        rm_txt,
        points=[x * 256 for x in (1.75, 2.5, 5, 10)],
        levels=[0, 1, 1, 0],
        xpass=[0, 0],
        return_expr=False,
    )

    masked = weighted.std.BoxBlur(hradius=8, vradius=8).std.Expr(f"x {65535 * 0.2} - {1 / (1 - 0.2)} *")
    return masked


def deband_texmask(clip: vs.VideoNode, rady: int = 2, edge: int = 30, edge_dilate: int = 6) -> vs.VideoNode:
    clip_y = get_y(clip)
    tex_mask = texture_mask(clip_y, rady)
    edgem = (
        core.std.Prewitt(clip_y)
        .std.Binarize(edge << 7)
        .std.Deflate()
        .std.Inflate()
        .morpho.Close(size=edge_dilate)
    )
    edgem = core.rgvs.RemoveGrain(edgem, 17)
    return core.std.Expr([tex_mask, edgem], "x y +").rgvs.RemoveGrain(17)


def debandshit(
    clip: vs.VideoNode,
    iterations: int = 1,
    radius: float = 16.0,
    thr: list[float] | float = 4.0,
    grains: list[float] | float = 4.0,
    dither: PlaceboDither = PlaceboDither.DEFAULT,
    rady: int = 2,
    show_mask: bool = False,
    clip_mask: vs.VideoNode | None = None,
):
    tex_mask = clip_mask or deband_texmask(clip, rady)
    if show_mask:
        return tex_mask
    pb = Placebo(radius=radius, thr=thr, grains=grains, dither=dither, iterations=iterations)
    deband = pb.deband(clip)
    return core.std.MaskedMerge(deband, clip, tex_mask)


def dlisr_upscale(clip: vs.VideoNode, scale: int, device_id: int = 0) -> vs.VideoNode:
    """
    A wrapper for AkarinVS's DLISR plugin.

    Automatically convert from source format to RGBS and back source format.

    :param clip: Input clip
    :type clip: vs.VideoNode
    :param scale: Scale factor
    :type scale: int
    :param device_id: Device ID
    :type device_id: int
    """
    if scale not in [2, 4, 8]:
        raise ValueError(f"dlisr_upscale: scale must be 2, 4 or 8, not {scale}")

    matrix = Matrix.from_video(clip, False)
    kernel = Kernel.ensure_obj(Catrom)

    out = kernel.resample(clip, vs.RGBS, Matrix.RGB, matrix)
    out = out.std.Limiter()

    sout = vs.core.akarin.DLISR(out, scale, device_id)
    matrix_in = Matrix.from_video(sout, False)

    sback = kernel.resample(sout, clip.format, matrix, matrix_in)
    return sback


def create_img_mask(img_name: str, ref: vs.VideoNode, length: int = 1, inflate: int = 3):
    mask_path = CURRENT_DIR / "masks" / img_name
    img_mask = open_image(mask_path, ref)
    img_mask = iterate(get_y(img_mask), core.std.Inflate, inflate)
    return img_mask * length


class Boccher(Enum):
    ENCODE = 1
    KEYFRAME = 2
    CAMBI = 3


def boccher_mode() -> Boccher:
    try:
        mode = sys.argv[1]
    except IndexError:
        return Boccher.ENCODE

    mode = mode.lower()
    match mode:
        case "keyframe" | "kf" | "keyframes":
            return Boccher.KEYFRAME
        case "cambi" | "banding":
            return Boccher.CAMBI
        case _:
            return Boccher.ENCODE


def do_boccher(source: FileInfo, filterchain: Callable[[vs.VideoNode, FileInfo], vs.VideoNode]):
    boccher = boccher_mode()

    generate_to = PREMUX_DIR / source.name_clip_output
    match boccher:
        case Boccher.ENCODE:
            print("Making boccher...")
            start_encode(source, filterchain(source.clip_cut, source))
        case Boccher.KEYFRAME:
            print("Generating keyframes")
            filepath = generate_to.with_suffix(".keyframes.txt")
            create_keyframes(filepath, source.clip_cut)
        case Boccher.CAMBI:
            print("Doing CAMBI shit")
            generate_cambi_file(source)
        case _:
            print("Invalid boccher mode")
