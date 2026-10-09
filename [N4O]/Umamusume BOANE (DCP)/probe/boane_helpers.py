from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from vapoursynth import VideoNode
from vstools import change_fps, core, initialize_clip, vs

CUR_DIR = Path(__file__).resolve().parent


def frame_to_rgb(frame: vs.VideoFrame) -> np.ndarray:
    return np.stack(
        [np.asarray(frame[p]) for p in range(3)],
        axis=-1,
    ).astype(np.float32)


def apply_color_matrix(
    clip: vs.VideoNode,
    matrix: list[list[float]],
    bias: list[float],
) -> vs.VideoNode:
    if clip.format.id != vs.RGBS:
        raise ValueError("Expected RGBS")

    r = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    g = core.std.ShufflePlanes(clip, 1, vs.GRAY)
    b = core.std.ShufflePlanes(clip, 2, vs.GRAY)

    channels = []

    for row, offset in zip(matrix, bias):
        expr = (
            f"x {row[0]} * "
            f"y {row[1]} * + "
            f"z {row[2]} * + "
            f"{offset} +"
        )

        result = core.std.Expr([r, g, b], expr=[expr])
        channels.append(result)

    return core.std.ShufflePlanes(
        channels,
        planes=[0, 0, 0],
        colorfamily=vs.RGB,
    )


def read_sample(kind: str, shift_frame: int = 0) -> tuple[VideoNode, VideoNode]:
    bd_clip = core.bs.VideoSource(f"boane-bd-sample-{kind}.mkv")
    dcp_clip = core.bs.VideoSource(f"boane-dcp-sample-{kind}.mxf")

    # Blu-ray: YUV420P8 Rec.709 -> RGB float32 Rec.709
    bd_clip = initialize_clip(bd_clip, bits=8)

    # DCP: 12-bit XYZ -> RGB float32 Rec.709
    dcp_clip = core.resize.Bicubic(
        dcp_clip,
        format=vs.RGBS,
        matrix_in_s="rgb",
        transfer_in_s="st428",
        primaries_in_s="xyz",
        range_in_s="full",
        transfer_s="709",
        primaries_s="709",
        chromatic_adaptation=True,
    )

    for name, clip in [("BD", bd_clip), ("DCP", dcp_clip)]:
        print(f"{name}: {clip}")
        frame = clip.get_frame(0)
        print(f"{name} props: {dict(frame.props)}")

    # crop bluray to cinema scope
    bd_clip = bd_clip.std.CropRel(top=138, bottom=138)

    # fix fps
    bd_clip = core.std.AssumeFPS(bd_clip, fpsnum=24, fpsden=1)
    dcp_clip = change_fps(dcp_clip, Fraction(24, 1))

    # dcp_rgb is already RGBS, nonlinear Rec.709
    dcp_clip = core.std.SetFrameProps(
        dcp_clip,
        _ColorRange=0,  # Full range
        _Matrix=0,      # RGB / identity
        _Transfer=1,    # Rec.709
        _Primaries=1,   # Rec.709
    )

    # cut the frame of BD for start, and DCP at last
    if shift_frame > 0:
        bd_clip = bd_clip[shift_frame:]
        dcp_clip = dcp_clip[:-shift_frame]

    return bd_clip, dcp_clip


def collect_samples(
    dcp: vs.VideoNode,
    bd: vs.VideoNode,
    frames: list[int],
    samples_per_frame: int = 10000,
    seed: int = 2406755786128298521,
) -> tuple[np.ndarray, np.ndarray]:
    if dcp.format.id != vs.RGBS or bd.format.id != vs.RGBS:
        raise ValueError("Both clips must be RGBS")

    if dcp.width != bd.width or dcp.height != bd.height:
        raise ValueError("Clips must have identical dimensions")

    rng = np.random.default_rng(seed)
    source_samples = []
    reference_samples = []
    total_frame = len(frames)

    for idx, n in enumerate(frames):
        print(f"Collecting samples from frame {idx + 1}/{total_frame}")
        source = frame_to_rgb(dcp.get_frame(n)).reshape(-1, 3)
        reference = frame_to_rgb(bd.get_frame(n)).reshape(-1, 3)

        # Identical pixel positions in both frames.
        indices = rng.choice(
            len(source),
            size=min(samples_per_frame, len(source)),
            replace=False,
        )

        source_samples.append(source[indices])
        reference_samples.append(reference[indices])

    print(f"Collected {len(source_samples) * len(source_samples[0])} samples")
    return (
        np.concatenate(source_samples),
        np.concatenate(reference_samples),
    )


def save_samples(filename: str, *, dcp_rgb: np.ndarray, bd_rgb: np.ndarray):
    np.savez_compressed(
        filename,
        dcp=dcp_rgb,
        bd=bd_rgb,
    )


LUT_DIR = CUR_DIR / "luts"

# Variant labels as they appear in the .cube filenames.
IDENTITY_VARIANTS = ("a-flex", "b-balanced", "c-stronger")
SMOOTHNESS_VARIANTS = ("a-local", "b-balanced", "c-smooth")


def _normalize_variant(
    value: str,
    options: tuple[str, ...],
    field: str,
) -> str:
    """Resolve a full variant label, or its "a"/"b"/"c" shorthand."""
    value = value.strip().lower()

    if value in options:
        return value

    shorthand = [option for option in options if option[0] == value]
    if len(shorthand) == 1:
        return shorthand[0]

    raise ValueError(
        f"Unknown {field} variant {value!r}. "
        f"Expected one of {', '.join(options)} (or shorthand a/b/c)."
    )


def pick_lut_table(
    grid: Literal[17, 33, 65] = 65,
    identity: str = "a-flex",
    smoothness: str = "a-local",
    *,
    lut_dir: Path = LUT_DIR,
) -> Path:
    """Resolve a DCP -> Blu-ray .cube LUT from the ``luts`` directory.

    Filenames follow the pattern
    ``boane-dcp-to-bluray-g{grid}_{identity}_{smoothness}_it{iters}.cube``.

    Parameters
    ----------
    grid:
        LUT grid size, one of ``17``, ``33`` or ``65``. Larger grids are
        more accurate but cost more to evaluate.
    identity:
        How strongly the fit is pulled back toward identity, i.e. how
        flexible the correction is. One of ``"a-flex"`` (most correction),
        ``"b-balanced"`` or ``"c-stronger"`` (most conservative). The bare
        letters ``"a"``/``"b"``/``"c"`` are accepted as shorthand.
    smoothness:
        How strongly the fit is regularised across the grid. One of
        ``"a-local"`` (most detail), ``"b-balanced"`` or ``"c-smooth"``
        (smoothest). Shorthand ``"a"``/``"b"``/``"c"`` also accepted.
    lut_dir:
        Directory holding the ``.cube`` files. Defaults to ``luts`` next to
        this module.

    Returns
    -------
    The path to the matching ``.cube`` file, ready to hand to
    :func:`apply_3d_lut_and_yuv`.
    """
    if grid not in (17, 33, 65):
        raise ValueError(
            f"Unknown grid size {grid!r}. Expected one of 17, 33, 65."
        )

    identity = _normalize_variant(identity, IDENTITY_VARIANTS, "identity")
    smoothness = _normalize_variant(smoothness, SMOOTHNESS_VARIANTS, "smoothness")

    pattern = f"boane-dcp-to-bluray-g{grid}_{identity}_{smoothness}_it*.cube"
    matches = sorted(lut_dir.glob(pattern))

    if len(matches) == 1:
        return matches[0]

    available = ", ".join(sorted(p.name for p in lut_dir.glob("*.cube"))) or "none"

    if not matches:
        raise FileNotFoundError(
            f"No LUT matching {pattern!r} in {lut_dir}.\n"
            f"Available LUTs: {available}"
        )

    raise FileNotFoundError(
        f"Ambiguous LUT selection, {len(matches)} files match "
        f"{pattern!r} in {lut_dir}: "
        f"{', '.join(p.name for p in matches)}"
    )


def apply_3d_lut(
    clip: vs.VideoNode,
    lut_path: Path,
    interpolate: Literal["linear"] | Literal["tetra"] = "linear",
) -> vs.VideoNode:
    return core.timecube.Cube(
        clip,
        cube=str(lut_path),
        range=1,  # full range
        interp=1 if interpolate == "tetra" else 0,
        cpu=3,  # avx512 fast as hell
    )


def apply_3d_lut_and_yuv(
    clip: vs.VideoNode,
    lut_path: Path,
    interpolate: Literal["linear"] | Literal["tetra"] = "linear",
) -> vs.VideoNode:
    # Apply the 3D LUT first
    matched = apply_3d_lut(clip, lut_path, interpolate)

    # Convert to YUV444P10 Rec.709
    return core.resize.Bicubic(
        matched,
        format=vs.YUV444P10,
        matrix_s="709",
        transfer_s="709",
        primaries_s="709",
        range_in_s="full",
        range_s="full",
        dither_type="error_diffusion",
    )
