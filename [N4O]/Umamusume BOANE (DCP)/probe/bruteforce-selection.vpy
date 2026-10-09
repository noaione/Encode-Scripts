"""Rank every candidate DCP -> Blu-ray LUT against the Blu-ray reference.

The fit yields 27 LUTs (3 grid sizes x 3 identity strengths x 3 smoothness
settings). This script scores each one by comparing the LUT-corrected DCP
against the Blu-ray reference on a set of representative frames per scene.

Two full-reference metrics are computed per frame on the GPU (VSHIP):

* ``SSIMULACRA2`` - colour-aware, MOS-correlated similarity in ``(-inf, 100]``.
  Higher is better; ~90+ is visually lossless, 80+ is high quality.
* ``Butteraugli`` - perceptual distance, lower is better; around ``1.0`` is the
  imperceptibility threshold. Butteraugli is very sensitive in the
  near-lossless regime and rarely reports a good image as bad.

LUTs are ranked by the *sum of their ranks* on the two metrics, so a LUT has
to score well on both to reach the top. The full table is printed and saved
to ``exports/lut-ranking.json``.

Run::

    python boane-vs-export.vpy
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Literal, cast

import vapoursynth as vs
from vstools import core

from boane_helpers import (
    IDENTITY_VARIANTS,
    SMOOTHNESS_VARIANTS,
    apply_3d_lut,
    pick_lut_table,
    read_sample,
)

OUT_DIR = Path(__file__).resolve().parent / "exports"

# Every combination of these axes is scored.
GRIDS: tuple[Literal[17, 33, 65], ...] = (17, 33, 65)

# Concurrent frames piped through the GPU by VSHIP. Lower it if VRAM is tight.
NUM_STREAM = 3

# Frame cache, in MB. Keeps the decoded/aligned BD + DCP frames alive while all
# 27 LUTs are scored, so the shared decode is not repeated for every LUT.
CACHE_MB = 4096

SSIM = "ssimulacra2"
BUTTER = "butteraugli"
BUTTER_INF = "butteraugli_infnorm"

# Per-scene frame selections, copied from the original ``boane-vs-*.vpy`` files.
# ``bd_trim`` reproduces the extra ``bd_clip[24:]`` cut the logotype and
# tachyon-zone scripts applied on top of ``read_sample``'s own shift.
SCENES = {
    "darkscene": {
        "shift": 100,
        "bd_trim": 0,
        "frames": [7, 204, 236, 334, 847, 1211, 1301],
    },
    "logotype": {
        "shift": 0,
        "bd_trim": 24,
        "frames": [54, 247, 305, 362, 403],
    },
    "t0020": {
        "shift": 62,
        "bd_trim": 0,
        "frames": [850, 294, 660, 95, 266, 410, 362, 1130, 1323],
    },
    "tachyon-zone": {
        "shift": 85,
        "bd_trim": 24,
        "frames": [7, 383, 480, 496, 671, 714, 797, 868, 879, 927, 1075, 1203],
    },
}


def all_luts() -> list[tuple[Literal[17, 33, 65], str, str]]:
    """Every (grid, identity, smoothness) combination, in a stable order."""
    return [
        (grid, identity, smoothness)
        for grid in GRIDS
        for identity in IDENTITY_VARIANTS
        for smoothness in SMOOTHNESS_VARIANTS
    ]


def lut_label(grid: int, identity: str, smoothness: str) -> str:
    return f"g{grid}/{identity}/{smoothness}"


def _prop_float(frame: vs.VideoFrame, key: str) -> float:
    """Read a float frame property (VSHIP stores metric scores as props)."""
    return float(cast(float, frame.props[key]))


def build_scene(kind: str, spec: dict) -> dict[str, dict[str, list[float]]]:
    """Score every candidate LUT on one scene.

    Returns ``{lut_label: {"ssimulacra2": [...], "butteraugli": [...],
    "butteraugli_infnorm": [...]}}`` with one value per selected frame.
    """
    bd_clip, dcp_clip = read_sample(kind, spec["shift"])
    if spec["bd_trim"]:
        bd_clip = bd_clip[spec["bd_trim"] :]

    # Both metrics need matching lengths; the two sources differ by a few
    # frames. Cut only the tail, so frame indices stay valid.
    length = min(len(bd_clip), len(dcp_clip))
    bd_clip, dcp_clip = bd_clip[:length], dcp_clip[:length]

    frames = spec["frames"]
    out_of_range = [n for n in frames if n >= length]
    if out_of_range:
        raise ValueError(
            f"{kind}: frame(s) {out_of_range} outside the aligned range "
            f"0..{length - 1}"
        )

    luts = all_luts()
    scores: dict[str, dict[str, list[float]]] = {}

    for index, (grid, identity, smoothness) in enumerate(luts, start=1):
        label = lut_label(grid, identity, smoothness)
        print(f"    [{index:2d}/{len(luts)}] {label}", flush=True)

        lut_clip = apply_3d_lut(
            dcp_clip, pick_lut_table(grid, identity, smoothness)
        )
        ssim_clip = core.vship.SSIMULACRA2(
            bd_clip, lut_clip, numStream=NUM_STREAM
        )
        butter_clip = core.vship.BUTTERAUGLI(
            bd_clip, lut_clip, numStream=NUM_STREAM
        )

        ssim_values: list[float] = []
        butter_values: list[float] = []
        butter_inf_values: list[float] = []

        for n in frames:
            ssim_values.append(
                _prop_float(ssim_clip.get_frame(n), "_SSIMULACRA2")
            )
            butter_frame = butter_clip.get_frame(n)
            butter_values.append(
                _prop_float(butter_frame, "_BUTTERAUGLI_3Norm")
            )
            butter_inf_values.append(
                _prop_float(butter_frame, "_BUTTERAUGLI_INFNorm")
            )

        scores[label] = {
            SSIM: ssim_values,
            BUTTER: butter_values,
            BUTTER_INF: butter_inf_values,
        }

        # Drop the GPU filters before building the next LUT's filters.
        del lut_clip, ssim_clip, butter_clip

    return scores


def summarise(
    per_scene: dict[str, dict[str, dict[str, list[float]]]],
) -> tuple[list[str], dict[str, dict]]:
    """Macro-average per LUT (scenes weighted equally) and rank."""
    summary: dict[str, dict] = {}

    for grid, identity, smoothness in all_luts():
        label = lut_label(grid, identity, smoothness)

        ssim_scene = {
            kind: statistics.fmean(per_scene[kind][label][SSIM])
            for kind in per_scene
        }
        butter_scene = {
            kind: statistics.fmean(per_scene[kind][label][BUTTER])
            for kind in per_scene
        }

        summary[label] = {
            "ssim_macro": statistics.fmean(ssim_scene.values()),
            "butter_macro": statistics.fmean(butter_scene.values()),
            "ssim_min": min(
                v for kind in per_scene for v in per_scene[kind][label][SSIM]
            ),
            "butter_inf_max": max(
                v
                for kind in per_scene
                for v in per_scene[kind][label][BUTTER_INF]
            ),
            "ssim_per_scene": ssim_scene,
            "butter_per_scene": butter_scene,
        }

    by_ssim = sorted(
        summary, key=lambda label: summary[label]["ssim_macro"], reverse=True
    )
    by_butter = sorted(
        summary, key=lambda label: summary[label]["butter_macro"]
    )

    for position, label in enumerate(by_ssim, start=1):
        summary[label]["rank_ssim"] = position
    for position, label in enumerate(by_butter, start=1):
        summary[label]["rank_butter"] = position
    for label in summary:
        summary[label]["combined"] = (
            summary[label]["rank_ssim"] + summary[label]["rank_butter"]
        )

    ranked = sorted(
        summary,
        key=lambda label: (
            summary[label]["combined"],
            -summary[label]["ssim_macro"],
        ),
    )
    return ranked, summary


def _print_ranking(ranked: list[str], summary: dict[str, dict]) -> None:
    header = (
        f"{'#':>2}  {'LUT':24}  {'SSIM2':>6}  {'Btr-3':>6}  "
        f"{'rk S/B':>7}  {'SSIM min':>8}  {'Btr-inf':>7}"
    )
    print(header)
    print("-" * len(header))

    for position, label in enumerate(ranked, start=1):
        row = summary[label]
        print(
            f"{position:>2}  {label:24}  {row['ssim_macro']:6.2f}  "
            f"{row['butter_macro']:6.3f}  "
            f"{row['rank_ssim']:>3}/{row['rank_butter']:<3}  "
            f"{row['ssim_min']:8.2f}  {row['butter_inf_max']:7.3f}"
        )


def main() -> None:
    try:
        core.max_cache_size = CACHE_MB
    except Exception:
        pass

    per_scene: dict[str, dict[str, dict[str, list[float]]]] = {}
    for kind, spec in SCENES.items():
        print(
            f"Scoring {kind}: {len(spec['frames'])} frames x "
            f"{len(all_luts())} LUTs",
            flush=True,
        )
        per_scene[kind] = build_scene(kind, spec)

    ranked, summary = summarise(per_scene)

    print()
    print(
        f"=== LUT ranking ({len(ranked)} LUTs, by SSIM2 rank + "
        f"Butteraugli rank) ==="
    )
    _print_ranking(ranked, summary)

    print()
    print("=== Top 5 ===")
    for position, label in enumerate(ranked[:5], start=1):
        row = summary[label]
        per_scene_text = "  ".join(
            f"{kind}={score:.2f}"
            for kind, score in row["ssim_per_scene"].items()
        )
        print(
            f"{position}. {label:24}  SSIM2={row['ssim_macro']:.2f}  "
            f"Butteraugli={row['butter_macro']:.3f}"
        )
        print(f"     per-scene SSIM2: {per_scene_text}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "lut-ranking.json"
    out.write_text(
        json.dumps({label: summary[label] for label in ranked}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
