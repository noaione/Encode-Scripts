
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


def make_grid(size: int) -> np.ndarray:
    """Identity LUT, flattened with B as the fastest axis."""
    axis = np.linspace(0.0, 1.0, size)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)


def interpolation_matrix(rgb: np.ndarray, size: int):
    """Trilinear interpolation weights for RGB samples."""
    rgb = np.clip(rgb, 0.0, 1.0)
    coords = rgb * (size - 1)

    lo = np.minimum(
        np.floor(coords).astype(np.int32),
        size - 2,
    )
    t = coords - lo

    n = len(rgb)
    rows = []
    cols = []
    weights = []

    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                offset = np.array([dr, dg, db])

                p = lo + offset

                idx = (
                    (p[:, 0] * size + p[:, 1]) * size
                    + p[:, 2]
                )

                w = np.prod(
                    np.where(offset == 1, t, 1.0 - t),
                    axis=1,
                )

                rows.append(np.arange(n))
                cols.append(idx)
                weights.append(w)

    return sparse.coo_matrix(
        (
            np.concatenate(weights),
            (
                np.concatenate(rows),
                np.concatenate(cols),
            ),
        ),
        shape=(n, size**3),
    ).tocsr()


def smoothness_matrix(size: int):
    """First differences along R, G, B grid axes."""
    ids = np.arange(size**3).reshape(
        size, size, size
    )

    pairs = []

    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3

        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)

        a = ids[tuple(left)].ravel()
        b = ids[tuple(right)].ravel()

        pairs.append((a, b))

    a = np.concatenate([p[0] for p in pairs])
    b = np.concatenate([p[1] for p in pairs])

    rows = np.repeat(np.arange(len(a)), 2)
    cols = np.column_stack([a, b]).ravel()
    values = np.tile([1.0, -1.0], len(a))

    return sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(len(a), size**3),
    ).tocsr()


def fit_lut(
    source: np.ndarray,
    reference: np.ndarray,
    size: int = 17,
    identity_strength: float = 0.001,
    smoothness: float = 0.1,
    iterations: int = 500,
    *,
    stop_tolerance: float = 1e-7,
) -> np.ndarray:
    """
    Fit a regularized 3D LUT using sparse least squares.

    Source and reference:
        (N, 3) float arrays, RGB in [0, 1].

    Returns:
        (size, size, size, 3) LUT.
    """
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if source.shape != reference.shape:
        raise ValueError("Sample shapes do not match")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Expected RGB arrays of shape (N, 3)")
    if len(source) == 0:
        raise ValueError("No input samples")
    if identity_strength < 0 or smoothness < 0:
        raise ValueError("Regularization must be nonnegative")

    valid = (
        np.isfinite(source).all(axis=1)
        & np.isfinite(reference).all(axis=1)
        & (source >= 0).all(axis=1)
        & (source <= 1).all(axis=1)
        & (reference >= 0).all(axis=1)
        & (reference <= 1).all(axis=1)
    )

    source = source[valid]
    reference = reference[valid]

    if len(source) == 0:
        raise ValueError("No valid samples")

    grid = make_grid(size)

    # Fit corrections to the identity LUT.
    # The source is exactly reproduced by trilinear
    # interpolation of the identity grid.
    delta = reference - source

    W = interpolation_matrix(source, size)
    D = smoothness_matrix(size)

    # Normalize each objective by its row count.
    data_scale = 1.0 / np.sqrt(len(source))
    identity_scale = np.sqrt(
        identity_strength / len(grid)
    )
    smooth_scale = np.sqrt(
        smoothness / D.shape[0]
    )

    A = sparse.vstack(
        [
            W * data_scale,
            sparse.eye(
                size**3, format="csr"
            ) * identity_scale,
            D * smooth_scale,
        ],
        format="csr",
    )

    zeros = np.zeros(size**3 + D.shape[0])

    corrections = np.empty_like(grid)

    for channel in range(3):
        target = np.concatenate([
            delta[:, channel] * data_scale,
            zeros,
        ])

        result = lsqr(
            A,
            target,
            atol=stop_tolerance,
            btol=stop_tolerance,
            iter_lim=iterations,
        )

        corrections[:, channel] = result[0]

        istop = result[1]
        residual = result[3]
        normal_residual = result[4]
        condition = result[6]

        print(
            f"Channel {channel}: "
            f"stop={istop}, "
            f"iterations={iterations}, "
            f"residual={residual:.6f}, "
            f"normal_residual={normal_residual:.6f}, "
            f"condition={condition:.2e}"
        )

    lut = np.clip(grid + corrections, 0.0, 1.0)

    return lut.reshape(size, size, size, 3)


def write_cube(
    lut: np.ndarray,
    path: Path,
) -> None:
    size = lut.shape[0]

    if lut.shape != (size, size, size, 3):
        raise ValueError("Invalid LUT shape")

    with path.open("w", encoding="utf-8") as f:
        f.write('TITLE "BOANE DCP to Blu-ray"\n')
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0 0 0\n")
        f.write("DOMAIN_MAX 1 1 1\n")

        # .cube ordering: R changes fastest.
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    rgb = lut[r, g, b]
                    f.write(
                        f"{rgb[0]:.9f} "
                        f"{rgb[1]:.9f} "
                        f"{rgb[2]:.9f}\n"
                    )


def make_filename(grid: int, ident: float, smoothness: float, iterate: int) -> str:
    match_ident = {
        0.0001: "a-flex",
        0.001: "b-balanced",
        0.01: "c-stronger",
    }
    sm_ident = {
        0.01: "a-local",
        0.1: "b-balanced",
        1.0: "c-smooth"
    }
    return f"boane-dcp-to-bluray-g{grid}_{match_ident[ident]}_{sm_ident[smoothness]}_it{iterate}.cube"


def merge_samples(directory: str, max_per_scene: int = 80_000, seed: int = 8654491321826895238):
    rng = np.random.default_rng(seed)
    sources = []
    references = []

    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path) as data:
            source = data["dcp"]
            reference = data["bd"]

        if source.shape != reference.shape:
            raise ValueError(f"Shape mismatch in {path}")

        count = min(len(source), max_per_scene)
        indices = rng.choice(len(source), size=count, replace=False)

        sources.append(source[indices])
        references.append(reference[indices])

        print(f"{path.name}: selected {count:,} samples")
    if not sources:
        raise ValueError("No samples found")
    return np.concatenate(sources), np.concatenate(references)


if __name__ == "__main__":
    source, reference = merge_samples("samples")
    target_dir = Path("luts")
    target_dir.mkdir(exist_ok=True)

    # for grid, iterate in [(17, 500), (33, 1000), (65, 2000)]:
    #     for ident in [0.0001, 0.001, 0.01]:
    #         for smoothness in [0.01, 0.1, 1.0]:
    #             lut = fit_lut(
    #                 source,
    #                 reference,
    #                 size=grid,
    #                 identity_strength=ident,
    #                 smoothness=smoothness,
    #                 iterations=iterate,
    #             )

    #             filename = make_filename(grid, ident, smoothness, iterate)
    #             write_cube(lut, target_dir / filename)
    #             print(f"Saved {filename}")

    # Saved boane-dcp-to-bluray-g17it500_a-flex_a-local.cube
    SIZE = 65
    IDENT = 0.001
    SMOOTHNESS = 0.1
    ITERATE = 2000
    STOP_TOLERANCE = 1e-5
    lut = fit_lut(
        source,
        reference,
        size=SIZE,
        identity_strength=IDENT,
        smoothness=SMOOTHNESS,
        iterations=ITERATE,
        stop_tolerance=STOP_TOLERANCE,
    )

    filename = make_filename(SIZE, IDENT, SMOOTHNESS, ITERATE)
    write_cube(lut, target_dir / filename)
    print(f"Saved {filename}")
