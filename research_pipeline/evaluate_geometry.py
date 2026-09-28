"""Point-set geometry metrics for aligned reconstruction and ground truth.

Let ``R`` be reconstruction samples and ``G`` ground-truth samples. The nearest
neighbor distances are ``d(R, G)`` and ``d(G, R)``. This evaluator reports the
sum of both directional mean Chamfer terms using unsquared Euclidean distances:

    CD = mean(d(R, G)) + mean(d(G, R))

For a required distance threshold ``t``:

    precision = mean(d(R, G) < t)
    recall    = mean(d(G, R) < t)
    F1        = 2 * precision * recall / (precision + recall)

F1 is zero when precision + recall is zero. No ICP, scale normalization,
orientation correction, or automatic alignment is performed. Both geometries
must already share coordinate system, scale, orientation, and alignment.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

LOGGER = logging.getLogger(__name__)


@dataclass
class GeometryMetrics:
    """Geometry metric values for a pair of aligned point sets."""

    chamfer_distance: float
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    distance_threshold: Optional[float]


def _load_point_set(
    geometry_path: Union[str, Path], sample_points: int, seed: int
) -> Any:
    """Load a point cloud or sample a mesh surface using trimesh."""

    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Geometry evaluation requires numpy and trimesh. Install "
            "requirements-research.txt on the PC-server."
        ) from exc

    path = Path(geometry_path)
    if not path.is_file():
        raise FileNotFoundError("Geometry file not found: {}".format(path))

    geometry = trimesh.load(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        meshes = list(geometry.geometry.values())
        if not meshes:
            raise ValueError("Geometry scene is empty: {}".format(path))
        geometry = trimesh.util.concatenate(meshes)

    vertices = np.asarray(getattr(geometry, "vertices", []), dtype=np.float64)
    faces = np.asarray(getattr(geometry, "faces", []))
    if faces.size > 0:
        if sample_points <= 0:
            raise ValueError("sample_points must be greater than zero")
        points, _ = trimesh.sample.sample_surface(
            geometry, sample_points, seed=seed
        )
        points = np.asarray(points, dtype=np.float64)
    else:
        points = vertices
        if sample_points > 0 and len(points) > sample_points:
            indices = np.random.default_rng(seed).choice(
                len(points), size=sample_points, replace=False
            )
            points = points[indices]

    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Expected a non-empty Nx3 point set: {}".format(path))
    if not np.isfinite(points).all():
        raise ValueError("Geometry contains non-finite coordinates: {}".format(path))
    return points


def evaluate_point_sets(
    reconstruction_points: Any,
    ground_truth_points: Any,
    distance_threshold: Optional[float],
) -> GeometryMetrics:
    """Compute bidirectional-sum Chamfer distance and strict-threshold P/R/F1."""

    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "Geometry evaluation requires numpy and scipy. Install "
            "requirements-research.txt on the PC-server."
        ) from exc

    reconstruction = np.asarray(reconstruction_points, dtype=np.float64)
    ground_truth = np.asarray(ground_truth_points, dtype=np.float64)
    for name, points in (
        ("reconstruction", reconstruction),
        ("ground truth", ground_truth),
    ):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("{} must be a non-empty Nx3 array".format(name))

    reconstruction_to_gt = cKDTree(ground_truth).query(
        reconstruction, k=1, workers=-1
    )[0]
    gt_to_reconstruction = cKDTree(reconstruction).query(
        ground_truth, k=1, workers=-1
    )[0]
    chamfer = float(np.mean(reconstruction_to_gt)) + float(
        np.mean(gt_to_reconstruction)
    )

    if distance_threshold is None:
        LOGGER.warning(
            "distance_threshold is null; precision, recall, and F1-score "
            "will be left empty. Chamfer distance is still computed."
        )
        return GeometryMetrics(chamfer, None, None, None, None)
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative")

    precision = float(np.mean(reconstruction_to_gt < distance_threshold))
    recall = float(np.mean(gt_to_reconstruction < distance_threshold))
    denominator = precision + recall
    f1_score = 0.0 if denominator == 0 else 2.0 * precision * recall / denominator
    return GeometryMetrics(
        chamfer, precision, recall, f1_score, float(distance_threshold)
    )


def evaluate_geometry(
    reconstruction_path: Union[str, Path],
    ground_truth_path: Union[str, Path],
    distance_threshold: Optional[float],
    sample_points: int = 100_000,
    seed: int = 0,
) -> GeometryMetrics:
    """Load two geometries and evaluate them as aligned point sets."""

    reconstruction = _load_point_set(reconstruction_path, sample_points, seed)
    ground_truth = _load_point_set(ground_truth_path, sample_points, seed + 1)
    return evaluate_point_sets(reconstruction, ground_truth, distance_threshold)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument("--sample-points", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = evaluate_geometry(
        args.reconstruction,
        args.ground_truth,
        args.distance_threshold,
        args.sample_points,
        args.seed,
    )
    print(json.dumps(asdict(metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
