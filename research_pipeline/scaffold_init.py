"""Convert official Scaffold-GS anchors into a 2DGS initialization PLY."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)
INITIALIZATION_METHOD = "scaffold_anchor"
INITIALIZATION_MANIFEST = "initialization.json"
INITIALIZATION_PLY = "initial_points.ply"
INITIAL_VERTEX_DTYPE = np.dtype(
    [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


class ScaffoldInitializationError(RuntimeError):
    """Raised when anchors cannot be converted into valid 2DGS input."""


@dataclass(frozen=True)
class ScaffoldInitializationResult:
    """Metadata for a generated or reused initialization point cloud."""

    output_ply: Path
    manifest_path: Path
    anchor_count: Optional[int]
    colmap_point_count: Optional[int]
    mean_color_nn_distance: Optional[float]
    max_color_nn_distance: Optional[float]
    reused: bool


def _ply_data_type() -> Any:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise ScaffoldInitializationError(
            "plyfile is required to read Scaffold anchors and write initial_points.ply."
        ) from exc
    return PlyData, PlyElement


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scaffold_anchor_xyz(point_cloud_path: Path) -> np.ndarray:
    """Read Scaffold ``x/y/z`` fields, which official ``save_ply`` writes from ``_anchor``."""

    PlyData, _ = _ply_data_type()
    try:
        vertex = PlyData.read(str(point_cloud_path))["vertex"]
        names = set(vertex.data.dtype.names or ())
        missing = {"x", "y", "z"} - names
        if missing:
            raise ScaffoldInitializationError(
                "Scaffold point cloud lacks anchor XYZ fields: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        anchors = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
            np.float64, copy=False
        )
    except ScaffoldInitializationError:
        raise
    except Exception as exc:
        raise ScaffoldInitializationError(
            "Unable to read Scaffold anchor PLY '{}': {}".format(point_cloud_path, exc)
        ) from exc
    if anchors.ndim != 2 or anchors.shape[1] != 3 or not len(anchors):
        raise ScaffoldInitializationError("Scaffold anchor PLY contains no XYZ points")
    if not np.isfinite(anchors).all():
        raise ScaffoldInitializationError("Scaffold anchor XYZ contains non-finite values")
    return anchors


def _load_colmap_reader(repo_root: Path) -> Any:
    module_path = repo_root / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("research_colmap_loader", module_path)
    if spec is None or spec.loader is None:
        raise ScaffoldInitializationError(
            "Unable to load 2DGS COLMAP reader: {}".format(module_path)
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_colmap_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    PlyData, _ = _ply_data_type()
    vertex = PlyData.read(str(path))["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"}
    if not required.issubset(names):
        raise ScaffoldInitializationError(
            "COLMAP PLY must contain XYZ and RGB fields: {}".format(path)
        )
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")])
    rgb = np.column_stack([vertex[name] for name in ("red", "green", "blue")])
    return xyz, rgb


def _find_colmap_point_source(source_path: Path) -> Path:
    sparse_dir = source_path / "sparse" / "0"
    for name in ("points3D.bin", "points3D.txt", "points3D.ply"):
        candidate = sparse_dir / name
        if candidate.is_file():
            return candidate
    raise ScaffoldInitializationError(
        "No sparse COLMAP points found. Expected points3D.bin, points3D.txt, "
        "or points3D.ply in {}".format(sparse_dir)
    )


def read_colmap_sparse_points(
    repo_root: Path, source_path: Path
) -> Tuple[np.ndarray, np.ndarray, Path]:
    """Read sparse COLMAP XYZ/RGB from binary, text, or PLY output."""

    selected = _find_colmap_point_source(source_path)
    if selected.suffix.lower() in (".bin", ".txt"):
        loader = _load_colmap_reader(repo_root)
        if selected.suffix.lower() == ".bin":
            xyz, rgb, _ = loader.read_points3D_binary(str(selected))
        else:
            xyz, rgb, _ = loader.read_points3D_text(str(selected))
    else:
        xyz, rgb = _read_colmap_ply(selected)

    xyz_array = np.asarray(xyz, dtype=np.float64)
    rgb_array = np.asarray(rgb)
    if xyz_array.shape != rgb_array.shape or xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
        raise ScaffoldInitializationError("Sparse COLMAP XYZ/RGB arrays are invalid")
    if not len(xyz_array) or not np.isfinite(xyz_array).all():
        raise ScaffoldInitializationError("Sparse COLMAP point cloud is empty or non-finite")
    rgb_array = np.clip(np.rint(rgb_array), 0, 255).astype(np.uint8)
    return xyz_array, rgb_array, selected


def assign_nearest_colmap_colors(
    anchor_xyz: np.ndarray,
    colmap_xyz: np.ndarray,
    colmap_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each anchor RGB from its nearest sparse COLMAP point in Euclidean XYZ."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ScaffoldInitializationError(
            "SciPy is required for nearest-neighbor anchor color assignment."
        ) from exc
    tree = cKDTree(np.asarray(colmap_xyz, dtype=np.float64))
    distances, indices = tree.query(np.asarray(anchor_xyz, dtype=np.float64), k=1)
    return np.asarray(colmap_rgb, dtype=np.uint8)[indices], np.asarray(distances)


def build_initial_vertex_array(anchor_xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Build the exact vertex schema expected by 2DGS ``fetchPly``."""

    anchors = np.asarray(anchor_xyz, dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.uint8)
    if anchors.shape != colors.shape or anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ScaffoldInitializationError("Anchor XYZ and RGB must both have shape (N, 3)")
    vertices = np.empty(len(anchors), dtype=INITIAL_VERTEX_DTYPE)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = anchors[:, index]
    for name in ("nx", "ny", "nz"):
        vertices[name] = 0.0
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, index]
    return vertices


def _write_initial_ply(path: Path, vertices: np.ndarray) -> None:
    PlyData, PlyElement = _ply_data_type()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(temporary))
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, Mapping) else None
    except (OSError, ValueError):
        return None


def _reusable_initialization(
    manifest_path: Path,
    output_ply: Path,
    source_path: Path,
    scaffold_point_cloud: Path,
    scaffold_iterations: int,
    scaffold_commit: str,
) -> Optional[Mapping[str, Any]]:
    manifest = _read_json(manifest_path)
    if manifest is None or not output_ply.is_file() or output_ply.stat().st_size == 0:
        return None
    expected = {
        "status": "success",
        "method": INITIALIZATION_METHOD,
        "source_scene": str(source_path.resolve()),
        "scaffold_point_cloud": str(scaffold_point_cloud.resolve()),
        "scaffold_iterations": scaffold_iterations,
        "scaffold_commit": scaffold_commit,
        "scaffold_point_cloud_sha256": _sha256(scaffold_point_cloud),
        "colmap_point_source": str(_find_colmap_point_source(source_path).resolve()),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if manifest.get("colmap_point_source_sha256") != _sha256(
        _find_colmap_point_source(source_path)
    ):
        return None
    if manifest.get("output_ply_sha256") != _sha256(output_ply):
        return None
    return manifest


def create_scaffold_initialization(
    repo_root: Path,
    source_path: Path,
    scaffold_point_cloud: Path,
    output_directory: Path,
    scaffold_iterations: int,
    scaffold_commit: str,
    reuse_successful: bool = True,
    dry_run: bool = False,
) -> ScaffoldInitializationResult:
    """Create or safely reuse a Scaffold-anchor initialization PLY."""

    output_ply = output_directory / INITIALIZATION_PLY
    manifest_path = output_directory / INITIALIZATION_MANIFEST
    if dry_run:
        return ScaffoldInitializationResult(
            output_ply, manifest_path, None, None, None, None, False
        )
    if not scaffold_point_cloud.is_file():
        raise ScaffoldInitializationError(
            "Scaffold anchor point cloud not found: {}".format(scaffold_point_cloud)
        )

    if reuse_successful:
        manifest = _reusable_initialization(
            manifest_path,
            output_ply,
            source_path,
            scaffold_point_cloud,
            scaffold_iterations,
            scaffold_commit,
        )
        if manifest is not None:
            LOGGER.info("Reusing validated Scaffold initialization: %s", output_ply)
            return ScaffoldInitializationResult(
                output_ply,
                manifest_path,
                int(manifest["anchor_count"]),
                int(manifest["colmap_point_count"]),
                float(manifest["mean_color_nn_distance"]),
                float(manifest["max_color_nn_distance"]),
                True,
            )

    anchors = read_scaffold_anchor_xyz(scaffold_point_cloud)
    colmap_xyz, colmap_rgb, colmap_source = read_colmap_sparse_points(repo_root, source_path)
    colors, distances = assign_nearest_colmap_colors(anchors, colmap_xyz, colmap_rgb)
    vertices = build_initial_vertex_array(anchors, colors)
    _write_initial_ply(output_ply, vertices)
    payload = {
        "status": "success",
        "method": INITIALIZATION_METHOD,
        "source_scene": str(source_path.resolve()),
        "scaffold_point_cloud": str(scaffold_point_cloud.resolve()),
        "scaffold_point_cloud_sha256": _sha256(scaffold_point_cloud),
        "scaffold_iterations": scaffold_iterations,
        "scaffold_commit": scaffold_commit,
        "anchor_count": len(anchors),
        "colmap_point_count": len(colmap_xyz),
        "colmap_point_source": str(colmap_source.resolve()),
        "colmap_point_source_sha256": _sha256(colmap_source),
        "mean_color_nn_distance": float(np.mean(distances)),
        "max_color_nn_distance": float(np.max(distances)),
        "output_ply": str(output_ply.resolve()),
        "output_ply_sha256": _sha256(output_ply),
    }
    _write_json(manifest_path, payload)
    return ScaffoldInitializationResult(
        output_ply,
        manifest_path,
        len(anchors),
        len(colmap_xyz),
        payload["mean_color_nn_distance"],
        payload["max_color_nn_distance"],
        False,
    )
