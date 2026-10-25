"""Safe orchestration and validation for the original 2DGS COLMAP converter."""

from __future__ import annotations

import logging
import shlex
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from research_pipeline.pipeline_utils import run_command

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
REQUIRED_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


class PreparationStatus(str, Enum):
    """Observed preparation state of one scene."""

    PREPARED = "prepared"
    NOT_PREPARED = "not_prepared"
    PARTIAL = "partial"


class PreparationAction(str, Enum):
    """Action taken (or planned) for one scene."""

    ALREADY_PREPARED = "already_prepared"
    PREPARED_NOW = "prepared_now"
    WOULD_PREPARE = "would_prepare"
    FAILED = "failed"


@dataclass(frozen=True)
class PreparationInspection:
    """Detailed state discovered from a scene directory."""

    status: PreparationStatus
    missing_artifacts: Tuple[str, ...]
    prepared_image_count: int


@dataclass(frozen=True)
class PreparationResult:
    """Outcome of preparing or validating one scene."""

    scene: str
    initial_status: PreparationStatus
    action: PreparationAction
    success: bool
    message: str
    command: Optional[Tuple[str, ...]] = None


class PreparationError(RuntimeError):
    """Raised when raw inputs cannot safely be passed to convert.py."""


def _image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_scene(source_path: Union[str, Path]) -> PreparationInspection:
    """Classify a scene as prepared, not prepared, or partial/invalid."""

    source = Path(source_path)
    images_dir = source / "images"
    sparse_dir = source / "sparse"
    sparse_model_dir = sparse_dir / "0"
    distorted_dir = source / "distorted"
    prepared_images = _image_files(images_dir)

    missing = []
    if not prepared_images:
        missing.append("images/ with at least one supported image")
    required_paths = [sparse_model_dir / name for name in REQUIRED_MODEL_FILES]
    for path in required_paths:
        if not path.is_file():
            missing.append(path.relative_to(source).as_posix())

    if not missing:
        status = PreparationStatus.PREPARED
    else:
        output_markers = [images_dir, sparse_dir, distorted_dir] + required_paths
        status = (
            PreparationStatus.PARTIAL
            if any(path.exists() for path in output_markers)
            else PreparationStatus.NOT_PREPARED
        )

    return PreparationInspection(status, tuple(missing), len(prepared_images))


def is_scene_prepared(source_path: Union[str, Path]) -> bool:
    """Return whether all required COLMAP artifacts exist for a scene."""

    return inspect_scene(source_path).status == PreparationStatus.PREPARED


def validate_raw_input(source_path: Union[str, Path]) -> Sequence[Path]:
    """Return input images or raise a clear preparation error."""

    source = Path(source_path)
    input_dir = source / "input"
    if not input_dir.is_dir():
        raise PreparationError("Raw input directory not found: {}".format(input_dir))
    images = _image_files(input_dir)
    if not images:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise PreparationError(
            "Raw input directory contains no supported images ({}): {}".format(
                extensions, input_dir
            )
        )
    return images


def build_convert_command(repo_root: Path, source_path: Path) -> Tuple[str, ...]:
    """Build the original 2DGS convert.py command for one scene."""

    return (
        sys.executable,
        str(repo_root / "convert.py"),
        "-s",
        str(source_path),
    )


def prepare_scene(
    scene_name: str,
    source_path: Union[str, Path],
    repo_root: Union[str, Path],
    dry_run: bool = False,
    force_preprocess: bool = False,
    command_runner: Callable[..., Any] = run_command,
) -> PreparationResult:
    """Validate or prepare one scene without destructive cleanup."""

    source = Path(source_path)
    root = Path(repo_root)
    inspection = inspect_scene(source)
    command = build_convert_command(root, source)

    if inspection.status == PreparationStatus.PREPARED:
        if force_preprocess:
            return PreparationResult(
                scene_name,
                inspection.status,
                PreparationAction.FAILED,
                False,
                "Forced preprocessing is unsafe because COLMAP outputs already "
                "exist. Archive or clean them manually before rerunning; no files "
                "were deleted.",
                command,
            )
        return PreparationResult(
            scene_name,
            inspection.status,
            PreparationAction.ALREADY_PREPARED,
            True,
            "COLMAP already prepared; skipping.",
            command,
        )

    if inspection.status == PreparationStatus.PARTIAL:
        return PreparationResult(
            scene_name,
            inspection.status,
            PreparationAction.FAILED,
            False,
            "Partial/invalid COLMAP outputs detected. Missing: {}. Resolve or "
            "archive existing outputs manually; no files were deleted.".format(
                ", ".join(inspection.missing_artifacts)
            ),
            command,
        )

    if dry_run:
        LOGGER.info("Would run: %s", shlex.join(command))
        return PreparationResult(
            scene_name,
            inspection.status,
            PreparationAction.WOULD_PREPARE,
            True,
            "COLMAP is not prepared; convert.py would be executed.",
            command,
        )

    try:
        validate_raw_input(source)
    except PreparationError as exc:
        return PreparationResult(
            scene_name,
            inspection.status,
            PreparationAction.FAILED,
            False,
            str(exc),
            command,
        )

    try:
        completed = command_runner(command, cwd=root)
        return_code = getattr(completed, "returncode", 0)
        if return_code != 0:
            raise PreparationError(
                "convert.py exited with code {}".format(return_code)
            )
    except Exception as exc:
        return PreparationResult(
            scene_name,
            inspection.status,
            PreparationAction.FAILED,
            False,
            "convert.py failed: {}".format(exc),
            command,
        )

    final_inspection = inspect_scene(source)
    if final_inspection.status != PreparationStatus.PREPARED:
        return PreparationResult(
            scene_name,
            inspection.status,
            PreparationAction.FAILED,
            False,
            "convert.py completed but required artifacts are incomplete. "
            "Missing: {}".format(", ".join(final_inspection.missing_artifacts)),
            command,
        )

    return PreparationResult(
        scene_name,
        inspection.status,
        PreparationAction.PREPARED_NOW,
        True,
        "COLMAP preparation completed and artifacts were validated.",
        command,
    )
