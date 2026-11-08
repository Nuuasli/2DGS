"""Sequentially orchestrate 2DGS research experiments per scene."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_pipeline.evaluate_geometry import evaluate_geometry
from research_pipeline.gpu_monitor import GPUVRAMMonitor, VRAMStats
from research_pipeline.pipeline_utils import (
    format_duration,
    get_git_commit,
    load_yaml,
    read_2dgs_metrics,
    run_command,
)
from research_pipeline.preprocess import (
    PreparationAction,
    PreparationResult,
    PreparationStatus,
    prepare_scene,
)
from research_pipeline.result_logger import (
    SCAFFOLD_CSV_FIELDS,
    ResultLogger,
    read_successful_scenes,
)
from research_pipeline.scaffold_init import (
    INITIALIZATION_METHOD,
    create_scaffold_initialization,
)
from research_pipeline.scaffold_runner import run_scaffold_stage

LOGGER = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(repo_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _path_for_record(repo_root: Path, path: Path) -> str:
    """Prefer a repository-relative POSIX path for portable CSV records."""

    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a mapping".format(label))
    return value


def _resolve_stage_enablement(
    experiment_config: Mapping[str, Any],
    geometry_config: Mapping[str, Any],
) -> Tuple[bool, bool]:
    """Return geometry/mesh flags and reject an impossible configuration."""

    geometry_enabled = bool(geometry_config.get("enabled", True))
    mesh_value = experiment_config.get("mesh_extraction")
    if mesh_value is None:
        # Legacy configs predate this section and historically generated meshes.
        mesh_enabled = True
    else:
        mesh_config = _require_mapping(mesh_value, "mesh_extraction")
        mesh_enabled = bool(mesh_config.get("enabled", True))

    if geometry_enabled and not mesh_enabled:
        raise ValueError(
            "geometry_evaluation.enabled=true requires "
            "mesh_extraction.enabled=true because geometry evaluation needs "
            "a reconstructed mesh"
        )
    return geometry_enabled, mesh_enabled


def _initialization_method(experiment_config: Mapping[str, Any]) -> str:
    """Return the configured initialization without changing legacy configs."""

    value = experiment_config.get("initialization")
    if value is None:
        return "colmap"
    config = _require_mapping(value, "initialization")
    method = str(config.get("method", "colmap")).strip().lower()
    if method not in ("colmap", INITIALIZATION_METHOD):
        raise ValueError("Unsupported initialization.method: {}".format(method))
    return method


def _dataset_namespace(source_path: Path) -> str:
    """Derive a generic dataset directory from a scene source path."""

    namespace = source_path.parent.name.strip()
    if not namespace:
        raise ValueError(
            "Unable to derive dataset output namespace from {}".format(source_path)
        )
    return namespace


def _scene_output_path(
    repo_root: Path,
    experiment: Mapping[str, Any],
    scene: Mapping[str, Any],
    source_path: Path,
    initialization_method: str,
) -> Path:
    """Keep legacy scene output paths; isolate non-baseline ablation output."""

    if initialization_method != INITIALIZATION_METHOD:
        return _resolve_path(repo_root, scene["output"])
    output_root = _resolve_path(repo_root, experiment["output_root"])
    return output_root / _dataset_namespace(source_path) / str(scene["name"])


def _build_commands(
    repo_root: Path,
    source_path: Path,
    output_path: Path,
    iterations: int,
    init_ply: Optional[Path] = None,
    mesh_extraction_enabled: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    python = sys.executable
    train = [
        python,
        str(repo_root / "train.py"),
        "--source_path",
        str(source_path),
        "--model_path",
        str(output_path),
        "--iterations",
        str(iterations),
        "--eval",
    ]
    if init_ply is not None:
        train.extend(["--init_ply", str(init_ply)])
    render = [
        python,
        str(repo_root / "render.py"),
        "--source_path",
        str(source_path),
        "--model_path",
        str(output_path),
        "--iteration",
        str(iterations),
        "--skip_train",
    ]
    if not mesh_extraction_enabled:
        render.append("--skip_mesh")
    metrics = [
        python,
        str(repo_root / "metrics.py"),
        "--model_paths",
        str(output_path),
    ]
    return train, render, metrics


def _expected_mesh_path(output_path: Path, iterations: int) -> Path:
    return output_path / "train" / "ours_{}".format(iterations) / "fuse_post.ply"


def _find_reconstructed_mesh(output_path: Path, iterations: int) -> Path:
    iteration_dir = output_path / "train" / "ours_{}".format(iterations)
    preferred = [
        iteration_dir / "fuse_post.ply",
        iteration_dir / "fuse_unbounded_post.ply",
        iteration_dir / "fuse.ply",
        iteration_dir / "fuse_unbounded.ply",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No reconstructed mesh found for configured iteration {} in {}. "
        "Expected one of: {}".format(
            iterations,
            iteration_dir,
            ", ".join(candidate.name for candidate in preferred),
        )
    )


def _new_result_row(
    dataset: str,
    scene: str,
    output_path: str,
    iterations: int,
    git_commit: str,
    threshold: Optional[float],
    geometry_sample_points: int,
    initialization_method: str = "colmap",
) -> Dict[str, Any]:
    row = {
        "run_id": str(uuid.uuid4()),
        "dataset": dataset,
        "scene": scene,
        "status": "failed",
        "chamfer_distance": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "distance_threshold": threshold,
        "geometry_sample_points": geometry_sample_points,
        "ssim": None,
        "psnr": None,
        "lpips": None,
        "training_seconds": None,
        "training_time": None,
        "peak_vram_mb": None,
        "average_vram_mb": None,
        "iterations": iterations,
        "git_commit": git_commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_path": output_path,
        "error": None,
    }
    if initialization_method == INITIALIZATION_METHOD:
        row.update(
            {
                "initialization_method": INITIALIZATION_METHOD,
                "scaffold_iterations": None,
                "scaffold_training_seconds": None,
                "scaffold_peak_vram_mb": None,
                "scaffold_average_vram_mb": None,
                "scaffold_anchor_count": None,
                "scaffold_commit": None,
                "scaffold_reused": None,
                "init_ply": None,
                "total_ablation_seconds": None,
            }
        )
    return row


def _log_dry_run_scene(
    index: int,
    total: int,
    dataset: str,
    scene_name: str,
    source_path: Path,
    ground_truth_path: Path,
    output_path: Path,
    commands: Sequence[Sequence[str]],
    expected_mesh: Path,
    threshold: Optional[float],
    scaffold_output: Optional[Path] = None,
    init_ply: Optional[Path] = None,
) -> None:
    LOGGER.info("[DRY RUN] Dataset: %s", dataset)
    LOGGER.info("[DRY RUN] Scene %d/%d: %s", index, total, scene_name)
    LOGGER.info("[DRY RUN] Source: %s", source_path)
    LOGGER.info("[DRY RUN] Ground truth: %s", ground_truth_path)
    LOGGER.info("[DRY RUN] Output: %s", output_path)
    LOGGER.info("[DRY RUN] Expected mesh: %s", expected_mesh)
    LOGGER.info("[DRY RUN] Distance threshold: %s", threshold)
    if scaffold_output is not None:
        LOGGER.info("[DRY RUN] Scaffold output: %s", scaffold_output)
    if init_ply is not None:
        LOGGER.info("[DRY RUN] Scaffold initialization PLY: %s", init_ply)
    for label, command in zip(("Training", "Rendering", "Metrics"), commands):
        LOGGER.info("[DRY RUN] %s:", label)
        run_command(command, cwd=_repo_root(), dry_run=True)


def _run_preparation_phase(
    repo_root: Path,
    scenes: Sequence[Mapping[str, Any]],
    dry_run: bool,
    force_preprocess: bool,
) -> bool:
    """Prepare all selected scenes before allowing the experiment phase."""

    LOGGER.info("=== Dataset Preparation ===")
    results: List[PreparationResult] = []
    for index, scene in enumerate(scenes, start=1):
        scene_name = str(scene["name"])
        source_path = _resolve_path(repo_root, scene["source"])
        LOGGER.info("[PREPARE %d/%d] %s", index, len(scenes), scene_name)
        try:
            result = prepare_scene(
                scene_name,
                source_path,
                repo_root,
                dry_run=dry_run,
                force_preprocess=force_preprocess,
            )
        except Exception as exc:
            result = PreparationResult(
                scene_name,
                PreparationStatus.PARTIAL,
                PreparationAction.FAILED,
                False,
                "Unexpected preparation validation error: {}".format(exc),
            )
        results.append(result)
        LOGGER.info("Status: %s", result.initial_status.value)
        LOGGER.info("Action: %s", result.action.value)
        LOGGER.info("Result: %s", result.message)

    already_prepared = sum(
        result.action == PreparationAction.ALREADY_PREPARED for result in results
    )
    prepared_now = sum(
        result.action == PreparationAction.PREPARED_NOW for result in results
    )
    would_prepare = sum(
        result.action == PreparationAction.WOULD_PREPARE for result in results
    )
    failed = [result for result in results if not result.success]
    LOGGER.info("Preparation summary:")
    LOGGER.info("Already prepared: %d", already_prepared)
    LOGGER.info("Prepared now: %d", prepared_now)
    if dry_run:
        LOGGER.info("Would prepare: %d", would_prepare)
    LOGGER.info("Failed: %d", len(failed))

    if failed:
        LOGGER.error(
            "Dataset preparation failed for %d scene(s):", len(failed)
        )
        for result in failed:
            LOGGER.error("- %s: %s", result.scene, result.message)
        LOGGER.error("No training was started.")
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="Dataset configuration YAML"
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiment_default.yaml"),
        help=(
            "Experiment configuration YAML "
            "(default: configs/experiment_default.yaml)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Validate/prepare all selected scenes with the original convert.py",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run dataset preparation/validation and exit without experiments",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Request preprocessing again without automatic cleanup",
    )
    parser.add_argument("--scene", help="Run only the named scene")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.resume and args.force:
        parser.error("--resume and --force cannot be used together")
    if args.force_preprocess and not (args.prepare or args.prepare_only):
        parser.error("--force-preprocess requires --prepare or --prepare-only")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = _repo_root()
    experiment_config = load_yaml(_resolve_path(repo_root, args.experiment_config))
    dataset_config = load_yaml(_resolve_path(repo_root, args.config))

    experiment = _require_mapping(
        experiment_config.get("experiment"), "experiment"
    )
    training = _require_mapping(experiment_config.get("training"), "training")
    geometry_config = _require_mapping(
        experiment_config.get("geometry_evaluation"), "geometry_evaluation"
    )
    geometry_enabled, mesh_extraction_enabled = _resolve_stage_enablement(
        experiment_config, geometry_config
    )
    metric_config = _require_mapping(
        experiment_config.get("metrics"), "metrics"
    )
    monitoring = _require_mapping(
        experiment_config.get("monitoring"), "monitoring"
    )
    initialization_method = _initialization_method(experiment_config)
    scaffold_value = experiment_config.get("scaffold")
    scaffold_config: Optional[Mapping[str, Any]] = None
    if initialization_method == INITIALIZATION_METHOD:
        scaffold_config = _require_mapping(scaffold_value, "scaffold")
        if not bool(scaffold_config.get("enabled", False)):
            raise ValueError(
                "initialization.method=scaffold_anchor requires scaffold.enabled=true"
            )
    elif isinstance(scaffold_value, Mapping) and bool(
        scaffold_value.get("enabled", False)
    ):
        raise ValueError(
            "scaffold.enabled=true requires initialization.method=scaffold_anchor"
        )

    dataset_name = str(dataset_config.get("dataset_name", "")).strip()
    if not dataset_name:
        raise ValueError("dataset_name is required in the dataset config")
    scenes = dataset_config.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scenes must be a non-empty list")
    if args.scene:
        scenes = [scene for scene in scenes if scene.get("name") == args.scene]
        if not scenes:
            parser.error("scene '{}' is not present in {}".format(args.scene, args.config))

    validated_scenes = []
    for scene in scenes:
        scene_data = _require_mapping(scene, "scene entry")
        scene_name = str(scene_data.get("name", "")).strip()
        if not scene_name:
            raise ValueError("Each scene requires a non-empty name")
        if "source" not in scene_data:
            raise ValueError("Scene '{}' requires a source path".format(scene_name))
        validated_scenes.append(scene_data)
    scenes = validated_scenes

    iterations = int(training.get("iterations", 30_000))
    threshold_value = geometry_config.get("distance_threshold")
    threshold = None if threshold_value is None else float(threshold_value)
    sample_points = int(geometry_config.get("sample_points", 100_000))
    results_path = _resolve_path(repo_root, experiment["results_file"])

    LOGGER.info("Dataset: %s", dataset_name)
    LOGGER.info("Scenes selected: %d", len(scenes))
    LOGGER.info("Initialization method: %s", initialization_method)
    if args.prepare or args.prepare_only:
        preparation_succeeded = _run_preparation_phase(
            repo_root,
            scenes,
            dry_run=args.dry_run,
            force_preprocess=args.force_preprocess,
        )
        if not preparation_succeeded:
            return 1
        if args.prepare_only:
            LOGGER.info("Preparation-only mode complete; no training was started.")
            return 0

    successful = (
        read_successful_scenes(results_path, dataset_name) if args.resume else set()
    )
    if args.dry_run:
        result_logger = None
    else:
        try:
            result_logger = ResultLogger(
                results_path,
                extra_fields=(
                    SCAFFOLD_CSV_FIELDS
                    if initialization_method == INITIALIZATION_METHOD
                    else ()
                ),
            )
        except Exception as exc:
            LOGGER.critical(
                "Unable to initialize master results CSV %s: %s. "
                "Aborting batch because subsequent results cannot be recorded "
                "reliably.",
                results_path,
                exc,
                exc_info=True,
            )
            raise
    git_commit = get_git_commit(repo_root)
    failures = 0

    LOGGER.info("=== Experiment Phase ===")
    if args.dry_run:
        LOGGER.info("[DRY RUN] No training, rendering, metrics, or evaluation will run.")

    for index, scene in enumerate(scenes, start=1):
        scene_data = scene
        scene_name = str(scene_data["name"])
        if args.resume and scene_name in successful:
            LOGGER.info("Skipping successful scene due to --resume: %s", scene_name)
            continue

        source_path = _resolve_path(repo_root, scene_data["source"])
        ground_truth_path = _resolve_path(repo_root, scene_data["ground_truth"])
        output_path = _scene_output_path(
            repo_root,
            experiment,
            scene_data,
            source_path,
            initialization_method,
        )
        scaffold_output: Optional[Path] = None
        init_ply: Optional[Path] = None
        if initialization_method == INITIALIZATION_METHOD:
            assert scaffold_config is not None
            scaffold_output_root = _resolve_path(
                repo_root, scaffold_config["output_root"]
            )
            scaffold_output = (
                scaffold_output_root
                / _dataset_namespace(source_path)
                / scene_name
            )
            if args.dry_run:
                scaffold_stage = run_scaffold_stage(
                    repo_root,
                    scene_name,
                    source_path,
                    scaffold_output,
                    scaffold_config,
                    monitoring,
                    dry_run=True,
                )
                initialization = create_scaffold_initialization(
                    repo_root,
                    source_path,
                    scaffold_stage.point_cloud,
                    scaffold_output,
                    scaffold_stage.iterations,
                    scaffold_stage.scaffold_commit,
                    reuse_successful=bool(
                        scaffold_config.get("reuse_successful", True)
                    ),
                    dry_run=True,
                )
                init_ply = initialization.output_ply

        commands = _build_commands(
            repo_root,
            source_path,
            output_path,
            iterations,
            init_ply=init_ply,
            mesh_extraction_enabled=mesh_extraction_enabled,
        )

        if args.dry_run:
            _log_dry_run_scene(
                index,
                len(scenes),
                dataset_name,
                scene_name,
                source_path,
                ground_truth_path,
                output_path,
                commands,
                _expected_mesh_path(output_path, iterations),
                threshold,
                scaffold_output=scaffold_output,
                init_ply=init_ply,
            )
            continue

        LOGGER.info("Scene %d/%d: %s", index, len(scenes), scene_name)
        row = _new_result_row(
            dataset_name,
            scene_name,
            _path_for_record(repo_root, output_path),
            iterations,
            git_commit,
            threshold,
            sample_points,
            initialization_method,
        )
        monitor: Optional[GPUVRAMMonitor] = None
        current_stage = (
            "scaffold_training"
            if initialization_method == INITIALIZATION_METHOD
            else "2dgs_training"
        )
        try:
            if not source_path.is_dir():
                raise FileNotFoundError(
                    "Dataset source directory not found: {}".format(source_path)
                )

            if initialization_method == INITIALIZATION_METHOD:
                assert scaffold_config is not None
                assert scaffold_output is not None
                scaffold_stage = run_scaffold_stage(
                    repo_root,
                    scene_name,
                    source_path,
                    scaffold_output,
                    scaffold_config,
                    monitoring,
                )
                row["scaffold_iterations"] = scaffold_stage.iterations
                row["scaffold_training_seconds"] = (
                    scaffold_stage.training_seconds
                )
                row["scaffold_peak_vram_mb"] = scaffold_stage.peak_vram_mb
                row["scaffold_average_vram_mb"] = (
                    scaffold_stage.average_vram_mb
                )
                row["scaffold_commit"] = scaffold_stage.scaffold_commit
                row["scaffold_reused"] = scaffold_stage.reused

                current_stage = "scaffold_initialization"
                initialization = create_scaffold_initialization(
                    repo_root,
                    source_path,
                    scaffold_stage.point_cloud,
                    scaffold_output,
                    scaffold_stage.iterations,
                    scaffold_stage.scaffold_commit,
                    reuse_successful=bool(
                        scaffold_config.get("reuse_successful", True)
                    ),
                )
                init_ply = initialization.output_ply
                row["scaffold_anchor_count"] = initialization.anchor_count
                row["init_ply"] = _path_for_record(repo_root, init_ply)
                commands = _build_commands(
                    repo_root,
                    source_path,
                    output_path,
                    iterations,
                    init_ply=init_ply,
                    mesh_extraction_enabled=mesh_extraction_enabled,
                )

            current_stage = "2dgs_training"

            if bool(monitoring.get("gpu_vram", True)):
                monitor = GPUVRAMMonitor(
                    float(monitoring.get("gpu_sampling_interval_seconds", 0.5))
                )
                monitor.prepare()

            training_started = time.perf_counter()
            try:
                run_command(
                    commands[0],
                    cwd=repo_root,
                    on_process_started=monitor.start if monitor else None,
                )
            finally:
                elapsed = time.perf_counter() - training_started
                if bool(monitoring.get("training_time", True)):
                    row["training_seconds"] = round(elapsed, 6)
                    row["training_time"] = format_duration(elapsed)
                stats = monitor.stop() if monitor else VRAMStats(None, None)
                row["peak_vram_mb"] = stats.peak_vram_mb
                row["average_vram_mb"] = stats.average_vram_mb
                if stats.mode is not None:
                    LOGGER.info("VRAM monitoring mode: %s", stats.mode)
                if stats.peak_vram_mb is not None:
                    LOGGER.info("Peak VRAM: %.2f MiB", stats.peak_vram_mb)
                if stats.average_vram_mb is not None:
                    LOGGER.info("Average VRAM: %.2f MiB", stats.average_vram_mb)

            current_stage = "render"
            run_command(commands[1], cwd=repo_root)

            image_metrics_enabled = any(
                bool(metric_config.get(name, True))
                for name in ("ssim", "psnr", "lpips")
            )
            if image_metrics_enabled:
                current_stage = "metrics"
                results_json = output_path / "results.json"
                previous_mtime_ns = (
                    results_json.stat().st_mtime_ns if results_json.exists() else None
                )
                run_command(commands[2], cwd=repo_root)
                if not results_json.is_file():
                    raise FileNotFoundError(
                        "metrics.py did not create {}".format(results_json)
                    )
                if (
                    previous_mtime_ns is not None
                    and results_json.stat().st_mtime_ns == previous_mtime_ns
                ):
                    raise RuntimeError(
                        "metrics.py did not update {}; refusing to reuse stale "
                        "metrics".format(results_json)
                    )
                image_metrics = read_2dgs_metrics(results_json)
                for metric_name in ("ssim", "psnr", "lpips"):
                    if bool(metric_config.get(metric_name, True)):
                        row[metric_name] = image_metrics.get(metric_name)

            if geometry_enabled:
                current_stage = "geometry_evaluation"
                reconstructed_mesh = _find_reconstructed_mesh(output_path, iterations)
                geometry_metrics = evaluate_geometry(
                    reconstructed_mesh,
                    ground_truth_path,
                    threshold,
                    sample_points=sample_points,
                )
                geometry_values = asdict(geometry_metrics)
                for metric_name in (
                    "chamfer_distance",
                    "precision",
                    "recall",
                    "f1_score",
                ):
                    if bool(metric_config.get(metric_name, True)):
                        row[metric_name] = geometry_values[metric_name]

            row["status"] = "success"
        except Exception as exc:
            failures += 1
            row["status"] = "failed"
            row["error"] = "{}: {}: {}".format(
                current_stage, type(exc).__name__, exc
            )
            LOGGER.exception("Scene failed at %s: %s", current_stage, scene_name)
        finally:
            if initialization_method == INITIALIZATION_METHOD:
                scaffold_seconds = row.get("scaffold_training_seconds")
                training_seconds = row.get("training_seconds")
                if scaffold_seconds is not None and training_seconds is not None:
                    row["total_ablation_seconds"] = round(
                        float(scaffold_seconds) + float(training_seconds), 6
                    )
            row["timestamp"] = datetime.now(timezone.utc).isoformat()
            if result_logger is not None:
                try:
                    result_logger.append(row)
                except Exception as exc:
                    LOGGER.critical(
                        "Unable to write master results CSV %s: %s. "
                        "Unsaved scene result: %r. Aborting batch because "
                        "subsequent results cannot be recorded reliably.",
                        results_path,
                        exc,
                        row,
                        exc_info=True,
                    )
                    raise

    if args.dry_run:
        return 0
    LOGGER.info("Batch complete: %d failure(s)", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
