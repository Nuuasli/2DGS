"""Run the official Scaffold-GS trainer as an isolated per-scene stage."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from research_pipeline.gpu_monitor import GPUVRAMMonitor, VRAMStats
from research_pipeline.pipeline_utils import get_git_commit, run_command

LOGGER = logging.getLogger(__name__)
MANIFEST_NAME = "scaffold_stage.json"
TRAINING_COMPLETE_MARKER = "Training complete."


class ScaffoldStageError(RuntimeError):
    """Raised when the external Scaffold-GS stage is invalid or fails."""


@dataclass(frozen=True)
class ScaffoldStageResult:
    """Validated result of one Scaffold-GS training stage."""

    point_cloud: Path
    output_path: Path
    manifest_path: Path
    iterations: int
    training_seconds: Optional[float]
    peak_vram_mb: Optional[float]
    average_vram_mb: Optional[float]
    scaffold_commit: str
    reused: bool
    command: Tuple[str, ...]
    python_executable: str


def resolve_scaffold_repo(repo_root: Path, configured_path: Any) -> Path:
    """Resolve a Scaffold repository path relative to the main repository."""

    path = Path(str(configured_path))
    return path if path.is_absolute() else repo_root / path


def resolve_scaffold_python(
    repo_root: Path, configured_python: Any, environ: Optional[Mapping[str, str]] = None
) -> Path:
    """Resolve ``SCAFFOLD_PYTHON`` first, then the configured executable."""

    environment = os.environ if environ is None else environ
    value = str(environment.get("SCAFFOLD_PYTHON") or configured_python or "").strip()
    if not value:
        raise ScaffoldStageError(
            "Scaffold Python executable is not configured. Set SCAFFOLD_PYTHON "
            "or scaffold.python_executable in the experiment config."
        )

    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_absolute() or expanded.parent != Path("."):
        candidate = expanded if expanded.is_absolute() else repo_root / expanded
        if not candidate.is_file():
            raise ScaffoldStageError(
                "Scaffold Python executable not found: {}".format(candidate)
            )
        return candidate.resolve()

    discovered = shutil.which(value)
    if discovered is None:
        raise ScaffoldStageError(
            "Scaffold Python executable not found on PATH: {}".format(value)
        )
    return Path(discovered).resolve()


def scaffold_iteration_dir(output_path: Path, iterations: int) -> Path:
    """Return the official Scaffold-GS save directory for one iteration."""

    return output_path / "point_cloud" / "iteration_{}".format(iterations)


def scaffold_point_cloud_path(output_path: Path, iterations: int) -> Path:
    """Return the exact anchor PLY path for the requested iteration."""

    return scaffold_iteration_dir(output_path, iterations) / "point_cloud.ply"


def build_scaffold_command(
    python_executable: Path,
    scaffold_repo: Path,
    source_path: Path,
    output_path: Path,
    iterations: int,
    resolution: int,
    eval_enabled: bool,
) -> Tuple[str, ...]:
    """Build a command using arguments defined by official Scaffold-GS."""

    command = [
        str(python_executable),
        "-u",
        str(scaffold_repo / "train.py"),
        "-s",
        str(source_path),
        "-m",
        str(output_path),
        "--iterations",
        str(iterations),
        "-r",
        str(resolution),
    ]
    if eval_enabled:
        command.append("--eval")
    return tuple(command)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_artifacts(output_path: Path, iterations: int) -> Tuple[Path, ...]:
    directory = scaffold_iteration_dir(output_path, iterations)
    return (
        directory / "point_cloud.ply",
        directory / "opacity_mlp.pt",
        directory / "cov_mlp.pt",
        directory / "color_mlp.pt",
        directory / "embedding_appearance.pt",
    )


def _artifacts_valid(output_path: Path, iterations: int) -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in _required_artifacts(output_path, iterations)
    )


def _read_manifest(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, Mapping) else None
    except (OSError, ValueError):
        return None


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


class _TrainingBoundaryTracker:
    """Stop timing/VRAM at Scaffold's official end-of-training log marker."""

    def __init__(
        self,
        log_path: Path,
        monitor: Optional[GPUVRAMMonitor],
        sampling_interval: float,
    ) -> None:
        self.log_path = log_path
        self.monitor = monitor
        self.poll_interval = min(0.25, sampling_interval)
        self.initial_size = log_path.stat().st_size if log_path.is_file() else 0
        self.started_at = time.perf_counter()
        self.elapsed: Optional[float] = None
        self.stats: Optional[VRAMStats] = None
        self._stop_event = threading.Event()
        self._boundary_event = threading.Event()
        self._finish_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @property
    def boundary_found(self) -> bool:
        return self._boundary_event.is_set()

    def process_started(self, pid: int) -> None:
        if self.monitor is not None:
            self.monitor.start(pid)
        self._thread = threading.Thread(
            target=self._watch,
            name="scaffold-training-boundary",
            daemon=True,
        )
        self._thread.start()

    def _new_log_contains_marker(self) -> bool:
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self.initial_size)
                return TRAINING_COMPLETE_MARKER in handle.read()
        except OSError:
            return False

    def _finish_at_boundary(self) -> None:
        with self._finish_lock:
            if self._boundary_event.is_set():
                return
            self.elapsed = time.perf_counter() - self.started_at
            self.stats = (
                self.monitor.stop()
                if self.monitor is not None
                else VRAMStats(None, None)
            )
            self._boundary_event.set()

    def _watch(self) -> None:
        while not self._stop_event.is_set():
            if self._new_log_contains_marker():
                self._finish_at_boundary()
                return
            self._stop_event.wait(self.poll_interval)

    def finish(self) -> Tuple[float, VRAMStats]:
        """Return training-only measurements, falling back to process duration."""

        if not self.boundary_found and self._new_log_contains_marker():
            self._finish_at_boundary()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_interval * 4))
        if self.boundary_found:
            assert self.elapsed is not None and self.stats is not None
            return self.elapsed, self.stats
        elapsed = time.perf_counter() - self.started_at
        stats = (
            self.monitor.stop() if self.monitor is not None else VRAMStats(None, None)
        )
        return elapsed, stats


def reusable_scaffold_manifest(
    manifest_path: Path,
    scene_name: str,
    source_path: Path,
    output_path: Path,
    iterations: int,
    resolution: int,
    eval_enabled: bool,
    scaffold_commit: str,
    python_executable: Path,
) -> Optional[Mapping[str, Any]]:
    """Return a successful manifest only when config and artifacts still match."""

    manifest = _read_manifest(manifest_path)
    expected = {
        "status": "success",
        "scene": scene_name,
        "source_scene": str(source_path.resolve()),
        "iterations": iterations,
        "resolution": resolution,
        "eval": eval_enabled,
        "scaffold_commit": scaffold_commit,
        "python_executable": str(python_executable),
        "training_boundary_detected": True,
    }
    if manifest is None or any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if not _artifacts_valid(output_path, iterations):
        return None
    point_cloud = scaffold_point_cloud_path(output_path, iterations)
    if manifest.get("point_cloud_sha256") != _sha256(point_cloud):
        return None
    return manifest


def run_scaffold_stage(
    repo_root: Path,
    scene_name: str,
    source_path: Path,
    output_path: Path,
    scaffold_config: Mapping[str, Any],
    monitoring_config: Mapping[str, Any],
    dry_run: bool = False,
    command_runner: Callable[..., Any] = run_command,
    environ: Optional[Mapping[str, str]] = None,
) -> ScaffoldStageResult:
    """Run or safely reuse Scaffold-GS for one scene."""

    scaffold_repo = resolve_scaffold_repo(
        repo_root, scaffold_config.get("repo_path", "external/Scaffold-GS")
    ).resolve()
    train_script = scaffold_repo / "train.py"
    if not scaffold_repo.is_dir() or not train_script.is_file():
        raise ScaffoldStageError(
            "Scaffold-GS repository or train.py is unavailable: {}".format(scaffold_repo)
        )
    if not dry_run and not source_path.is_dir():
        raise ScaffoldStageError(
            "Scaffold dataset source directory not found: {}".format(source_path)
        )

    python_executable = resolve_scaffold_python(
        repo_root, scaffold_config.get("python_executable"), environ=environ
    )
    iterations = int(scaffold_config.get("iterations", 30_000))
    resolution = int(scaffold_config.get("resolution", 1))
    eval_enabled = bool(scaffold_config.get("eval", True))
    scaffold_commit = get_git_commit(scaffold_repo)
    manifest_path = output_path / MANIFEST_NAME
    point_cloud = scaffold_point_cloud_path(output_path, iterations)
    command = build_scaffold_command(
        python_executable,
        scaffold_repo,
        source_path,
        output_path,
        iterations,
        resolution,
        eval_enabled,
    )

    if bool(scaffold_config.get("reuse_successful", True)):
        manifest = reusable_scaffold_manifest(
            manifest_path,
            scene_name,
            source_path,
            output_path,
            iterations,
            resolution,
            eval_enabled,
            scaffold_commit,
            python_executable,
        )
        if manifest is not None:
            LOGGER.info("Reusing validated Scaffold-GS stage: %s", point_cloud)
            return ScaffoldStageResult(
                point_cloud,
                output_path,
                manifest_path,
                iterations,
                manifest.get("training_seconds"),
                manifest.get("peak_vram_mb"),
                manifest.get("average_vram_mb"),
                scaffold_commit,
                True,
                command,
                str(python_executable),
            )

    if dry_run:
        command_runner(command, cwd=scaffold_repo, dry_run=True)
        return ScaffoldStageResult(
            point_cloud,
            output_path,
            manifest_path,
            iterations,
            None,
            None,
            None,
            scaffold_commit,
            False,
            command,
            str(python_executable),
        )

    monitor: Optional[GPUVRAMMonitor] = None
    if bool(monitoring_config.get("gpu_vram", True)):
        monitor = GPUVRAMMonitor(
            float(monitoring_config.get("gpu_sampling_interval_seconds", 0.5))
        )
        monitor.prepare()

    tracker = _TrainingBoundaryTracker(
        output_path / "outputs.log",
        monitor,
        float(monitoring_config.get("gpu_sampling_interval_seconds", 0.5)),
    )
    command_error: Optional[Exception] = None
    try:
        command_runner(
            command,
            cwd=scaffold_repo,
            on_process_started=tracker.process_started,
        )
    except Exception as exc:
        command_error = exc

    elapsed, stats = tracker.finish()
    artifacts_valid = _artifacts_valid(output_path, iterations)
    if command_error is not None and not (
        tracker.boundary_found and artifacts_valid
    ):
        _write_manifest(
            manifest_path,
            {
                "status": "failed",
                "scene": scene_name,
                "source_scene": str(source_path.resolve()),
                "iterations": iterations,
                "resolution": resolution,
                "eval": eval_enabled,
                "training_seconds": round(elapsed, 6),
                "peak_vram_mb": stats.peak_vram_mb,
                "average_vram_mb": stats.average_vram_mb,
                "scaffold_commit": scaffold_commit,
                "command": list(command),
                "point_cloud": str(point_cloud),
                "python_executable": str(python_executable),
                "training_boundary_detected": tracker.boundary_found,
                "error": "{}: {}".format(
                    type(command_error).__name__, command_error
                ),
            },
        )
        raise ScaffoldStageError(
            "Scaffold-GS training failed: {}".format(command_error)
        ) from command_error

    if command_error is not None:
        LOGGER.warning(
            "Scaffold-GS training and anchor save completed, but its optional "
            "post-training render/metrics exited with an error: %s. The "
            "validated anchor checkpoint will still be used.",
            command_error,
        )

    if not artifacts_valid:
        missing = [
            str(path)
            for path in _required_artifacts(output_path, iterations)
            if not path.is_file() or path.stat().st_size == 0
        ]
        _write_manifest(
            manifest_path,
            {
                "status": "failed",
                "scene": scene_name,
                "source_scene": str(source_path.resolve()),
                "iterations": iterations,
                "resolution": resolution,
                "eval": eval_enabled,
                "training_seconds": round(elapsed, 6),
                "peak_vram_mb": stats.peak_vram_mb,
                "average_vram_mb": stats.average_vram_mb,
                "scaffold_commit": scaffold_commit,
                "command": list(command),
                "point_cloud": str(point_cloud),
                "python_executable": str(python_executable),
                "training_boundary_detected": tracker.boundary_found,
                "error": "Required artifacts missing or empty: {}".format(
                    ", ".join(missing)
                ),
            },
        )
        raise ScaffoldStageError(
            "Scaffold-GS completed but required iteration {} artifacts are "
            "missing or empty: {}".format(iterations, ", ".join(missing))
        )

    payload = {
        "status": "success",
        "scene": scene_name,
        "source_scene": str(source_path.resolve()),
        "iterations": iterations,
        "resolution": resolution,
        "eval": eval_enabled,
        "training_seconds": round(elapsed, 6),
        "peak_vram_mb": stats.peak_vram_mb,
        "average_vram_mb": stats.average_vram_mb,
        "scaffold_commit": scaffold_commit,
        "command": list(command),
        "point_cloud": str(point_cloud),
        "point_cloud_sha256": _sha256(point_cloud),
        "python_executable": str(python_executable),
        "training_boundary_detected": tracker.boundary_found,
    }
    if command_error is not None:
        payload["post_training_error"] = "{}: {}".format(
            type(command_error).__name__, command_error
        )
    _write_manifest(manifest_path, payload)
    return ScaffoldStageResult(
        point_cloud,
        output_path,
        manifest_path,
        iterations,
        payload["training_seconds"],
        stats.peak_vram_mb,
        stats.average_vram_mb,
        scaffold_commit,
        False,
        command,
        str(python_executable),
    )
