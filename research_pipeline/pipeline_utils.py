"""Shared utilities for the research pipeline."""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

LOGGER = logging.getLogger(__name__)
CommandPart = Union[str, Path]


class CommandExecutionError(RuntimeError):
    """Raised when an external command exits with a non-zero return code."""


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML mapping from *path*.

    PyYAML is imported lazily so modules that do not read configuration files can
    still be imported before the research dependencies are installed.
    """

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load experiment configuration. "
            "Install requirements-research.txt on the PC-server."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping: {}".format(config_path))
    return data


def get_git_commit(repo_root: Optional[Union[str, Path]] = None) -> str:
    """Return the current HEAD commit, or ``unknown`` when Git is unavailable."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(repo_root)) if repo_root is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        LOGGER.warning("Unable to determine the current Git commit.")
        return "unknown"


def format_duration(seconds: Optional[float]) -> Optional[str]:
    """Format a duration as ``HH:MM:SS`` using whole elapsed seconds."""

    if seconds is None:
        return None
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, secs)


def run_command(
    command: Sequence[CommandPart],
    cwd: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    on_process_started: Optional[Callable[[int], None]] = None,
) -> subprocess.CompletedProcess:
    """Run a command without a shell and raise a clear error on failure.

    ``on_process_started`` receives the child PID and is used by the GPU monitor
    without requiring any changes to the original 2DGS scripts.
    """

    if isinstance(command, (str, bytes)):
        raise TypeError("command must be a sequence of arguments, not a shell string")

    argv = [str(part) for part in command]
    if not argv:
        raise ValueError("command must not be empty")

    display_command = shlex.join(argv)
    if dry_run:
        LOGGER.info("[DRY RUN] %s", display_command)
        return subprocess.CompletedProcess(argv, 0)

    LOGGER.info("Running: %s", display_command)
    try:
        process = subprocess.Popen(argv, cwd=str(Path(cwd)) if cwd else None)
    except OSError as exc:
        raise CommandExecutionError(
            "Unable to start command '{}': {}".format(display_command, exc)
        ) from exc

    if on_process_started is not None:
        on_process_started(process.pid)

    return_code = process.wait()
    if return_code != 0:
        raise CommandExecutionError(
            "Command failed with exit code {}: {}".format(
                return_code, display_command
            )
        )
    return subprocess.CompletedProcess(argv, return_code)


def _coerce_metric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def read_2dgs_metrics(results_json_path: Union[str, Path]) -> Dict[str, Optional[float]]:
    """Read SSIM, PSNR, and LPIPS from a potentially nested 2DGS results file.

    The original ``metrics.py`` commonly stores metrics below a method key such
    as ``ours_30000``. This parser recursively finds metric dictionaries and
    prefers the most complete candidate with the highest numeric method suffix.
    """

    path = Path(results_json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metric_names = ("SSIM", "PSNR", "LPIPS")
    candidates = []

    def visit(node: Any, trail: List[str]) -> None:
        if isinstance(node, Mapping):
            normalized = {str(key).upper(): value for key, value in node.items()}
            metrics = {
                name.lower(): _coerce_metric(normalized.get(name))
                for name in metric_names
            }
            present = sum(value is not None for value in metrics.values())
            if present:
                numbers = [
                    int(number)
                    for part in trail
                    for number in re.findall(r"\d+", part)
                ]
                iteration_rank = max(numbers) if numbers else -1
                candidates.append((present, iteration_rank, len(trail), metrics))
            for key, value in node.items():
                visit(value, trail + [str(key)])
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, trail + [str(index)])

    visit(payload, [])
    if not candidates:
        LOGGER.warning("No SSIM, PSNR, or LPIPS values found in %s", path)
        return {"ssim": None, "psnr": None, "lpips": None}

    return max(candidates, key=lambda candidate: candidate[:3])[3]
