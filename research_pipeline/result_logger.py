"""Append one durable CSV result row for every processed scene."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Set, Union

CSV_FIELDS = [
    "run_id",
    "dataset",
    "scene",
    "status",
    "chamfer_distance",
    "precision",
    "recall",
    "f1_score",
    "distance_threshold",
    "geometry_sample_points",
    "ssim",
    "psnr",
    "lpips",
    "training_seconds",
    "training_time",
    "peak_vram_mb",
    "average_vram_mb",
    "iterations",
    "git_commit",
    "timestamp",
    "output_path",
    "error",
]


class ResultLogger:
    """Create a result CSV when needed and append rows without buffering them."""

    def __init__(self, csv_path: Union[str, Path]) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, result: Mapping[str, Any]) -> None:
        """Append and immediately flush a scene result to disk."""

        unknown_fields = set(result) - set(CSV_FIELDS)
        if unknown_fields:
            raise ValueError(
                "Unknown result fields: {}".format(
                    ", ".join(sorted(unknown_fields))
                )
            )
        row = {field: result.get(field) for field in CSV_FIELDS}
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


def read_successful_scenes(
    csv_path: Union[str, Path], dataset: str
) -> Set[str]:
    """Return scene names with a successful row for the selected dataset."""

    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return set()

    successful = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") == dataset and row.get("status") == "success":
                scene = row.get("scene")
                if scene:
                    successful.add(scene)
    return successful
