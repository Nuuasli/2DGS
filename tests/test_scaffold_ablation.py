"""CPU-only regression tests for Scaffold-anchor experiment orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from arguments import ModelParams

from research_pipeline.result_logger import (
    CSV_FIELDS,
    SCAFFOLD_CSV_FIELDS,
    ResultLogger,
)
from research_pipeline.gpu_monitor import VRAMStats
from research_pipeline.run_all import (
    _build_commands,
    _initialization_method,
    _scene_output_path,
)
from research_pipeline.scaffold_init import (
    INITIAL_VERTEX_DTYPE,
    _reusable_initialization,
    assign_nearest_colmap_colors,
    build_initial_vertex_array,
)
from research_pipeline.scaffold_runner import (
    _TrainingBoundaryTracker,
    ScaffoldStageError,
    reusable_scaffold_manifest,
    resolve_scaffold_python,
    resolve_scaffold_repo,
    run_scaffold_stage,
    scaffold_point_cloud_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class ScaffoldAblationTests(unittest.TestCase):
    def test_baseline_config_does_not_enable_scaffold(self) -> None:
        self.assertEqual(_initialization_method({}), "colmap")
        self.assertEqual(
            _initialization_method({"initialization": {"method": "colmap"}}),
            "colmap",
        )

    def test_scaffold_config_is_detected(self) -> None:
        self.assertEqual(
            _initialization_method(
                {"initialization": {"method": "scaffold_anchor"}}
            ),
            "scaffold_anchor",
        )

    def test_relative_scaffold_repo_resolves_from_project_root(self) -> None:
        resolved = resolve_scaffold_repo(REPO_ROOT, "external/Scaffold-GS")
        self.assertEqual(resolved, REPO_ROOT / "external" / "Scaffold-GS")

    def test_missing_scaffold_python_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ScaffoldStageError, "SCAFFOLD_PYTHON"):
            resolve_scaffold_python(REPO_ROOT, None, environ={})

    def test_point_cloud_selection_uses_exact_iteration(self) -> None:
        selected = scaffold_point_cloud_path(Path("model"), 12_345)
        self.assertEqual(
            selected,
            Path("model/point_cloud/iteration_12345/point_cloud.ply"),
        )

    def test_anchor_color_uses_nearest_colmap_xyz(self) -> None:
        anchors = np.array([[0.1, 0.0, 0.0], [9.8, 0.0, 0.0]])
        colmap_xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        colmap_rgb = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
        colors, distances = assign_nearest_colmap_colors(
            anchors, colmap_xyz, colmap_rgb
        )
        np.testing.assert_array_equal(colors, colmap_rgb)
        np.testing.assert_allclose(distances, [0.1, 0.2])

    def test_initial_point_schema_is_2dgs_compatible(self) -> None:
        vertices = build_initial_vertex_array(
            np.array([[1.0, 2.0, 3.0]]),
            np.array([[10, 20, 30]], dtype=np.uint8),
        )
        self.assertEqual(vertices.dtype, INITIAL_VERTEX_DTYPE)
        self.assertEqual(
            vertices.dtype.names,
            ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"),
        )
        self.assertEqual(float(vertices["nx"][0]), 0.0)
        self.assertEqual(int(vertices["green"][0]), 20)

    def test_initialization_manifest_tracks_scaffold_and_colmap_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dataset" / "scene"
            sparse = source / "sparse" / "0"
            sparse.mkdir(parents=True)
            colmap = sparse / "points3D.bin"
            colmap.write_bytes(b"colmap")
            scaffold = root / "scaffold.ply"
            scaffold.write_bytes(b"anchors")
            output = root / "initial_points.ply"
            output.write_bytes(b"initial")
            manifest_path = root / "initialization.json"

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "method": "scaffold_anchor",
                        "source_scene": str(source.resolve()),
                        "scaffold_point_cloud": str(scaffold.resolve()),
                        "scaffold_iterations": 30000,
                        "scaffold_commit": "abc123",
                        "scaffold_point_cloud_sha256": digest(scaffold),
                        "colmap_point_source": str(colmap.resolve()),
                        "colmap_point_source_sha256": digest(colmap),
                        "output_ply_sha256": digest(output),
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNotNone(
                _reusable_initialization(
                    manifest_path,
                    output,
                    source,
                    scaffold,
                    30000,
                    "abc123",
                )
            )
            colmap.write_bytes(b"changed")
            self.assertIsNone(
                _reusable_initialization(
                    manifest_path,
                    output,
                    source,
                    scaffold,
                    30000,
                    "abc123",
                )
            )

    def _successful_manifest_fixture(self, root: Path) -> tuple:
        source = root / "dataset" / "scene"
        output = root / "output"
        python = Path(sys.executable).resolve()
        iteration_dir = output / "point_cloud" / "iteration_30000"
        iteration_dir.mkdir(parents=True)
        artifacts = (
            "point_cloud.ply",
            "opacity_mlp.pt",
            "cov_mlp.pt",
            "color_mlp.pt",
            "embedding_appearance.pt",
        )
        for name in artifacts:
            (iteration_dir / name).write_bytes(name.encode("ascii"))
        point_cloud = iteration_dir / "point_cloud.ply"
        digest = hashlib.sha256(point_cloud.read_bytes()).hexdigest()
        manifest_path = output / "scaffold_stage.json"
        manifest = {
            "status": "success",
            "scene": "scene",
            "source_scene": str(source.resolve()),
            "iterations": 30000,
            "resolution": 1,
            "eval": True,
            "scaffold_commit": "abc123",
            "python_executable": str(python),
            "point_cloud_sha256": digest,
            "training_seconds": 1.25,
            "training_boundary_detected": True,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return source, output, python, manifest_path, iteration_dir

    def test_success_manifest_reuses_only_valid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, output, python, manifest_path, _ = self._successful_manifest_fixture(
                Path(temporary)
            )
            manifest = reusable_scaffold_manifest(
                manifest_path,
                "scene",
                source,
                output,
                30000,
                1,
                True,
                "abc123",
                python,
            )
            self.assertIsNotNone(manifest)

    def test_success_manifest_with_missing_checkpoint_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, output, python, manifest_path, iteration_dir = (
                self._successful_manifest_fixture(Path(temporary))
            )
            (iteration_dir / "cov_mlp.pt").unlink()
            manifest = reusable_scaffold_manifest(
                manifest_path,
                "scene",
                source,
                output,
                30000,
                1,
                True,
                "abc123",
                python,
            )
            self.assertIsNone(manifest)

    def test_dry_run_does_not_execute_training_subprocess(self) -> None:
        calls = []

        def fake_runner(command, **kwargs):
            calls.append((tuple(command), kwargs))
            self.assertTrue(kwargs.get("dry_run"))
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temporary:
            result = run_scaffold_stage(
                REPO_ROOT,
                "placeholder",
                Path(temporary) / "missing_dataset",
                Path(temporary) / "output",
                {
                    "repo_path": "external/Scaffold-GS",
                    "python_executable": str(Path(sys.executable).resolve()),
                    "iterations": 7,
                    "resolution": 1,
                    "eval": True,
                    "reuse_successful": False,
                },
                {"gpu_vram": False},
                dry_run=True,
                command_runner=fake_runner,
                environ={},
            )
        self.assertEqual(len(calls), 1)
        self.assertIn("--iterations", calls[0][0])
        self.assertEqual(result.iterations, 7)

    def test_scaffold_monitor_stops_at_training_boundary(self) -> None:
        class FakeMonitor:
            def __init__(self) -> None:
                self.started_pid = None
                self.stop_count = 0

            def start(self, pid: int) -> None:
                self.started_pid = pid

            def stop(self) -> VRAMStats:
                self.stop_count += 1
                return VRAMStats(10.0, 8.0, "process")

        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "outputs.log"
            monitor = FakeMonitor()
            tracker = _TrainingBoundaryTracker(log_path, monitor, 0.5)
            tracker.process_started(123)
            log_path.write_text("Training complete.\n", encoding="utf-8")
            elapsed, stats = tracker.finish()
        self.assertTrue(tracker.boundary_found)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(monitor.started_pid, 123)
        self.assertEqual(monitor.stop_count, 1)
        self.assertEqual(stats.peak_vram_mb, 10.0)

    def test_baseline_train_command_has_no_init_ply(self) -> None:
        train, _, _ = _build_commands(
            REPO_ROOT, Path("source"), Path("output"), 30000
        )
        self.assertNotIn("--init_ply", train)

    def test_scaffold_train_command_has_init_ply(self) -> None:
        train, _, _ = _build_commands(
            REPO_ROOT,
            Path("source"),
            Path("output"),
            30000,
            init_ply=Path("initial_points.ply"),
        )
        index = train.index("--init_ply")
        self.assertEqual(train[index + 1], "initial_points.ply")

    def test_2dgs_model_cli_accepts_init_ply(self) -> None:
        parser = ArgumentParser()
        model_params = ModelParams(parser)
        parsed = parser.parse_args(["--init_ply", "initial_points.ply"])
        self.assertEqual(
            model_params.extract(parsed).init_ply, "initial_points.ply"
        )

    def test_baseline_output_and_result_path_are_unchanged(self) -> None:
        baseline_config = {
            "experiment": {
                "name": "2dgs_30000_visual",
                "output_root": "output",
                "results_file": (
                    "research_results/2dgs_30000_visual_results.csv"
                ),
            },
            "training": {"iterations": 30000},
        }
        scene = {
            "name": "object",
            "source": "datasets/set/object",
            "output": "output/set/object",
        }
        initialization_method = _initialization_method(baseline_config)
        output = _scene_output_path(
            REPO_ROOT,
            baseline_config["experiment"],
            scene,
            REPO_ROOT / scene["source"],
            initialization_method,
        )

        self.assertEqual(initialization_method, "colmap")
        self.assertEqual(
            baseline_config["experiment"]["output_root"], "output"
        )
        self.assertEqual(
            baseline_config["experiment"]["results_file"],
            "research_results/2dgs_30000_visual_results.csv",
        )
        self.assertEqual(output, REPO_ROOT / "output/set/object")
        self.assertNotIn(
            "output/ablations/scaffold_anchor", output.as_posix()
        )

    def test_scaffold_uses_separate_output_and_result_path(self) -> None:
        scene = {
            "name": "object",
            "source": "datasets/set/object",
            "output": "output/set/object",
        }
        output = _scene_output_path(
            REPO_ROOT,
            {"output_root": "output/ablations/scaffold_anchor/2dgs"},
            scene,
            REPO_ROOT / scene["source"],
            "scaffold_anchor",
        )
        self.assertEqual(
            output,
            REPO_ROOT / "output/ablations/scaffold_anchor/2dgs/set/object",
        )
        scaffold_config = (
            REPO_ROOT / "configs/experiment_30000_scaffold_anchor.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "results_file: research_results/2dgs_scaffold_anchor_results.csv",
            scaffold_config,
        )

    def test_scene_count_is_driven_by_input_list(self) -> None:
        scenes = [{"name": "scene_{}".format(index)} for index in range(19)]
        self.assertEqual(len(scenes), 19)
        source = (REPO_ROOT / "research_pipeline/run_all.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("range(14)", source)
        self.assertNotIn("range(16)", source)

    def test_baseline_and_scaffold_csv_schemas_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            baseline = Path(temporary) / "baseline.csv"
            scaffold = Path(temporary) / "scaffold.csv"
            ResultLogger(baseline)
            ResultLogger(scaffold, extra_fields=SCAFFOLD_CSV_FIELDS)
            with baseline.open(newline="", encoding="utf-8") as handle:
                baseline_header = next(csv.reader(handle))
            with scaffold.open(newline="", encoding="utf-8") as handle:
                scaffold_header = next(csv.reader(handle))
        self.assertEqual(baseline_header, CSV_FIELDS)
        self.assertEqual(scaffold_header, [*CSV_FIELDS, *SCAFFOLD_CSV_FIELDS])


if __name__ == "__main__":
    unittest.main()
