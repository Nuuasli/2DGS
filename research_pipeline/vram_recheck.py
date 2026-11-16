#!/usr/bin/env python3
"""Isolated VRAM remeasurement for 2DGS and Scaffold-GS experiments.

Run from the 2DGS repository root in the 2dgs conda environment.
All training outputs and results are separate from the original experiment.
WSL NVML fallback measures GPU device-level memory delta, not isolated process VRAM.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone


STAGES = {
    "baseline2dgs": {"detya", "anggada", "hanoman", "dewi_saraswati"},
    "scaffold2dgs": {"dewa_brahma", "dewa_sambhu", "dewi_gangga"},
    "scaffold": {"merdah", "anggada"},
}
RESULTS_COLUMNS = [
    "run_id", "timestamp_utc", "scene", "stage", "status", "training_marker_seen",
    "exit_code", "peak_vram_mb", "average_vram_mb", "sample_count",
    "baseline_device_used_mb", "monitoring_mode", "sample_interval_s",
    "training_seconds", "output_path", "log_path", "command", "git_commit",
    "note",
]


def atomic_csv_append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RESULTS_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def has_previous_success(csv_path, stage, scene):
    if not csv_path.is_file():
        return False
    with csv_path.open(newline="") as f:
        return any(
            r.get("stage") == stage and r.get("scene") == scene
            and r.get("status") == "success"
            and r.get("peak_vram_mb") not in (None, "")
            for r in csv.DictReader(f)
        )


def nvml_init(gpu_index):
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return pynvml, handle
    except Exception as exc:
        raise RuntimeError(f"NVML unavailable: {type(exc).__name__}: {exc}")


def mem_mb(nvml, handle):
    return nvml.nvmlDeviceGetMemoryInfo(handle).used / (1024.0 ** 2)


def git_commit(repo):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ""


def build_command(args, repo, output):
    source = repo / "datasets" / "togog_bali" / args.scene
    if not source.is_dir():
        raise RuntimeError(f"Dataset source does not exist: {source}")
    if args.stage == "scaffold":
        python = os.environ.get("SCAFFOLD_PYTHON", "")
        if not python or not Path(python).is_file():
            raise RuntimeError("Set SCAFFOLD_PYTHON to the Scaffold conda python")
        cmd = [
            python, "-u", str(repo / "external/Scaffold-GS/train.py"),
            "-s", str(source), "-m", str(output),
            "--iterations", "30000", "-r", "1", "--eval",
        ]
    else:
        if not (repo / "train.py").is_file():
            raise RuntimeError(f"train.py not found under repo: {repo}")
        cmd = [
            sys.executable, "-u", str(repo / "train.py"),
            "--source_path", str(source), "--model_path", str(output),
            "--iterations", "30000", "--eval",
        ]
        if args.stage == "scaffold2dgs":
            init_ply = (
                repo / "output/ablations/scaffold_anchor/scaffold/togog_bali"
                / args.scene / "initial_points.ply"
            )
            if not init_ply.is_file():
                raise RuntimeError(f"Original Scaffold initial_points.ply not found: {init_ply}")
            cmd += ["--init_ply", str(init_ply)]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--max-baseline-mb", type=float, default=2500.0,
                        help="Abort if GPU memory is already above this threshold")
    parser.add_argument("--allow-repeat", action="store_true")
    args = parser.parse_args()
    if args.scene not in STAGES[args.stage]:
        parser.error(f"Only these verified missing-VRAM scenes are accepted for {args.stage}: "
                     + ", ".join(sorted(STAGES[args.stage])))
    if args.interval <= 0:
        parser.error("--interval must be positive")

    repo = Path.cwd().resolve()
    results_csv = repo / "research_results/vram_recheck_results.csv"
    if has_previous_success(results_csv, args.stage, args.scene) and not args.allow_repeat:
        parser.error("This stage/scene already has a successful remeasurement; "
                     "use --allow-repeat only if a new measurement is intended")

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    output = repo / "output/vram_rechecks" / args.stage / "togog_bali" / args.scene / run_id
    log_path = repo / "research_results/vram_recheck_logs" / f"{args.stage}_{args.scene}_{run_id}.log"
    cmd = build_command(args, repo, output)
    print("Stage:", args.stage, "| Scene:", args.scene, flush=True)
    print("Training output (new, isolated):", output, flush=True)
    print("Log:", log_path, flush=True)
    print("VRAM CSV (separate):", results_csv, flush=True)
    print("Command:", subprocess.list2cmdline(cmd), flush=True)
    if args.dry_run:
        print("DRY RUN: no training, GPU monitoring, or file writes.", flush=True)
        return 0
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {output}")

    nvml, handle = nvml_init(args.gpu_index)
    try:
        before = mem_mb(nvml, handle)
        print(f"GPU memory used before training: {before:.2f} MiB", flush=True)
        if before > args.max_baseline_mb:
            raise RuntimeError(
                f"GPU is busy (> {args.max_baseline_mb:.0f} MiB baseline). "
                "Use an idle GPU time to reduce interference and OOM risk."
            )
        print("WARNING: mode=device_delta (WSL): other GPU jobs can contaminate VRAM.", flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        done = threading.Event()
        samples = []
        errors = []
        marker_timestamp = [None]
        started = time.monotonic()
        with log_path.open("w", buffering=1) as logfile:
            proc = subprocess.Popen(
                cmd, cwd=str(repo), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
            )

            def read_output():
                try:
                    for line in proc.stdout:
                        logfile.write(line)
                        print(line, end="", flush=True)
                        if "Training complete." in line and not done.is_set():
                            marker_timestamp[0] = time.monotonic()
                            done.set()
                finally:
                    if proc.stdout:
                        proc.stdout.close()

            thread = threading.Thread(target=read_output, daemon=True)
            thread.start()
            while not done.is_set() and proc.poll() is None:
                try:
                    value = max(0.0, mem_mb(nvml, handle) - before)
                    samples.append(value)
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                if done.wait(timeout=args.interval):
                    break
            try:
                returncode = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                returncode = proc.wait()
                raise
            finally:
                thread.join(timeout=10)

        peak = max(samples) if samples else None
        avg = sum(samples) / len(samples) if samples else None
        valid = done.is_set() and returncode == 0 and bool(samples)
        row = dict(
            run_id=run_id, timestamp_utc=datetime.now(timezone.utc).isoformat(),
            scene=args.scene, stage=args.stage, status="success" if valid else "failed",
            training_marker_seen=done.is_set(), exit_code=returncode,
            peak_vram_mb=f"{peak:.6f}" if peak is not None else "",
            average_vram_mb=f"{avg:.6f}" if avg is not None else "",
            sample_count=len(samples), baseline_device_used_mb=f"{before:.3f}",
            monitoring_mode="device_delta", sample_interval_s=args.interval,
            training_seconds=f"{marker_timestamp[0]-started:.6f}" if marker_timestamp[0] else "",
            output_path=str(output.relative_to(repo)), log_path=str(log_path.relative_to(repo)),
            command=json.dumps(cmd), git_commit=git_commit(repo),
            note="Independent VRAM remeasurement; NOT the original completed run."
                 + (" Monitoring errors: " + "; ".join(errors[:3]) if errors else ""),
        )
        atomic_csv_append(results_csv, row)
        print("\n=== VRAM REMEASUREMENT ===", flush=True)
        print("Status:", row["status"], "| samples:", len(samples), flush=True)
        print("Peak (MiB):", row["peak_vram_mb"], flush=True)
        print("Average (MiB):", row["average_vram_mb"], flush=True)
        print("Saved to:", results_csv, flush=True)
        return 0 if valid else 1
    finally:
        nvml.nvmlShutdown()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
