"""Optional process-level VRAM monitoring with a device-delta fallback."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)
BYTES_PER_MIB = 1024 * 1024


@dataclass
class VRAMStats:
    """Summary of sampled training VRAM usage in MiB."""

    peak_vram_mb: Optional[float]
    average_vram_mb: Optional[float]
    mode: Optional[str] = None


class GPUVRAMMonitor:
    """Sample training VRAM on a background thread.

    Per-process NVML usage is preferred. When NVML can find the target PID but
    cannot report its memory usage (as can happen under WSL), the monitor uses
    device memory growth relative to a baseline captured before training.

    NVML is loaded lazily by ``prepare``. Importing this module is safe on
    development machines without an NVIDIA GPU or ``nvidia-ml-py``.
    """

    def __init__(self, sampling_interval_seconds: float = 0.5) -> None:
        if sampling_interval_seconds <= 0:
            raise ValueError("GPU sampling interval must be greater than zero")
        self.sampling_interval_seconds = sampling_interval_seconds
        self._pid: Optional[int] = None
        self._samples_mb: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pynvml: Optional[Any] = None
        self._initialized = False
        self._prepare_attempted = False
        self._baseline_used_bytes: Dict[int, int] = {}
        self._device_delta_indices: List[int] = []
        self._mode: Optional[str] = None
        self._fallback_warning_emitted = False

    def prepare(self) -> bool:
        """Initialize NVML and record per-device memory before training.

        All failures are non-fatal. ``False`` means monitoring is unavailable,
        while training may continue normally.
        """

        if self._initialized:
            return True
        if self._prepare_attempted:
            return False
        self._prepare_attempted = True

        try:
            import pynvml
        except ImportError:
            LOGGER.warning(
                "nvidia-ml-py is unavailable; VRAM metrics will be empty."
            )
            return False

        try:
            pynvml.nvmlInit()
        except Exception as exc:  # NVML raises library-specific exceptions.
            LOGGER.warning("NVML could not be initialized: %s", exc)
            return False

        self._pynvml = pynvml
        self._initialized = True
        self._baseline_used_bytes.clear()
        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except Exception as exc:
            LOGGER.warning("Unable to enumerate GPUs for VRAM monitoring: %s", exc)
            return True

        for device_index in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                used_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).used
                if self._is_valid_memory_value(used_bytes):
                    self._baseline_used_bytes[device_index] = used_bytes
                else:
                    LOGGER.warning(
                        "Unable to record a valid VRAM baseline for GPU %d.",
                        device_index,
                    )
            except Exception as exc:
                LOGGER.warning(
                    "Unable to record VRAM baseline for GPU %d: %s",
                    device_index,
                    exc,
                )
        return True

    def start(self, pid: int) -> None:
        """Start monitoring *pid*, preparing automatically when necessary."""

        self._pid = pid
        if not self.prepare():
            return
        if self._thread is not None and self._thread.is_alive():
            LOGGER.debug("VRAM monitor is already running for PID %s", self._pid)
            return

        self._stop_event.clear()
        try:
            self._thread = threading.Thread(
                target=self._sample_loop,
                name="training-vram-monitor",
                daemon=True,
            )
            self._thread.start()
        except Exception as exc:
            self._thread = None
            LOGGER.warning("Unable to start the VRAM monitoring thread: %s", exc)

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._read_memory_mb()
            if sample is not None:
                self._samples_mb.append(sample)
            self._stop_event.wait(self.sampling_interval_seconds)

    def _is_valid_memory_value(self, value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
        unavailable = getattr(self._pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
        return unavailable is None or value != unavailable

    def _scan_target_processes(self) -> Tuple[List[int], Dict[int, int]]:
        if self._pynvml is None or self._pid is None:
            return [], {}

        pynvml = self._pynvml
        found_indices: List[int] = []
        valid_usage_by_device: Dict[int, int] = {}
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for device_index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                processes_by_pid: Dict[int, Any] = {}
                for query_name in (
                    "nvmlDeviceGetComputeRunningProcesses",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                ):
                    query = getattr(pynvml, query_name, None)
                    if query is None:
                        continue
                    try:
                        for process in query(handle):
                            existing = processes_by_pid.get(process.pid)
                            existing_usage = getattr(
                                existing, "usedGpuMemory", None
                            )
                            process_usage = getattr(
                                process, "usedGpuMemory", None
                            )
                            if existing is None or (
                                not self._is_valid_memory_value(existing_usage)
                                and self._is_valid_memory_value(process_usage)
                            ):
                                processes_by_pid[process.pid] = process
                    except Exception:
                        continue

                process = processes_by_pid.get(self._pid)
                if process is None:
                    continue
                found_indices.append(device_index)
                used_bytes = getattr(process, "usedGpuMemory", None)
                if self._is_valid_memory_value(used_bytes):
                    valid_usage_by_device[device_index] = used_bytes
        except Exception as exc:
            LOGGER.debug("NVML sampling failed: %s", exc)
            return [], {}
        return found_indices, valid_usage_by_device

    def _read_device_delta_mb(self) -> Optional[float]:
        if self._pynvml is None or not self._device_delta_indices:
            return None

        total_delta_bytes = 0
        sampled = False
        for device_index in self._device_delta_indices:
            baseline = self._baseline_used_bytes.get(device_index)
            if baseline is None:
                continue
            try:
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_index)
                current_used = self._pynvml.nvmlDeviceGetMemoryInfo(handle).used
                if not self._is_valid_memory_value(current_used):
                    continue
                total_delta_bytes += max(0, current_used - baseline)
                sampled = True
            except Exception as exc:
                LOGGER.debug(
                    "Device-level VRAM sampling failed for GPU %d: %s",
                    device_index,
                    exc,
                )

        if not sampled:
            return None
        return total_delta_bytes / BYTES_PER_MIB

    def _read_memory_mb(self) -> Optional[float]:
        if self._mode == "device_delta":
            return self._read_device_delta_mb()

        found_indices, valid_usage = self._scan_target_processes()
        if self._mode == "process":
            if not valid_usage:
                return None
            return sum(valid_usage.values()) / BYTES_PER_MIB

        if not found_indices:
            return None

        if len(valid_usage) == len(found_indices):
            self._mode = "process"
            return sum(valid_usage.values()) / BYTES_PER_MIB

        self._mode = "device_delta"
        self._device_delta_indices = sorted(set(found_indices))
        if not self._fallback_warning_emitted:
            LOGGER.warning(
                "Per-process VRAM usage is unavailable for PID %s; falling "
                "back to device-level VRAM delta. Measurements may include "
                "memory changes from other processes sharing the same GPU.",
                self._pid,
            )
            self._fallback_warning_emitted = True
        return self._read_device_delta_mb()

    def stop(self) -> VRAMStats:
        """Stop monitoring, release NVML, and return aggregate statistics."""

        self._stop_event.set()
        if self._thread is not None:
            try:
                self._thread.join(
                    timeout=max(1.0, self.sampling_interval_seconds * 2)
                )
            except Exception as exc:
                LOGGER.debug("Unable to join the VRAM monitoring thread: %s", exc)

        if self._initialized and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception as exc:
                LOGGER.debug("NVML shutdown failed: %s", exc)

        if not self._samples_mb:
            return VRAMStats(None, None, self._mode)
        return VRAMStats(
            max(self._samples_mb), mean(self._samples_mb), self._mode
        )
