"""Optional process-level NVIDIA VRAM monitoring for the training subprocess."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)
BYTES_PER_MIB = 1024 * 1024


@dataclass
class VRAMStats:
    """Summary of sampled process VRAM usage in MiB."""

    peak_vram_mb: Optional[float]
    average_vram_mb: Optional[float]


class GPUVRAMMonitor:
    """Sample VRAM used by one process on a background thread.

    NVML is loaded only when ``start`` is called. Importing this module is safe
    on development machines without an NVIDIA GPU or ``nvidia-ml-py``.
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

    def start(self, pid: int) -> None:
        """Start monitoring *pid*; warn and remain disabled if NVML is absent."""

        self._pid = pid
        try:
            import pynvml
        except ImportError:
            LOGGER.warning(
                "nvidia-ml-py is unavailable; VRAM metrics will be empty."
            )
            return

        try:
            pynvml.nvmlInit()
        except Exception as exc:  # NVML raises library-specific exceptions.
            LOGGER.warning("NVML could not be initialized: %s", exc)
            return

        self._pynvml = pynvml
        self._initialized = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="training-vram-monitor",
            daemon=True,
        )
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._read_process_memory_mb()
            if sample is not None:
                self._samples_mb.append(sample)
            self._stop_event.wait(self.sampling_interval_seconds)

    def _read_process_memory_mb(self) -> Optional[float]:
        if self._pynvml is None or self._pid is None:
            return None

        pynvml = self._pynvml
        total_bytes = 0
        found = False
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
                            processes_by_pid[process.pid] = process
                    except Exception:
                        continue

                process = processes_by_pid.get(self._pid)
                if process is None:
                    continue
                used_bytes = getattr(process, "usedGpuMemory", None)
                if isinstance(used_bytes, int) and used_bytes >= 0:
                    total_bytes += used_bytes
                    found = True
        except Exception as exc:
            LOGGER.debug("NVML sampling failed: %s", exc)
            return None

        if not found:
            return None
        return total_bytes / BYTES_PER_MIB

    def stop(self) -> VRAMStats:
        """Stop monitoring, release NVML, and return aggregate statistics."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sampling_interval_seconds * 2))

        if self._initialized and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception as exc:
                LOGGER.debug("NVML shutdown failed: %s", exc)

        if not self._samples_mb:
            return VRAMStats(None, None)
        return VRAMStats(max(self._samples_mb), mean(self._samples_mb))
