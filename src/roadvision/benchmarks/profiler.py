"""Hardware inventory and benchmark run protocol.

All hardware numbers are *collected at runtime* — never hard-coded or
guessed (project honesty rule #2).
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Runtime-collected hardware/software inventory."""

    hostname: str
    os: str
    python_version: str
    cpu_model: str
    cpu_count_physical: int | None
    cpu_count_logical: int | None
    ram_total_mb: float
    torch_version: str
    torchvision_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_total_memory_mb: float | None
    device_used: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        # Linux
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_total_mb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1 << 20)
    except Exception:
        return float("nan")


def _tv_version() -> str | None:
    try:
        import torchvision

        return torchvision.__version__
    except Exception:
        return None


def collect_hardware_info(device_used: str = "cpu") -> HardwareInfo:
    """Collect the inventory that must accompany every benchmark result."""
    cuda = torch.cuda.is_available()
    gpu_name = None
    gpu_mem = None
    if cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1 << 20)

    return HardwareInfo(
        hostname=platform.node(),
        os=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=sys.version.split()[0],
        cpu_model=_cpu_model(),
        cpu_count_physical=None,  # filled below if psutil available
        cpu_count_logical=None,
        ram_total_mb=_ram_total_mb(),
        torch_version=torch.__version__,
        torchvision_version=_tv_version(),
        cuda_available=cuda,
        cuda_version=torch.version.cuda if cuda else None,
        gpu_name=gpu_name,
        gpu_total_memory_mb=gpu_mem,
        device_used=device_used,
    )


def _fill_cpu_counts(info: HardwareInfo) -> HardwareInfo:
    try:
        import psutil

        info.cpu_count_physical = psutil.cpu_count(logical=False)
        info.cpu_count_logical = psutil.cpu_count(logical=True)
    except Exception:
        info.cpu_count_logical = None
    return info


def hardware_report(device_used: str = "cpu") -> dict[str, Any]:
    """Convenience: filled HardwareInfo as a plain dict."""
    return _fill_cpu_counts(collect_hardware_info(device_used)).to_dict()


# ---------------------------------------------------------------------------
# Repeated benchmark protocol
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Statistics over repeated measured runs (warm-up excluded)."""

    name: str
    device: str
    samples: list[float] = field(default_factory=list)
    warmup: int = 0
    unit: str = "s"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else float("nan")

    @property
    def std(self) -> float:
        n = len(self.samples)
        if n < 2:
            return 0.0 if n == 1 else float("nan")
        m = self.mean
        var = sum((x - m) ** 2 for x in self.samples) / (n - 1)
        return var ** 0.5

    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else float("nan")

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "unit": self.unit,
            "n": len(self.samples),
            "warmup": self.warmup,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "samples": list(self.samples),
            "extra": self.extra,
        }


def synchronize(device: torch.device) -> None:
    """CUDA sync no-op on CPU (the one place GPU timing correctness lives)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    name: str,
    warmup: int = 3,
    repeats: int = 5,
    unit: str = "s",
) -> BenchmarkResult:
    """Run ``fn`` ``warmup`` times (discarded), then ``repeats`` timed runs.

    CUDA is synchronized immediately before/after every timed call so
    asynchronous kernel launches cannot leak into the next sample.
    """
    for _ in range(warmup):
        fn()
        synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        t1 = time.perf_counter()
        samples.append(t1 - t0)

    return BenchmarkResult(
        name=name, device=str(device), samples=samples, warmup=warmup, unit=unit
    )


def save_json(obj: Any, path: str | Path) -> Path:
    """Write a JSON artifact (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
    return p
