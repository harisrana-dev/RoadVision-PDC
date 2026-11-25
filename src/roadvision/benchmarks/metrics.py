"""Benchmark metric aggregation: speedup, throughput, memory.

What is and is not computed (docs/EXPERIMENT_DESIGN.md §3)
----------------------------------------------------------
* speedup      = CPU elapsed / GPU elapsed   (same workload, same config)
* throughput   = images processed / elapsed second
* mean / std   over repeated samples (n >= 2 for a meaningful std)
* CPU peak RSS and GPU peak allocated are kept as *separate* columns —
  they measure different things and are never subtracted from each other.
* **Parallel efficiency is intentionally NOT computed.** Dividing a
  two-device speedup by a CPU core count is not a valid scaling metric.
  A real efficiency figure requires a documented strong/weak scaling
  experiment (varying thread counts on a fixed problem), which this
  project has not run. The function ``parallel_efficiency`` raises unless
  given an explicit, valid scaling study result.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeviceRunStats:
    """Aggregated stats for one device on one configuration."""

    device: str
    times_s: list[float] = field(default_factory=list)
    images_per_run: list[int] = field(default_factory=list)
    cpu_rss_mb: list[float] = field(default_factory=list)
    gpu_alloc_mb: list[float] = field(default_factory=list)

    @property
    def mean_time(self) -> float:
        return statistics.fmean(self.times_s) if self.times_s else math.nan

    @property
    def std_time(self) -> float:
        if len(self.times_s) < 2:
            return 0.0 if self.times_s else math.nan
        return statistics.stdev(self.times_s)

    @property
    def mean_throughput(self) -> float:
        vals = [
            n / t for n, t in zip(self.images_per_run, self.times_s) if t > 0
        ]
        return statistics.fmean(vals) if vals else math.nan

    @property
    def std_throughput(self) -> float:
        vals = [
            n / t for n, t in zip(self.images_per_run, self.times_s) if t > 0
        ]
        if len(vals) < 2:
            return 0.0 if vals else math.nan
        return statistics.stdev(vals)

    @property
    def peak_cpu_rss_mb(self) -> float:
        return max(self.cpu_rss_mb) if self.cpu_rss_mb else math.nan

    @property
    def peak_gpu_alloc_mb(self) -> float | None:
        return max(self.gpu_alloc_mb) if self.gpu_alloc_mb else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "n_runs": len(self.times_s),
            "mean_time_s": self.mean_time,
            "std_time_s": self.std_time,
            "mean_throughput_img_s": self.mean_throughput,
            "std_throughput_img_s": self.std_throughput,
            "peak_cpu_rss_mb": self.peak_cpu_rss_mb,
            "peak_gpu_alloc_mb": self.peak_gpu_alloc_mb,
        }


def speedup(cpu_time_s: float, gpu_time_s: float) -> float:
    """CPU elapsed / GPU elapsed for an identical workload.

    Returns NaN for non-positive inputs rather than raising, so a partially
    failed run cannot masquerade as a valid ratio.
    """
    if not (cpu_time_s > 0 and gpu_time_s > 0):
        return math.nan
    return cpu_time_s / gpu_time_s


def throughput(images: int, elapsed_s: float) -> float:
    """Images processed per second."""
    if not (elapsed_s > 0):
        return math.nan
    return images / elapsed_s


def mean_std(values: list[float]) -> tuple[float, float]:
    """Arithmetic mean and sample standard deviation (n-1).

    n < 2 -> std 0.0 (a single sample has no dispersion to estimate);
    empty -> (nan, nan).
    """
    if not values:
        return math.nan, math.nan
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, 0.0
    return m, statistics.stdev(values)


class InvalidScalingStudy(ValueError):
    """Raised when parallel efficiency is requested without a valid study."""


def parallel_efficiency(
    *,
    speedup_value: float | None = None,
    core_counts: list[int] | None = None,
    strong_scaling_times: dict[int, float] | None = None,
) -> float:
    """Compute parallel efficiency ONLY from a valid strong-scaling study.

    A mathematically valid strong-scaling efficiency is::

        E(p) = T(1) / (p * T(p))

    where T(1) and T(p) are wall times of the *same problem* run with 1 and
    p threads/cores respectively.

    Passing a raw CPU-vs-GPU speedup and a core count (the common mistake)
    raises :class:`InvalidScalingStudy`. There is no silent fallback.
    """
    if strong_scaling_times is None:
        raise InvalidScalingStudy(
            "Parallel efficiency requires a strong-scaling study: pass "
            "strong_scaling_times={1: T1, p: Tp} measured on the SAME problem "
            "size. CPU-vs-GPU speedup divided by core count is NOT a valid "
            "efficiency metric and will not be computed."
        )
    if 1 not in strong_scaling_times or len(strong_scaling_times) < 2:
        raise InvalidScalingStudy(
            "strong_scaling_times must include the 1-thread baseline and at "
            "least one other thread count."
        )
    t1 = strong_scaling_times[1]
    worst = math.nan
    for p, tp in strong_scaling_times.items():
        if p <= 0 or tp <= 0 or t1 <= 0:
            continue
        eff = t1 / (p * tp)
        if math.isnan(worst) or eff < worst:
            worst = eff  # report the bottleneck efficiency (conservative)
    if math.isnan(worst):
        raise InvalidScalingStudy("No valid (p, T(p)) pairs in the study data.")
    # Note: speedup_value/core_counts are accepted only to fail loudly if
    # someone tries the invalid route first.
    if speedup_value is not None and core_counts is not None:
        raise InvalidScalingStudy(
            "speedup/core_count form is not accepted; use strong_scaling_times."
        )
    return worst
