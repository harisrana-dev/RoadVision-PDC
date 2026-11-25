"""Tests for benchmark metrics and profiler protocol."""

from __future__ import annotations

import math
import time

import pytest
import torch

from roadvision.benchmarks.metrics import (
    DeviceRunStats,
    InvalidScalingStudy,
    mean_std,
    parallel_efficiency,
    speedup,
    throughput,
)
from roadvision.benchmarks.profiler import (
    benchmark_callable,
    hardware_report,
    save_json,
    synchronize,
)

# ---------------------------------------------------------------------------
# Pure metric math
# ---------------------------------------------------------------------------

def test_speedup_basic():
    assert speedup(10.0, 2.0) == pytest.approx(5.0)
    assert math.isnan(speedup(0.0, 2.0))
    assert math.isnan(speedup(-1.0, 2.0))
    assert math.isnan(speedup(10.0, 0.0))


def test_throughput_basic():
    assert throughput(100, 4.0) == pytest.approx(25.0)
    assert math.isnan(throughput(100, 0.0))
    assert throughput(0, 4.0) == 0.0


def test_mean_std():
    m, s = mean_std([2.0, 4.0, 6.0, 8.0])
    assert m == pytest.approx(5.0)
    assert s == pytest.approx(math.sqrt(20 / 3))  # sample std
    m1, s1 = mean_std([5.0])
    assert m1 == 5.0 and s1 == 0.0
    m0, s0 = mean_std([])
    assert math.isnan(m0) and math.isnan(s0)


def test_device_run_stats():
    stats = DeviceRunStats(
        device="cpu",
        times_s=[2.0, 4.0, 6.0],
        images_per_run=[8, 8, 8],
        cpu_rss_mb=[500.0, 550.0, 520.0],
        gpu_alloc_mb=[],
    )
    assert stats.mean_time == pytest.approx(4.0)
    assert stats.std_time == pytest.approx(2.0)
    # throughput per run: 8/2, 8/4, 8/6 -> mean = (4 + 2 + 1.333...) / 3
    assert stats.mean_throughput == pytest.approx((4.0 + 2.0 + 8.0 / 6.0) / 3.0)
    assert stats.peak_cpu_rss_mb == 550.0
    assert stats.peak_gpu_alloc_mb is None
    d = stats.to_dict()
    assert d["device"] == "cpu"
    assert d["peak_gpu_alloc_mb"] is None


def test_device_run_stats_gpu_memory():
    stats = DeviceRunStats(device="cuda", times_s=[1.0], images_per_run=[8],
                            gpu_alloc_mb=[100.0, 250.0, 180.0])
    assert stats.peak_gpu_alloc_mb == 250.0
    assert stats.std_time == 0.0  # single sample


# ---------------------------------------------------------------------------
# Parallel efficiency is gated behind a valid scaling study
# ---------------------------------------------------------------------------

def test_parallel_efficiency_rejects_speedup_over_cores():
    with pytest.raises(InvalidScalingStudy, match="strong-scaling"):
        parallel_efficiency(speedup_value=4.0, core_counts=[4])


def test_parallel_efficiency_rejects_missing_baseline():
    with pytest.raises(InvalidScalingStudy):
        parallel_efficiency(strong_scaling_times={2: 1.0, 4: 0.6})


def test_parallel_efficiency_valid_strong_scaling():
    # T(1)=8s, T(4)=2.5s -> E = 8/(4*2.5) = 0.8
    eff = parallel_efficiency(strong_scaling_times={1: 8.0, 4: 2.5})
    assert eff == pytest.approx(0.8)
    # Conservative (min) across provided points:
    # T(1)=8, T(2)=3 -> E=8/(2*3)=1.33; T(4)=2.5 -> 0.8 => min 0.8
    eff2 = parallel_efficiency(strong_scaling_times={1: 8.0, 2: 3.0, 4: 2.5})
    assert eff2 == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Profiler protocol
# ---------------------------------------------------------------------------

def test_benchmark_callable_excludes_warmup_and_synchronizes():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1

    res = benchmark_callable(
        fn, device=torch.device("cpu"), name="count", warmup=3, repeats=5
    )
    assert counter["n"] == 8  # 3 warmup + 5 timed
    assert len(res.samples) == 5
    assert res.warmup == 3
    assert res.mean >= 0 and res.std >= 0
    d = res.to_dict()
    assert d["n"] == 5


def test_benchmark_callable_measures_positive_time():
    def sleep_10ms():
        time.sleep(0.01)

    res = benchmark_callable(
        sleep_10ms, device=torch.device("cpu"), name="sleep", warmup=1, repeats=3
    )
    assert all(s >= 0.009 for s in res.samples), res.samples


def test_synchronize_cpu_noop():
    synchronize(torch.device("cpu"))  # must not raise


def test_hardware_report_contains_required_fields():
    info = hardware_report(device_used="cpu")
    required = {
        "os",
        "python_version",
        "cpu_model",
        "ram_total_mb",
        "torch_version",
        "cuda_available",
        "device_used",
    }
    assert required <= set(info)
    assert info["device_used"] == "cpu"
    # Never fabricate: torch version must match the installed one.
    assert info["torch_version"] == torch.__version__


def test_save_json_roundtrip(tmp_path):
    path = save_json({"a": 1, "b": [1.5, 2.5]}, tmp_path / "x" / "y.json")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"a": 1, "b": [1.5, 2.5]}
