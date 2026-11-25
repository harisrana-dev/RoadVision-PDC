"""Plotting helpers.

Every function only renders the arrays it is given — there are no default
"example" datasets and no synthetic curves. If the input is empty the
function raises, so an empty experiment cannot silently produce a
plausible-looking figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe (scripts + CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _require_nonempty(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} is empty — refusing to plot (would fabricate a figure).")
    return arr


def plot_loss_curve(
    losses: Sequence[float],
    *,
    title: str = "Training loss",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot per-batch or per-epoch loss values as given."""
    arr = _require_nonempty(losses, "losses")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(arr.size), arr, marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _finish(fig, save_path, show)
    return out


def plot_device_comparison(
    labels: Sequence[str],
    cpu_values: Sequence[float],
    gpu_values: Sequence[float],
    *,
    ylabel: str,
    title: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Grouped bar chart: one pair of bars per label (e.g. batch sizes).

    NaN entries are drawn as missing bars with a marker — never replaced by
    zeros (a failed/OOM configuration must stay visibly failed).
    """
    labs = [str(x) for x in labels]
    cpu = _require_nonempty(cpu_values, "cpu_values")
    gpu = _require_nonempty(gpu_values, "gpu_values")
    if len(labs) != len(cpu) or len(labs) != len(gpu):
        raise ValueError("labels, cpu_values, gpu_values must have equal length")

    x = np.arange(len(labs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, cpu, width, label="CPU", color="#4C72B0")
    ax.bar(x + width / 2, gpu, width, label="GPU", color="#55A868")
    for i, (c, g) in enumerate(zip(cpu, gpu)):
        if np.isnan(c):
            ax.text(i - width / 2, 0.02, "FAIL", ha="center", va="bottom", color="red", fontsize=8)
        if np.isnan(g):
            ax.text(i + width / 2, 0.02, "FAIL", ha="center", va="bottom", color="red", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _finish(fig, save_path, show)


def plot_timing_breakdown(
    categories: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "Timing breakdown",
    ylabel: str = "seconds",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Horizontal bar chart of timing buckets (data/transfer/compute/other)."""
    cats = [str(c) for c in categories]
    vals = _require_nonempty(values, "values")
    if len(cats) != len(vals):
        raise ValueError("categories and values must have equal length")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(cats[::-1], vals[::-1], color="#8172B2")
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _finish(fig, save_path, show)


def _finish(fig, save_path, show) -> Path | None:
    out: Path | None = None
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    if show:  # pragma: no cover - interactive only
        plt.show()
    else:
        plt.close(fig)
    return out
