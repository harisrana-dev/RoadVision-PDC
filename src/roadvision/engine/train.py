"""Shared training engine — ONE implementation for CPU and GPU.

The device is a *parameter*, never a fork in logic. Experiment D (Colab CPU
vs Colab GPU in the same runtime) therefore exercises identical code.

Timing model (see docs/EXPERIMENT_DESIGN.md §4)
-----------------------------------------------
Per iteration the engine measures, on the critical path:

    data_loading : time to pull the next batch from the DataLoader
    transfer     : host->device copy (+ sync)          [0 on CPU]
    compute      : forward + loss + backward + step (+ sync on CUDA)
    end_to_end   : wall clock of the whole measured epoch (contains the above)

Buckets are *not* summed to reconstruct end_to_end — pipelined loaders make
data_loading for batch i+1 overlap compute for batch i; the engine reports
each bucket and an explicit ``other`` residual.

CUDA rules: ``torch.cuda.synchronize()`` before every GPU timestamp;
warm-up iterations are excluded from all reported statistics.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@contextmanager
def cuda_sync_if(device: torch.device) -> Iterator[None]:
    """Synchronize CUDA before/after the block when running on GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class IterationTiming:
    data_loading: float = 0.0
    transfer: float = 0.0
    compute: float = 0.0

    @property
    def accounted(self) -> float:
        return self.data_loading + self.transfer + self.compute


@dataclass
class EpochReport:
    """Aggregated measurements for one measured epoch (warm-up excluded)."""

    device: str
    images: int = 0
    end_to_end: float = 0.0
    data_loading: float = 0.0
    transfer: float = 0.0
    compute: float = 0.0
    other: float = 0.0          # end_to_end - accounted (>= 0 by construction)
    loss_mean: float = float("nan")
    loss_std: float = float("nan")
    throughput_img_s: float = float("nan")
    peak_cpu_rss_mb: float = float("nan")
    peak_gpu_alloc_mb: float | None = None    # None on CPU runs
    peak_gpu_reserved_mb: float | None = None
    loss_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["loss_history"] = list(self.loss_history)
        return d


def _peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (None-safe on all OS)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1 << 20)
    except Exception:  # pragma: no cover - psutil optional at runtime
        return float("nan")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _ensure_model_device(model: nn.Module, device: torch.device) -> None:
    """Fail fast with an actionable error when model and device disagree.

    The engine never moves the model implicitly (``run_training`` does that
    once, explicitly); a mismatch deep inside torchvision would otherwise
    surface as an opaque device RuntimeError.
    """
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device=cuda requested but CUDA is not available on this host. "
            "Use device=cpu here, or run on a CUDA-enabled machine (Colab GPU)."
        )
    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        return  # no parameters — nothing to place
    mismatch = param_device.type != device.type or (
        device.index is not None and param_device.index != device.index
    )
    if mismatch:
        raise RuntimeError(
            f"Model parameters are on '{param_device}' but the training device "
            f"is '{device}'. Call model.to(device) first "
            "(run_training does this automatically)."
        )


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    *,
    epoch: int = 0,
    grad_clip_norm: float | None = None,
    max_batches: int | None = None,
    collect_timing: bool = True,
) -> EpochReport:
    """Run one training epoch on ``device`` and return measured timings.

    Parameters
    ----------
    max_batches:
        Optional cap for pilot/smoke runs (still measured honestly).
    collect_timing:
        When False, only loss is tracked (cheaper; used for pure loss curves).
    """
    _ensure_model_device(model, device)
    model.train()
    report = EpochReport(device=str(device))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- optional RSS sampling thread-free approach: sample per batch ----
    losses: list[float] = []

    # Warm timing buckets
    t_data = t_xfer = t_compute = 0.0
    t_epoch_start = time.perf_counter()

    iterator = iter(data_loader)
    batches_done = 0
    while True:
        if max_batches is not None and batches_done >= max_batches:
            break

        # -- data_loading (critical path: we block on the iterator here) --
        t0 = time.perf_counter() if collect_timing else 0.0
        try:
            images, targets = next(iterator)
        except StopIteration:
            break
        t1 = time.perf_counter() if collect_timing else 0.0
        data_dt = (t1 - t0) if collect_timing else 0.0

        # -- transfer (host -> device) --
        t2 = time.perf_counter() if collect_timing else 0.0
        if device.type == "cuda":
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [
                {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in t.items()}
                for t in targets
            ]
            _sync(device)
        t3 = time.perf_counter() if collect_timing else 0.0
        xfer_dt = (t3 - t2) if collect_timing and device.type == "cuda" else 0.0

        # -- compute: forward + loss + backward + optimizer step --
        t4 = time.perf_counter() if collect_timing else 0.0
        _sync(device)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch} batch {batches_done}: {loss_dict}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        _sync(device)
        t5 = time.perf_counter() if collect_timing else 0.0
        compute_dt = (t5 - t4) if collect_timing else 0.0

        losses.append(float(loss.detach()))
        t_data += data_dt
        t_xfer += xfer_dt
        t_compute += compute_dt
        report.images += len(images)
        batches_done += 1

    report.end_to_end = time.perf_counter() - t_epoch_start
    report.data_loading = t_data
    report.transfer = t_xfer
    report.compute = t_compute
    report.other = max(0.0, report.end_to_end - (t_data + t_xfer + t_compute))
    if losses:
        n = len(losses)
        mean = sum(losses) / n
        var = sum((x - mean) ** 2 for x in losses) / (n - 1) if n > 1 else 0.0
        report.loss_mean = mean
        report.loss_std = var ** 0.5
        report.loss_history = losses
    if report.end_to_end > 0:
        report.throughput_img_s = report.images / report.end_to_end
    report.peak_cpu_rss_mb = _peak_rss_mb()
    if device.type == "cuda":
        report.peak_gpu_alloc_mb = torch.cuda.max_memory_allocated(device) / (1 << 20)
        report.peak_gpu_reserved_mb = torch.cuda.max_memory_reserved(device) / (1 << 20)
    else:
        report.peak_gpu_alloc_mb = None
        report.peak_gpu_reserved_mb = None
    return report


# ---------------------------------------------------------------------------
# Evaluation timing (validation pass)
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    device: str
    images: int = 0
    end_to_end: float = 0.0
    data_loading: float = 0.0
    transfer: float = 0.0
    inference: float = 0.0
    throughput_img_s: float = float("nan")
    peak_gpu_alloc_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@torch.no_grad()
def evaluate_timed(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> EvalReport:
    """Timed validation/inference pass (no gradients, same bucket rules)."""
    _ensure_model_device(model, device)
    model.eval()
    report = EvalReport(device=str(device))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_data = t_xfer = t_infer = 0.0
    t_start = time.perf_counter()

    iterator = iter(data_loader)
    batches_done = 0
    while True:
        if max_batches is not None and batches_done >= max_batches:
            break
        t0 = time.perf_counter()
        try:
            images, _targets = next(iterator)
        except StopIteration:
            break
        t1 = time.perf_counter()
        t_data += t1 - t0

        t2 = time.perf_counter()
        if device.type == "cuda":
            images = [img.to(device, non_blocking=True) for img in images]
            _sync(device)
        t3 = time.perf_counter()
        t_xfer += (t3 - t2) if device.type == "cuda" else 0.0

        t4 = time.perf_counter()
        _sync(device)
        model(images)          # inference path: eval mode, no targets
        _sync(device)
        t5 = time.perf_counter()
        t_infer += t5 - t4
        report.images += len(images)
        batches_done += 1

    report.end_to_end = time.perf_counter() - t_start
    report.data_loading = t_data
    report.transfer = t_xfer
    report.inference = t_infer
    if report.end_to_end > 0:
        report.throughput_img_s = report.images / report.end_to_end
    if device.type == "cuda":
        report.peak_gpu_alloc_mb = torch.cuda.max_memory_allocated(device) / (1 << 20)
    return report


# ---------------------------------------------------------------------------
# Full run orchestration (used by scripts / notebooks)
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Construct the optimizer exactly as specified in the config."""
    train_cfg = config["training"]
    name = str(train_cfg.get("optimizer", "sgd")).lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=wd,
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    """StepLR (or none) per config."""
    train_cfg = config["training"]
    name = str(train_cfg.get("lr_scheduler", "none")).lower()
    if name in {"none", "null", ""}:
        return None
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("lr_step_size", 8)),
            gamma=float(train_cfg.get("lr_gamma", 0.1)),
        )
    raise ValueError(f"Unknown lr_scheduler {name!r}")


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    *,
    val_loader: DataLoader | None = None,
    epochs: int | None = None,
    max_batches_per_epoch: int | None = None,
    on_epoch_end=None,
) -> dict[str, Any]:
    """Train for N epochs on ``device`` — the single shared entry point.

    CPU and GPU experiments both call this function; only ``device``
    differs. Returns a history dict (epoch reports + validation reports).
    """
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device=cuda requested but CUDA is not available on this host. "
            "Use device=cpu here, or run on a CUDA-enabled machine (Colab GPU)."
        )
    model.to(device)  # place first; optimizer/scheduler then reference on-device params
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    n_epochs = int(epochs if epochs is not None else config["training"]["epochs"])
    grad_clip = config["training"].get("grad_clip_norm")

    history: dict[str, Any] = {"device": str(device), "epochs": []}

    for epoch in range(n_epochs):
        train_report = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch=epoch,
            grad_clip_norm=grad_clip,
            max_batches=max_batches_per_epoch,
        )
        entry: dict[str, Any] = {"epoch": epoch, "train": train_report.to_dict()}

        if val_loader is not None:
            eval_report = evaluate_timed(
                model, val_loader, device, max_batches=max_batches_per_epoch
            )
            entry["val"] = eval_report.to_dict()

        if scheduler is not None:
            scheduler.step()
            entry["lr"] = optimizer.param_groups[0]["lr"]

        history["epochs"].append(entry)
        logger.info(
            "epoch %d device=%s loss=%.4f±%.4f e2e=%.2fs thr=%.2f img/s",
            epoch,
            device,
            train_report.loss_mean,
            train_report.loss_std,
            train_report.end_to_end,
            train_report.throughput_img_s,
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, entry)

    return history


# ---------------------------------------------------------------------------
# Checkpointing (full training resume; controlled-state support per
# docs/EXPERIMENT_DESIGN.md §1.1/§4)
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    epoch: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a full training checkpoint (model + optimizer + scheduler).

    ``optimizer``/``scheduler`` are stored whenever given so a later
    :func:`load_checkpoint` can resume exactly — momentum buffers, LR
    counters and the epoch number are never dropped on save.
    """
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "extra": dict(extra or {}),
    }
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint written by :func:`save_checkpoint`.

    When the caller passes an optimizer/scheduler, the checkpoint MUST
    contain that state: a full training resume never silently omits
    optimizer momentum, scheduler counters or the epoch number. A
    weights-only restore is explicit — call with ``optimizer=None`` and
    ``scheduler=None``.

    Returns ``{"epoch": int, "extra": dict}`` from the payload.
    """
    # Our own local checkpoint format (tensors + plain containers) —
    # weights_only=False is required for the nested optimizer/scheduler
    # dicts on torch>=2.6; only ever load files written by save_checkpoint.
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)

    ckpt_opt = payload.get("optimizer")
    if optimizer is not None:
        if ckpt_opt is None:
            raise ValueError(
                f"Checkpoint '{path}' has no optimizer state; a full resume must "
                "restore it. Re-save via save_checkpoint(..., optimizer=...) or "
                "load with optimizer=None for an explicit weights-only restore."
            )
        optimizer.load_state_dict(ckpt_opt)

    ckpt_sched = payload.get("scheduler")
    if scheduler is not None:
        if ckpt_sched is None:
            raise ValueError(
                f"Checkpoint '{path}' has no scheduler state; a full resume must "
                "restore it. Re-save via save_checkpoint(..., scheduler=...) or "
                "load with scheduler=None."
            )
        scheduler.load_state_dict(ckpt_sched)

    return {"epoch": int(payload.get("epoch", 0)), "extra": payload.get("extra", {})}
