"""Controlled-run state protocol for Experiment D (docs/EXPERIMENT_DESIGN.md §1.1/§4).

The protocol answers two questions the experiment design must not leave to
guesswork:

1. How does every device run / repeat start from *identical* initial state?
   -> :func:`snapshot_run_state` captures model, optimizer, scheduler and
      RNG state once; :func:`restore_run_state` reloads it before warm-up
      and again before each measured unit (state-restore warm-up).

2. How do we prove two runs used the same initialization?
   -> :func:`state_dict_hash` (initial-weight hash) plus
      :func:`roadvision.config.config_hash` (configuration hash), both
      recorded in the run manifest.

Reproducibility level (honest): the snapshot makes initial weights
byte-identical across runs by construction. Training trajectories are
deterministic on CPU with fixed seeds/data order, but GPU kernels may be
nondeterministic — never claim byte-identical GPU results.
"""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class RunState:
    """Immutable snapshot of everything a controlled run must restart from."""

    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    rng_state: dict[str, Any] = field(default_factory=dict)
    weight_hash: str = ""


def _clone_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a state_dict so later training cannot mutate the snapshot."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def snapshot_run_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> RunState:
    """Capture model/optimizer/scheduler/RNG state at a controlled start point.

    Call BEFORE any warm-up or measured training. Tensors are cloned to CPU
    so the snapshot is device-independent (the same bytes can initialize the
    CPU run and the GPU run of Experiment D).
    """
    rng_state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        rng_state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    rng_state["torch"] = torch.get_rng_state().cpu().clone()
    if torch.cuda.is_available():  # pragma: no cover - laptop has no CUDA
        rng_state["torch_cuda"] = [s.cpu().clone() for s in torch.cuda.get_rng_state_all()]

    model_state = _clone_to_cpu(model.state_dict())
    return RunState(
        model_state=model_state,
        optimizer_state=copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None,
        scheduler_state=copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        rng_state=rng_state,
        weight_hash=state_dict_hash_from(model_state),
    )


def restore_run_state(
    state: RunState,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> None:
    """Restore a snapshot so the next run starts from the exact same state.

    ``load_state_dict`` copies snapshot tensors into the model's existing
    parameters, so this works regardless of the model's current device.
    Optimizer/scheduler restore only when they were snapshotted and are
    provided here — a mismatch raises instead of silently skipping state.
    """
    model.load_state_dict(state.model_state, strict=True)

    if state.optimizer_state is not None:
        if optimizer is None:
            raise ValueError(
                "Snapshot contains optimizer state but no optimizer was given; "
                "a controlled run must restore optimizer state too."
            )
        optimizer.load_state_dict(state.optimizer_state)
    elif optimizer is not None and optimizer.state:
        raise ValueError(
            "Optimizer has state (momentum buffers) but the snapshot has none; "
            "re-take the snapshot from a freshly built optimizer."
        )

    if state.scheduler_state is not None:
        if scheduler is None:
            raise ValueError(
                "Snapshot contains scheduler state but no scheduler was given."
            )
        scheduler.load_state_dict(state.scheduler_state)

    if "python" in state.rng_state:
        random.setstate(state.rng_state["python"])
    if "numpy" in state.rng_state:
        try:
            import numpy as np

            np.random.set_state(state.rng_state["numpy"])
        except ImportError:  # pragma: no cover
            pass
    if "torch" in state.rng_state:
        torch.set_rng_state(state.rng_state["torch"].cpu())
    if "torch_cuda" in state.rng_state and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all([s.cpu() for s in state.rng_state["torch_cuda"]])


def state_dict_hash_from(model_state: dict[str, torch.Tensor]) -> str:
    """SHA-256 over sorted parameter names + dtypes + shapes + raw bytes.

    Operates on a (CPU) state_dict; deterministic for identical weights and
    sensitive to any value change. Used as the Experiment D *initial-weight
    hash* recorded in run manifests.
    """
    h = hashlib.sha256()
    for name in sorted(model_state):
        tensor = model_state[name]
        if not torch.is_tensor(tensor):
            continue
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def state_dict_hash(model: nn.Module) -> str:
    """Initial-weight hash of a model's current state (see above)."""
    return state_dict_hash_from(model.state_dict())
