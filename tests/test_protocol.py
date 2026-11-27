"""Tests for the controlled-run state protocol (EXPERIMENT_DESIGN §1.1/§4)."""

from __future__ import annotations

import copy
import random

import pytest
import torch
from torch import nn

from roadvision.benchmarks.protocol import (
    RunState,
    restore_run_state,
    snapshot_run_state,
    state_dict_hash,
    state_dict_hash_from,
)
from roadvision.config import config_hash, load_config


def _tiny_model(seed: int = 0) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
    return model, opt, sched


def _one_step(model: nn.Module, opt: torch.optim.Optimizer) -> None:
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    opt.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()


# ---------------------------------------------------------------------------
# Weight hash
# ---------------------------------------------------------------------------

def test_state_dict_hash_stable_across_identical_models():
    m1, _, _ = _tiny_model(seed=0)
    m2, _, _ = _tiny_model(seed=0)
    assert state_dict_hash(m1) == state_dict_hash(m2)


def test_state_dict_hash_changes_when_weights_change():
    m, _, _ = _tiny_model(seed=0)
    before = state_dict_hash(m)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.001)
    assert state_dict_hash(m) != before


def test_state_dict_hash_differs_across_seeds():
    assert state_dict_hash(_tiny_model(seed=0)[0]) != state_dict_hash(_tiny_model(seed=1)[0])


def test_state_dict_hash_from_is_plain_dict_hash():
    m, _, _ = _tiny_model(seed=0)
    h1 = state_dict_hash(m)
    h2 = state_dict_hash_from(m.state_dict())
    assert h1 == h2


# ---------------------------------------------------------------------------
# Snapshot / restore — model, optimizer, scheduler
# ---------------------------------------------------------------------------

def test_snapshot_tensors_live_on_cpu():
    model, opt, sched = _tiny_model()
    snap = snapshot_run_state(model, opt, sched)
    assert all(t.device.type == "cpu" for t in snap.model_state.values())
    assert snap.weight_hash == state_dict_hash(model)


def test_restore_model_weights_bitwise_equal():
    model, opt, sched = _tiny_model()
    snap = snapshot_run_state(model, opt, sched)
    expected = copy.deepcopy(snap.model_state)
    for _ in range(3):
        _one_step(model, opt)
        sched.step()
    # sanity: training actually changed something
    assert any(
        not torch.equal(expected[k], model.state_dict()[k]) for k in expected
    )
    restore_run_state(snap, model, opt, sched)
    for k, v in expected.items():
        assert torch.equal(v, model.state_dict()[k]), k
    assert state_dict_hash(model) == snap.weight_hash


def test_restore_optimizer_and_scheduler_state():
    model, opt, sched = _tiny_model()
    for _ in range(3):  # populate momentum buffers + scheduler counters
        _one_step(model, opt)
        sched.step()
    snap = snapshot_run_state(model, opt, sched)
    momentum_before = opt.state[next(iter(opt.state))]["momentum_buffer"].clone()
    last_epoch_before = sched.state_dict()["last_epoch"]

    for _ in range(2):  # drift further
        _one_step(model, opt)
        sched.step()
    assert sched.state_dict()["last_epoch"] != last_epoch_before

    restore_run_state(snap, model, opt, sched)
    assert torch.equal(
        opt.state[next(iter(opt.state))]["momentum_buffer"], momentum_before
    )
    assert sched.state_dict()["last_epoch"] == last_epoch_before


def test_restore_requires_optimizer_when_snapshot_has_one():
    model, opt, sched = _tiny_model()
    snap = snapshot_run_state(model, opt, sched)
    with pytest.raises(ValueError, match="optimizer"):
        restore_run_state(snap, model, None, None)


def test_restore_rejects_populated_optimizer_missing_from_snapshot():
    model, opt, _ = _tiny_model()
    snap = snapshot_run_state(model)  # no optimizer captured
    _one_step(model, opt)  # now optimizer HAS state
    with pytest.raises(ValueError, match="snapshot has none"):
        restore_run_state(snap, model, opt, None)


def test_restore_requires_scheduler_when_snapshot_has_one():
    model, opt, sched = _tiny_model()
    snap = snapshot_run_state(model, opt, sched)
    with pytest.raises(ValueError, match="scheduler"):
        restore_run_state(snap, model, opt, None)


# ---------------------------------------------------------------------------
# RNG state (data-order / augmentation control)
# ---------------------------------------------------------------------------

def test_restore_python_and_torch_rng():
    model, _, _ = _tiny_model()
    snap = snapshot_run_state(model)
    first = (random.random(), torch.rand(4).clone())
    restore_run_state(snap, model)
    second = (random.random(), torch.rand(4).clone())
    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_restore_numpy_rng():
    np = pytest.importorskip("numpy")
    model, _, _ = _tiny_model()
    snap = snapshot_run_state(model)
    first = np.random.rand(3)
    restore_run_state(snap, model)
    second = np.random.rand(3)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# State-restore warm-up workflow (EXPERIMENT_DESIGN §4)
# ---------------------------------------------------------------------------

def test_state_restore_warmup_workflow():
    """Warm-up must not leak weight drift into the measured run."""
    model, opt, sched = _tiny_model()
    snap = snapshot_run_state(model, opt, sched)
    initial = copy.deepcopy(snap.model_state)

    # Repeat 1: restore -> warm-up (weights change) -> restore -> measure
    restore_run_state(snap, model, opt, sched)
    for _ in range(3):  # warm-up batches, EXCLUDED
        _one_step(model, opt)
    assert any(not torch.equal(initial[k], model.state_dict()[k]) for k in initial)
    restore_run_state(snap, model, opt, sched)  # measured run starts clean
    measured_start = copy.deepcopy(model.state_dict())
    _one_step(model, opt)  # measured unit
    assert any(
        not torch.equal(measured_start[k], model.state_dict()[k]) for k in initial
    )

    # Repeat 2: same starting weights as repeat 1's measured run
    restore_run_state(snap, model, opt, sched)
    for k in initial:
        assert torch.equal(model.state_dict()[k], initial[k])


def test_snapshot_is_immutable_against_later_training():
    model, opt, _ = _tiny_model()
    snap = snapshot_run_state(model, opt)
    frozen = copy.deepcopy(snap.model_state)
    for _ in range(2):
        _one_step(model, opt)
    for k, v in snap.model_state.items():
        assert torch.equal(v, frozen[k]), k


def test_runstate_defaults_without_scheduler():
    model, opt, _ = _tiny_model()
    snap = snapshot_run_state(model, opt)
    assert isinstance(snap, RunState)
    assert snap.scheduler_state is None
    assert snap.optimizer_state is not None


# ---------------------------------------------------------------------------
# Config hash (manifest pair with the weight hash)
# ---------------------------------------------------------------------------

def test_config_hash_stable_and_sensitive():
    cfg = load_config()  # shipped configs/experiment.yaml
    h = config_hash(cfg)
    assert config_hash(load_config()) == h
    mutated = cfg.to_dict()
    mutated["seed"] = int(mutated["seed"]) + 1
    from roadvision.config import Config

    assert config_hash(Config(mutated)) != h


# ---------------------------------------------------------------------------
# Integration with the actual candidate detector
# ---------------------------------------------------------------------------

def test_detector_snapshot_restore_roundtrip():
    from roadvision.models.detector import build_detector

    model_config = {
        "seed": 42,
        "model": {
            "name": "ssdlite320_mobilenet_v3_large",
            "num_classes": 5,
            "weights": None,
            "weights_backbone": "NONE",  # no download in tests
            "trainable_backbone_layers": 6,
        },
        "training": {
            "optimizer": "sgd",
            "lr": 0.005,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "lr_scheduler": "steplr",
            "lr_step_size": 8,
            "lr_gamma": 0.1,
            "epochs": 1,
            "batch_size": 2,
            "grad_clip_norm": None,
        },
        "dataloader": {"num_workers": 0, "pin_memory": False},
        "preprocessing": {"resize": [320, 320], "hflip_prob": 0.0},
    }
    model = build_detector(model_config)
    opt = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    snap = snapshot_run_state(model, opt)
    assert len(snap.weight_hash) == 64
    restore_run_state(snap, model, opt)
    assert state_dict_hash(model) == snap.weight_hash
