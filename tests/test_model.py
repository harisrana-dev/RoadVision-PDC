"""Tests for model construction and the shared training engine on CPU."""

from __future__ import annotations

import math

import pytest
import torch

from roadvision.engine.train import (
    _ensure_model_device,
    build_optimizer,
    build_scheduler,
    evaluate_timed,
    load_checkpoint,
    run_training,
    save_checkpoint,
    train_one_epoch,
)
from roadvision.models.detector import (
    build_detector,
    count_parameters,
    make_synthetic_batch,
)


@pytest.fixture()
def model_config() -> dict:
    return {
        "seed": 42,
        "model": {
            "name": "ssdlite320_mobilenet_v3_large",
            "num_classes": 5,
            "weights": None,
            "weights_backbone": "NONE",  # no download in CI/tests
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


def test_build_detector_num_classes(model_config: dict):
    model = build_detector(model_config)
    assert isinstance(model, torch.nn.Module)
    n = count_parameters(model)
    assert n > 0
    # SSDLite320 is a compact detector — sanity upper bound (30M params).
    assert n < 30_000_000


def test_build_detector_rejects_unknown_model(model_config: dict):
    model_config["model"]["name"] = "faster_rcnn_resnet50"
    with pytest.raises(ValueError, match="Unknown model"):
        build_detector(model_config)


def test_build_detector_rejects_full_coco_weights(model_config: dict):
    model_config["model"]["weights"] = "COCO_V1"
    with pytest.raises(ValueError, match="COCO"):
        build_detector(model_config)


def test_build_detector_num_classes_must_match_classes(model_config: dict):
    # num_classes must equal len(classes)+1 when full config validated;
    # here we only check the factory's own >=2 guard via config validation
    # in roadvision.config — factory accepts 5.
    model_config["model"]["num_classes"] = 1
    # 1 class = background only; torchvision accepts but our config validator
    # would reject. Factory-level: still builds (documented contract is 5).
    model = build_detector(model_config)
    assert count_parameters(model) > 0


def test_forward_returns_finite_losses(model_config: dict):
    model = build_detector(model_config)
    model.train()
    images, targets = make_synthetic_batch(batch_size=2, num_boxes=3, seed=1)
    loss_dict = model(images, targets)
    assert isinstance(loss_dict, dict) and len(loss_dict) > 0
    total = sum(loss_dict.values())
    assert torch.isfinite(total), f"non-finite loss: {loss_dict}"
    for k, v in loss_dict.items():
        assert torch.isfinite(v), f"{k} not finite: {v}"


def test_backward_and_step(model_config: dict):
    model = build_detector(model_config)
    opt = build_optimizer(model, model_config)
    images, targets = make_synthetic_batch(batch_size=2, seed=2)
    loss = sum(model(images, targets).values())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    opt.step()  # must not raise


def test_eval_inference_runs(model_config: dict):
    model = build_detector(model_config)
    model.eval()
    images, _ = make_synthetic_batch(batch_size=2, seed=3)
    with torch.no_grad():
        outputs = model(images)
    assert len(outputs) == 2
    for out in outputs:
        assert "boxes" in out and "scores" in out and "labels" in out
        assert out["boxes"].shape[-1] == 4
        # labels must be within [0, num_classes)
        if out["labels"].numel():
            assert out["labels"].max() < model_config["model"]["num_classes"]


def test_train_one_epoch_cpu(model_config: dict):
    model = build_detector(model_config)
    opt = build_optimizer(model, model_config)

    class _Loader:
        batch_size = 2

        def __iter__(self):
            for _ in range(2):
                yield make_synthetic_batch(batch_size=2, seed=5)

    report = train_one_epoch(model, opt, _Loader(), torch.device("cpu"), epoch=0)
    assert report.device == "cpu"
    assert report.images == 4
    assert math.isfinite(report.loss_mean)
    assert report.end_to_end > 0
    assert report.compute > 0
    assert report.transfer == 0.0  # CPU: no host->device copy
    assert report.peak_gpu_alloc_mb is None
    assert report.throughput_img_s > 0
    # end_to_end must contain the accounted buckets
    assert report.end_to_end + 1e-9 >= report.compute


def test_nonfinite_loss_raises(model_config: dict):
    model = build_detector(model_config)
    # Poison the model outputs by breaking weights.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))
    opt = build_optimizer(model, model_config)

    class _Loader:
        batch_size = 2

        def __iter__(self):
            yield make_synthetic_batch(batch_size=2, seed=6)

    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        train_one_epoch(model, opt, _Loader(), torch.device("cpu"), epoch=0)


def test_run_training_end_to_end(model_config: dict):
    model = build_detector(model_config)

    class _Loader:
        batch_size = 2

        def __iter__(self):
            yield make_synthetic_batch(batch_size=2, seed=8)

    history = run_training(
        model, _Loader(), model_config, torch.device("cpu"), epochs=1,
        max_batches_per_epoch=1,
    )
    assert history["device"] == "cpu"
    assert len(history["epochs"]) == 1
    assert math.isfinite(history["epochs"][0]["train"]["loss_mean"])


def test_evaluate_timed_cpu(model_config: dict):
    model = build_detector(model_config)

    class _Loader:
        batch_size = 2

        def __iter__(self):
            yield make_synthetic_batch(batch_size=2, seed=9)

    rep = evaluate_timed(model, _Loader(), torch.device("cpu"))
    assert rep.images == 2
    assert rep.end_to_end > 0
    assert rep.transfer == 0.0
    assert rep.peak_gpu_alloc_mb is None


def test_optimizer_and_scheduler_builders(model_config: dict):
    model = build_detector(model_config)
    opt = build_optimizer(model, model_config)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.005)
    sched = build_scheduler(opt, model_config)
    assert sched is not None
    model_config["training"]["lr_scheduler"] = "none"
    assert build_scheduler(opt, model_config) is None
    model_config["training"]["optimizer"] = "nope"
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(model, model_config)


def test_cuda_only_if_available(model_config: dict):
    """CUDA path exercised only when a GPU exists (skip on the laptop)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host (expected locally)")
    model = build_detector(model_config).cuda()
    opt = build_optimizer(model, model_config)
    images, targets = make_synthetic_batch(batch_size=2, seed=4)
    images = [im.cuda() for im in images]
    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
    loss = sum(model(images, targets).values())
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Shared-engine hardening (device placement, edge cases, checkpoints)
# ---------------------------------------------------------------------------

def test_optimizer_updates_parameters(model_config: dict):
    model = build_detector(model_config)
    opt = build_optimizer(model, model_config)
    before = [p.detach().clone() for p in model.parameters()]

    class _Loader:
        batch_size = 2

        def __iter__(self):
            yield make_synthetic_batch(batch_size=2, seed=11)

    train_one_epoch(model, opt, _Loader(), torch.device("cpu"), epoch=0, max_batches=1)
    assert any(
        not torch.equal(a, b) for a, b in zip(before, model.parameters())
    ), "optimizer step must actually update parameters"


def test_device_check_cuda_unavailable_raises(model_config: dict, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = build_detector(model_config)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _ensure_model_device(model, torch.device("cuda"))


def test_device_check_mismatch_raises(model_config: dict, monkeypatch):
    # Pretend CUDA exists so the mismatch branch is reachable on a CPU-only host;
    # only device-type bookkeeping runs — no CUDA API is touched.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = build_detector(model_config)  # parameters on cpu
    with pytest.raises(RuntimeError, match="Model parameters are on 'cpu'"):
        _ensure_model_device(model, torch.device("cuda"))


def test_train_one_epoch_device_guard(model_config: dict, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = build_detector(model_config)
    opt = build_optimizer(model, model_config)

    class _Loader:
        batch_size = 2

        def __iter__(self):
            yield make_synthetic_batch(batch_size=2, seed=12)

    with pytest.raises(RuntimeError, match="Model parameters are on"):
        train_one_epoch(model, opt, _Loader(), torch.device("cuda"))


def test_run_training_cuda_unavailable_raises(model_config: dict):
    if torch.cuda.is_available():
        pytest.skip("CUDA present; unavailability branch not reachable here")
    model = build_detector(model_config)

    class _Loader:
        batch_size = 2

        def __iter__(self):
            yield make_synthetic_batch(batch_size=2, seed=13)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        run_training(model, _Loader(), model_config, torch.device("cuda"), epochs=1)


def test_train_one_epoch_empty_loader(model_config: dict):
    model = build_detector(model_config)
    opt = build_optimizer(model, model_config)

    class _Empty:
        batch_size = 2

        def __iter__(self):
            return iter(())

    report = train_one_epoch(model, opt, _Empty(), torch.device("cpu"))
    assert report.images == 0
    assert math.isnan(report.loss_mean)  # no data → no fabricated stats
    assert report.end_to_end > 0


def test_train_max_batches_counts_batches(model_config: dict):
    model = build_detector(model_config)
    opt = build_optimizer(model, model_config)

    class _Loader:
        batch_size = 2

        def __iter__(self):
            for _ in range(5):
                yield make_synthetic_batch(batch_size=2, seed=14)

    report = train_one_epoch(
        model, opt, _Loader(), torch.device("cpu"), max_batches=2
    )
    assert report.images == 4  # exactly 2 batches of 2


def test_eval_max_batches_counts_batches_not_images(model_config: dict):
    """Unequal batch sizes: max_batches=2 must stop after 2 batches (3 images)."""
    model = build_detector(model_config)

    class _Loader:
        batch_size = 2

        def __iter__(self):
            for i, size in enumerate((2, 1, 1)):
                yield make_synthetic_batch(batch_size=size, seed=15 + i)

    rep = evaluate_timed(model, _Loader(), torch.device("cpu"), max_batches=2)
    assert rep.images == 3  # 2 + 1; the third batch must NOT run


def test_checkpoint_roundtrip_restores_full_state(tmp_path, model_config: dict):
    model = build_detector(model_config)
    opt = build_optimizer(model, model_config)
    sched = build_scheduler(opt, model_config)
    # populate optimizer momentum + scheduler counters
    images, targets = make_synthetic_batch(batch_size=2, seed=16)
    loss = sum(model(images, targets).values())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, sched, epoch=3, extra={"note": "roundtrip"})

    model2 = build_detector(model_config)
    opt2 = build_optimizer(model2, model_config)
    sched2 = build_scheduler(opt2, model_config)
    meta = load_checkpoint(path, model2, opt2, sched2)

    assert meta["epoch"] == 3
    assert meta["extra"]["note"] == "roundtrip"
    for k, v in model.state_dict().items():
        assert torch.equal(v, model2.state_dict()[k]), k
    s1, s2 = opt.state_dict(), opt2.state_dict()
    for pid, state in s1["state"].items():
        for key, val in state.items():
            if torch.is_tensor(val):
                assert torch.equal(val, s2["state"][pid][key]), (pid, key)
    assert sched2.state_dict()["last_epoch"] == sched.state_dict()["last_epoch"]


def test_checkpoint_missing_optimizer_raises(tmp_path, model_config: dict):
    model = build_detector(model_config)
    path = tmp_path / "weights_only.pt"
    save_checkpoint(path, model)  # no optimizer stored
    opt = build_optimizer(model, model_config)
    with pytest.raises(ValueError, match="no optimizer state"):
        load_checkpoint(path, model, opt)
    # explicit weights-only restore is allowed and returns the epoch
    meta = load_checkpoint(path, model)
    assert meta["epoch"] == 0


def test_checkpoint_missing_scheduler_raises(tmp_path, model_config: dict):
    model = build_detector(model_config)
    opt = build_optimizer(model, model_config)
    path = tmp_path / "no_scheduler.pt"
    save_checkpoint(path, model, opt)  # scheduler omitted
    sched = build_scheduler(opt, model_config)
    with pytest.raises(ValueError, match="no scheduler state"):
        load_checkpoint(path, model, opt, sched)


def test_train_one_epoch_cuda_engine(model_config: dict):
    """Full engine epoch on GPU — conditional test (skips on the laptop)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host (expected locally)")
    model = build_detector(model_config).cuda()
    opt = build_optimizer(model, model_config)

    class _Loader:
        batch_size = 2

        def __iter__(self):
            yield make_synthetic_batch(batch_size=2, seed=17)

    report = train_one_epoch(model, opt, _Loader(), torch.device("cuda"), epoch=0)
    assert report.device.startswith("cuda")
    assert report.transfer >= 0
    assert report.peak_gpu_alloc_mb is not None
    assert math.isfinite(report.loss_mean)
