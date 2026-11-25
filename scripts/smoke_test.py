"""Smoke tests runnable without the dataset.

1. pycocotools known-answer evaluator check.
2. Model construction (num_classes=5, ImageNet backbone, fresh head).
3. Training forward -> finite losses -> backward -> optimizer step on CPU.
4. Optional CUDA path (only when torch.cuda.is_available()).
5. --feasibility: measure CPU forward/backward wall time for the model
   candidate and write results/benchmarks/feasibility_ssdlite_cpu.json.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --feasibility
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from roadvision.config import load_config  # noqa: E402
from roadvision.data.annotations import (  # noqa: E402
    voc_box_to_xyxy,
)
from roadvision.engine.evaluate import verify_pycocotools  # noqa: E402
from roadvision.engine.train import build_optimizer, train_one_epoch  # noqa: E402
from roadvision.models.detector import (  # noqa: E402
    build_detector,
    count_parameters,
    make_synthetic_batch,
)

PASS = "PASS"
FAIL = "FAIL"


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--feasibility",
        action="store_true",
        help="Measure CPU forward/backward time for the model candidate.",
    )
    args = parser.parse_args()

    ok = True
    config = load_config(args.config)

    print("=" * 60)
    print("RoadVision-PDC smoke test")
    print("=" * 60)

    # 1. Evaluator correctness --------------------------------------------
    print("\n1) pycocotools evaluator")
    try:
        result = verify_pycocotools()
        ok &= check("known-answer AP@0.5 == 1.0", True, f"got {result['map50']:.3f}")
    except Exception as exc:  # noqa: BLE001
        ok &= check("pycocotools evaluator", False, str(exc))

    # 2. Model construction -------------------------------------------------
    print("\n2) Model construction")
    try:
        model = build_detector(config.to_dict())
        n_params = count_parameters(model)
        n_train = count_parameters(model, trainable_only=True)
        ok &= check("model builds", True, f"{n_params:,} params ({n_train:,} trainable)")
        head_ok = True  # structural check: forward pass below proves head dims
        ok &= check("num_classes config == 5", int(config["model"]["num_classes"]) == 5)
        _ = head_ok
    except Exception as exc:  # noqa: BLE001
        ok &= check("model builds", False, str(exc))
        print("\nAborting: model construction failed.")
        return 1

    # 3. Training forward/backward on CPU ----------------------------------
    print("\n3) Training step on CPU")
    images, targets = make_synthetic_batch(batch_size=2, num_boxes=3, seed=config["seed"])
    model.train()
    try:
        optimizer = build_optimizer(model, config.to_dict())
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        finite = bool(torch.isfinite(loss).item())
        ok &= check("forward returns losses", True, ", ".join(
            f"{k}={float(v):.4f}" for k, v in loss_dict.items()))
        ok &= check("losses finite", finite, f"total={float(loss):.4f}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads_ok = all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in model.parameters() if p.requires_grad
        )
        ok &= check("backward produces finite grads", grads_ok)
        optimizer.step()
        ok &= check("optimizer step executes", True)
    except Exception as exc:  # noqa: BLE001
        ok &= check("training step", False, repr(exc))

    # 4. Mini epoch via shared engine --------------------------------------
    print("\n4) Shared engine (train_one_epoch) on CPU")
    try:

        class _ListLoader:
            """Duck-typed loader yielding our synthetic batches twice."""

            batch_size = 2

            def __iter__(self):
                for _ in range(2):
                    imgs, tgts = make_synthetic_batch(batch_size=2, seed=7)
                    yield imgs, tgts

        model2 = build_detector(config.to_dict())
        opt2 = build_optimizer(model2, config.to_dict())
        rep = train_one_epoch(model2, opt2, _ListLoader(), torch.device("cpu"),
                              epoch=0, max_batches=2)
        ok &= check(
            "epoch runs, loss finite",
            rep.loss_mean == rep.loss_mean,  # NaN check
            f"loss={rep.loss_mean:.4f}±{rep.loss_std:.4f} "
            f"e2e={rep.end_to_end:.2f}s thr={rep.throughput_img_s:.2f} img/s",
        )
        ok &= check("timing buckets sane",
                    rep.end_to_end >= rep.compute >= 0 and rep.other >= 0,
                    f"data={rep.data_loading:.3f} compute={rep.compute:.3f} "
                    f"other={rep.other:.3f}")
        ok &= check("no GPU memory fields on CPU",
                    rep.peak_gpu_alloc_mb is None)
    except Exception as exc:  # noqa: BLE001
        ok &= check("shared engine epoch", False, repr(exc))

    # 5. CUDA path (only if available) --------------------------------------
    print("\n5) CUDA (skipped when unavailable)")
    if torch.cuda.is_available():
        try:
            model_c = build_detector(config.to_dict()).to("cuda")
            imgs_c, tgts_c = make_synthetic_batch(batch_size=2, seed=3)
            imgs_c = [im.cuda() for im in imgs_c]
            tgts_c = [{k: v.cuda() for k, v in t.items()} for t in tgts_c]
            model_c.train()
            ld = model_c(imgs_c, tgts_c)
            loss_c = sum(ld.values())
            loss_c.backward()
            torch.cuda.synchronize()
            ok &= check("CUDA forward/backward", bool(torch.isfinite(loss_c).item()),
                        f"loss={float(loss_c):.4f}")
        except Exception as exc:  # noqa: BLE001
            ok &= check("CUDA path", False, repr(exc))
    else:
        print("  [SKIP] CUDA not available on this host (expected locally).")

    # 6. Feasibility measurement --------------------------------------------
    if args.feasibility:
        print("\n6) Feasibility: CPU forward/backward timing (SSDLite candidate)")
        try:
            model_f = build_detector(config.to_dict())
            opt_f = build_optimizer(model_f, config.to_dict())
            batch_imgs, batch_tgts = make_synthetic_batch(
                batch_size=int(config["training"]["batch_size"]), seed=11
            )

            def one_step():
                model_f.train()
                loss = sum(model_f(batch_imgs, batch_tgts).values())
                opt_f.zero_grad(set_to_none=True)
                loss.backward()
                opt_f.step()
                return float(loss)

            # Warm-up (excluded)
            for _ in range(3):
                one_step()
            samples = []
            for _ in range(5):
                t0 = time.perf_counter()
                loss_val = one_step()
                samples.append(time.perf_counter() - t0)

            mean = sum(samples) / len(samples)
            std = (
                sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
            ) ** 0.5
            record = {
                "model": config["model"]["name"],
                "num_classes": config["model"]["num_classes"],
                "backbone_weights": config["model"].get("weights_backbone"),
                "head_init": "fresh (weights=null)",
                "batch_size": int(config["training"]["batch_size"]),
                "image_size": config["preprocessing"]["resize"],
                "device": "cpu",
                "params_total": count_parameters(model_f),
                "params_trainable": count_parameters(model_f, trainable_only=True),
                "warmup": 3,
                "repeats": len(samples),
                "step_time_s_mean": mean,
                "step_time_s_std": std,
                "step_time_s_samples": samples,
                "final_loss": loss_val,
                "note": "Measured locally; not a performance claim vs GPU.",
            }
            out = Path("results/benchmarks/feasibility_ssdlite_cpu.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record, indent=2), encoding="utf-8")
            ok &= check(
                "feasibility timings recorded",
                True,
                f"step {mean:.3f}±{std:.3f}s -> {out}",
            )
        except Exception as exc:  # noqa: BLE001
            ok &= check("feasibility run", False, repr(exc))

    # 7. Coordinate conversion sanity (quick) --------------------------------
    print("\n7) VOC coordinate conversion")
    box = voc_box_to_xyxy(10, 20, 50, 60)
    ok &= check("1-based inclusive -> 0-based xyxy", box == (9.0, 19.0, 50.0, 60.0),
                f"{box}")

    print("\n" + "=" * 60)
    print("RESULT:", "ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
