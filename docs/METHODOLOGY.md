# Methodology — RoadVision-PDC

## 1. Overall approach

1. **Foundation first** — reproducible repo, CPU-only local environment,
   gated dataset tooling, shared training engine, tests.
2. **Pilot before scale** — every capability is exercised on synthetic data
   or a tiny subset before any full run.
3. **One engine, many devices** — CPU and GPU runs execute the *same*
   code path with a `device` parameter; no forked implementations.
4. **Measure, don't assert** — performance claims require protocol-compliant
   runs (see [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)).

## 2. Dataset methodology

- **Source:** RDD2022 official archives only (approval-gated).
- **Annotations:** Pascal VOC XML parsed by
  `src/roadvision/data/annotations.py`.

### Coordinate conversion (documented carefully)

VOC boxes are **1-based, inclusive** pixel coordinates
`(xmin, ymin, xmax, ymax)` on the original image. Torchvision detection
models expect **0-based float** `xyxy` in the *current* image frame:

```
x1_torch = xmin - 1.0
y1_torch = ymin - 1.0
x2_torch = xmax          # inclusive VOC xmax maps to exclusive-style x2
y2_torch = ymax
```

After any resize, boxes are scaled by `(new_w / old_w, new_h / old_h)` and
clamped to the new bounds. Degenerate boxes (`x2 <= x1` or `y2 <= y1`) or
boxes fully outside the image are **excluded and logged with a reason** —
never silently dropped, never "fixed" by inventing coordinates.

### Splits and leakage prevention

- Only **annotated training images** are split (official RDD2022 test
  images have **no ground truth** and are never used for metrics).
- Split is **image-level and deterministic**: seed 42, 80% train / 20%
  validation, keyed by SHA-256 of the relative image path — not by file
  order.
- **Leakage guard:** before finalize, the split builder asserts
  (a) zero filename overlap between splits,
  (b) zero content-hash overlap (same image bytes under different names
      cannot land in both splits),
  (c) duplicate basenames across countries are disambiguated by path.
- **Recorded artifacts:** split seed, per-split image counts, per-country
  distribution (JSON written to `data/processed/split_manifest.json`).

### Class mapping

Background is index **0** (Torchvision SSD convention); damage classes are
`D00→1, D10→2, D20→3, D40→4`. The detection head is built with
`num_classes = 5`.

### Targets

Each sample yields `(image, target)` where target contains
`boxes (FloatTensor N×4 xyxy)`, `labels (Int64 N)`, `image_id (Int64)`,
`area (FloatTensor N)`, `iscrowd (Int64 N, all zeros for RDD2022)`.

## 3. Model methodology

- Factory: `src/roadvision/models/detector.py::build_detector(config)`.
- **ImageNet-pretrained MobileNetV3-Large backbone + freshly initialized
  detection head** — the full COCO-trained SSDLite weights cannot be loaded
  into a 5-class head, so we keep backbone features (transfer learning) and
  re-initialize the task head. This is documented, not hidden.
- Architecture is **common to CPU and GPU** experiments; no device-specific
  layers.

## 4. Training methodology

- Single engine: `src/roadvision/engine/train.py`
  (`train_one_epoch`, `run_training`). Device is an argument.
- Reproducibility: seed python/numpy/torch (+ dataloader generator) from
  `configs/experiment.yaml`.
- Evaluation: `src/roadvision/engine/evaluate.py` using **pycocotools**
  mAP@0.5 on the validation split; installation and known-answer
  correctness verified by tests — no unvalidated mAP reimplementation.

## 5. Benchmark methodology

See [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) §§3–5. Summary of binding
rules:

- warm-up iterations excluded; ≥5 repeats; report mean ± std;
- CUDA synchronized around GPU timers;
- data-loading / transfer / compute / end-to-end measured on the critical
  path without double-counting;
- CPU peak RSS via psutil; GPU peak allocated via
  `torch.cuda.max_memory_allocated()`;
- **no parallel-efficiency metric** without a valid scaling experiment;
- hardware collected at runtime — never hard-coded or guessed.

## 6. Quality assurance

- pytest suite on **synthetic fixtures** (no dataset needed): annotation
  parsing, coordinate conversion, invalid boxes, dataset loading, model
  forward/backward, benchmark math, download-gate refusal.
- Ruff lint; GitHub Actions CI on push/PR.
- Notebooks validated as nbformat JSON and smoke-executed where feasible.

## 7. Threats to validity (acknowledged)

- Colab CPU/GPU host varies between sessions → Experiment D runs both
  devices in the *same* session to control this.
- Single seed (42) for v1 pilots → variance across seeds deferred to later
  work; speedup claims will note n=1 seed unless repeated.
- Mobile CPU (i5-8350U) is not a server CPU → labeled supplementary.
- RDD2022 class imbalance and country bias → detection-quality tables are
  broken down by class and country where sample counts permit.
