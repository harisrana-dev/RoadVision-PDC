# Experiment Design — RoadVision-PDC

This document is the binding protocol for all performance experiments.
Numbers in reports must come from runs that followed this protocol; anything
else must be labeled clearly (e.g., "exploratory, not protocol-compliant").

---

## 1. Experiment structure (mandatory corrections applied)

| ID | Comparison | Hardware | Role |
|----|------------|----------|------|
| **A** | Local laptop CPU | HP EliteBook, Intel i5-8350U (4C/8T), 16 GB RAM, **no NVIDIA GPU** | **Supplementary** — a single low-power mobile CPU; results are context, not a universal CPU baseline |
| **B** | Google Colab CPU | Colab host CPU (specs recorded at runtime) | Controlled reference |
| **C** | Google Colab GPU | Colab GPU (model varies per session; recorded at runtime) | Controlled reference |
| **D** | **CPU vs GPU inside ONE Colab GPU runtime** | Same physical host; only `device` differs | **PRIMARY controlled comparison** |

### Experiment D — controlled comparison rules

Everything except the device must be byte-identical:

- same model architecture and **identical initial model weights**
  (same seed → same initialization; optionally save the initial state_dict
  once and reload it for the second device),
- same dataset split (seed 42, same image lists),
- same preprocessing and augmentations (same RNG seeds),
- same batch size, optimizer, learning rate, scheduler, epochs,
- same DataLoader settings (`num_workers`, `pin_memory`),
- same evaluation protocol.

Recorded per device: hardware inventory, PyTorch/Torchvision/CUDA versions,
training time, validation time, throughput, peak memory, and all metrics
from §3.

**Hardware differences in A, B, C must be stated explicitly** in every
table/figure produced from them — they are *not* interchangeable with D.

---

## 2. Controlled variables checklist (all experiments)

| Variable | Value (v1) |
|---|---|
| Architecture | SSDLite320 MobileNetV3-Large (candidate) |
| num_classes | 5 (4 + background) |
| Backbone init | ImageNet-1K pretrained (`weights_backbone=IMAGENET1K_V1`) |
| Head init | fresh (COCO full weights incompatible with 5 classes) |
| Split seed | 42 (train/val 80/20 of annotated images) |
| Resize | 320×320 |
| Batch size | 8 (primary); sensitivity runs at 4 and 16 |
| Optimizer | SGD, lr=5e-3, momentum=0.9, weight_decay=5e-4 |
| Scheduler | StepLR(step=8, γ=0.1) |
| Epochs | pilot: 1–2; full: 10 (config `training.epochs`) |
| Seed | 42 everywhere (torch, numpy, python, dataloader generator) |

---

## 3. Metrics reported

For every timed configuration report:

1. **Training time** — end-to-end seconds per epoch and total, mean ± std
   across `benchmark.repeat_runs` (protocol §4).
2. **Throughput** — images processed / elapsed second (train and eval
   separately).
3. **Speedup** — `CPU elapsed time / GPU elapsed time` for otherwise
   identical runs (Experiment D primary; A/B/C only with a hardware
   disclaimer).
4. **CPU peak RSS** — via `psutil` sampling of the process (recorded on
   every device; on CPU runs this is the main memory metric).
5. **GPU peak allocated** — `torch.cuda.max_memory_allocated()` /
   `max_memory_reserved()`, reset before each measured run. Recorded only
   when CUDA is active; **never estimated for CPU runs**.
6. **Mean and standard deviation** over repeated measurements — single
   measurements are insufficient.

### Metrics explicitly NOT reported (v1)

- **CPU/GPU parallel efficiency** (`speedup / core count`) is **not
  reported** unless a mathematically valid strong/weak scaling experiment
  (fixed problem size vs. varying thread counts, or scaled problem size)
  is implemented and documented. Dividing a 2-device speedup by a core
  count is not a valid efficiency measure for this study.
- Any GPU speedup claim before Experiment D runs.
- Accuracy/detection-quality numbers before a real training run.

### Detection quality (separate from performance)

- mAP@IoU=0.5 per class and mean, computed with **pycocotools** on the
  validation split (see §6).
- Training performance and detection quality are always reported in
  separate tables.

---

## 4. Timing protocol

```
for run in range(repeat_runs):          # default 5
    for i in range(warmup_iterations):  # default 3 — EXCLUDED from stats
        train_one_batch(...)
    if device is cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    ... one measured epoch / step ...
    if device is cuda: torch.cuda.synchronize()
    t1 = time.perf_counter()
```

### Timing buckets (no double-counting)

| Bucket | Definition | Overlap rule |
|---|---|---|
| `data_loading` | wall time spent pulling the next batch from the DataLoader iterator | measured in the consumer *before* the batch is used; a batch already sitting in the prefetch queue counts here once |
| `transfer` | host→device copy (`tensor.to(device, non_blocking=...)` + sync) | starts *after* data_loading ends; on CPU runs = 0 |
| `compute` | forward + loss + backward + optimizer step, with sync before/after | starts *after* transfer ends |
| `end_to_end` | wall clock of the whole measured epoch | **contains** the other three — never summed with them |

Because `data_loading` for batch *i+1* can overlap `compute` for batch *i*
(pipelined loaders), the buckets are **measured on the critical path** of
each iteration and reported individually; we do **not** claim
`data_loading + transfer + compute = end_to_end`. The residual (scheduler
overhead, metric bookkeeping) is reported as `other = end_to_end −
(data_loading + transfer + compute)` when positive.

### CUDA rules

- `torch.cuda.synchronize()` immediately before every `t0`/`t1` on GPU.
- Warm-up iterations excluded from all statistics.
- GPU memory counters reset (`torch.cuda.reset_peak_memory_stats()`) before
  each measured run.

---

## 5. Proposed experiments

| # | Experiment | Configurations | Notes |
|---|---|---|---|
| A | CPU vs GPU baseline | device=cpu on laptop; device=cpu+cuda on Colab | A is supplementary; cross-hardware comparisons labeled as such |
| B | Batch-size sensitivity | batch ∈ {4, 8, 16} | if a device OOMs or cannot allocate, **record the failure** — do not hide or silently downscale |
| C | Throughput comparison | images/s per device per batch size | mean ± std over repeats |
| D | Training-time comparison (primary) | same Colab runtime, cpu vs cuda | controlled protocol §1 |
| E | Memory utilization | CPU RSS; GPU peak allocated/reserved | CPU and GPU metrics are different quantities — never compare them as equals |
| F | Detection-quality comparison | val mAP per device run | identical training config; differences beyond seed noise discussed honestly |

---

## 6. Evaluation dependency policy

- **Evaluator: pycocotools** (`pycocotools.coco.COCO` + `loadRes` +
  `evaluate('bbox')`), the standard COCO mAP implementation.
- pycocotools installation is **verified** in CI and by
  `scripts/smoke_test.py` (import + a tiny ground-truth/prediction round
  trip with a known-answer case).
- **No pure-PyTorch mAP fallback is implemented or claimed** unless it is
  validated against pycocotools on identical inputs with matching results
  (documented test). An unvalidated fallback would risk reporting wrong
  quality numbers.

---

## 7. Model feasibility (candidate not frozen)

`ssdlite320_mobilenet_v3_large` is the **initial candidate**, not a
permanent decision.

Before finalization the team runs the synthetic CPU feasibility test
(`python scripts/smoke_test.py --feasibility`):

1. model constructs with `num_classes=5`,
2. ImageNet backbone weights load (fresh head documented),
3. training forward returns finite losses on synthetic images,
4. backward + optimizer step succeed on CPU,
5. measured forward/backward wall time and parameter count recorded in
   `results/benchmarks/feasibility_ssdlite_cpu.json`,
6. alternative candidates (e.g., FasterRCNN-MobileNetV3, YOLO-class
   detectors outside torchvision) considered if feasibility fails or CPU
   throughput is too low for pilot runs.

Decision + rationale are appended to this section **after** the test runs —
with actual measured numbers only.

### Decision record

**Status: candidate RETAINED for the pilot phase (measured, not assumed).**

Synthetic CPU feasibility test run on the local laptop
(`python scripts/smoke_test.py --feasibility`, raw artifact:
`results/benchmarks/feasibility_ssdlite_cpu.json`):

| Quantity | Measured value |
|---|---|
| Model | `ssdlite320_mobilenet_v3_large`, `num_classes=5` |
| Total parameters | 3,758,900 (all trainable) |
| Backbone init | ImageNet-1K pretrained (`IMAGENET1K_V1`) |
| Head init | fresh (`weights=null`) |
| Batch size / image size | 8 / 320×320 |
| Step time (fwd+bwd+step), CPU | **3.03 ± 0.51 s** (warm-up 3 excluded, n=5) |
| Loss after test step | 8.006 (finite) |
| Forward/backward/grad/step | all PASS; CUDA path PASS/expected-skip |

Interpretation (honest):

- The model **runs correctly on CPU** — feasibility criterion met.
- At ~3 s/step the local i5-8350U makes **full-size training impractical
  locally**; local runs are limited to feasibility/smoke/pilot subsets
  (Experiment A = supplementary only), which matches the experiment plan.
- The architecture remains a **candidate**: finalization requires the
  Experiment B/C/D runs on Colab plus a detection-quality comparison
  (Experiment F). Alternatives (e.g., torchvision FasterRCNN-MobileNetV3)
  will be considered if Colab pilot throughput or quality is inadequate.
- These numbers are a local CPU measurement only — **not** a GPU
  comparison and not a performance claim across devices.

---

## 8. Reporting template

```
Experiment D run <id>
Host: <Colab runtime host, CPU cores, RAM>
GPU: <model, VRAM, CUDA, torch/tv versions>
Config hash: <sha256 of configs/experiment.yaml + split seed>
Epochs: N, batch: B, repeats: R (warmup W)

                CPU          GPU
train time (s)  x.xx ± y.yy  x.xx ± y.yy
throughput      x.xx img/s   x.xx img/s
peak mem        RSS x.x GB   alloc x.x GB
val mAP@0.5     x.xxx        x.xxx
speedup (GPU)  = CPU_time / GPU_time = x.xx ×   [only in D]
```
