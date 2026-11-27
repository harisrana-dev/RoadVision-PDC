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

Everything except the device must be identical — same *bytes* for the
config, the initial-weight snapshot and the data lists (see §1.1 for the
reproducibility level actually guaranteed):

- same model architecture and **identical initial model weights** obtained
  via the mandatory procedure in §1.1 (snapshot once → hash → reload the
  same snapshot into every device run),
- same dataset split (seed 42, same image lists),
- same preprocessing and augmentations (same RNG seeds),
- same batch size, optimizer, learning rate, scheduler, epochs,
- same DataLoader settings (`num_workers`, `pin_memory`, shuffle order —
  §2),
- same evaluation protocol.

Recorded per device: hardware inventory, PyTorch/Torchvision/CUDA versions,
training time, validation time, throughput, peak memory, and all metrics
from §3.

**Hardware differences in A, B, C must be stated explicitly** in every
table/figure produced from them — they are *not* interchangeable with D.

### §1.1 Identical-initial-state procedure (mandatory for D)

Implemented by `src/roadvision/benchmarks/protocol.py`
(`snapshot_run_state` / `restore_run_state` / `state_dict_hash`):

1. `seed_everything(seed)`; build the model (backbone/head init as
   configured), optimizer, scheduler exactly as the config specifies.
2. **Before any warm-up or training**, take `snapshot_run_state(model,
   optimizer, scheduler)` — model weights, optimizer state, scheduler state
   and all RNG states, cloned to CPU so the identical bytes serve both the
   CPU and the GPU run.
3. Record in the run manifest: `config_hash` (`roadvision.config.config_hash`)
   and `weight_hash` (`protocol.state_dict_hash`) — both must match across
   every run being compared.
4. For each device run × each repeat: `restore_run_state(snapshot)` →
   warm-up (§4) → `restore_run_state(snapshot)` again → measured unit.
5. **Reproducibility level (honest):** initial weights are byte-identical
   across runs by construction (same snapshot + hashes recorded). Training
   trajectories are deterministic on CPU with fixed seeds and data order,
   but GPU kernels may be nondeterministic — we report mean ± std over
   repeats and **never claim byte-identical GPU results**.

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
| DataLoader order | `shuffle=True` with a dedicated `torch.Generator(seed)` per run; same split-manifest image list (never rely on default global RNG for order) |
| Measured unit | one training epoch over the measured subset (pilots: fixed `max_batches` window) — same unit for every sample in a table |
| Initial state | snapshot + weight/config hashes per §1.1 before every device run × repeat |

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
4. **CPU peak RSS** — *sampled peak*: `psutil` current RSS read once per
   batch during the measured run; the reported value is the **maximum of
   those samples** (a lower bound on the instantaneous true peak between
   samples — never the process-lifetime OS peak, and never a single
   end-of-epoch spot check). Recorded on every device; on CPU runs this is
   the main memory metric.
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

### Definitions (config `benchmark:` section)

| Config key | Meaning |
|---|---|
| `warmup_iterations` | number of **training batches** run before each measured unit to warm system state (CUDA kernel/cudnn algorithm selection, GPU allocator caches, DataLoader worker startup, CPU thread pools). **Excluded from every statistic.** |
| `repeat_runs` | number of **measured units** (default: one training epoch each) aggregated into mean ± std |

Warm-up performs optimizer updates, so it *changes model weights*. A valid
protocol therefore restores the controlled initial state after warm-up —
otherwise the CPU and GPU runs (and every repeat) would measure from
different weights:

```
snapshot = snapshot_run_state(model, optimizer, scheduler)   # §1.1, before anything
for run in range(repeat_runs):              # each sample starts identically
    restore_run_state(snapshot, ...)        # model + optimizer + scheduler + RNG
    for i in range(warmup_iterations):      # system warm-up — EXCLUDED
        train_one_batch(...)                # weights change HERE
    restore_run_state(snapshot, ...)        # measured run starts from INITIAL state
    if device is cuda: torch.cuda.synchronize()
    reset GPU peak-memory stats; start RSS sampling window
    t0 = time.perf_counter()
    ... one measured unit (epoch / max_batches window) ...
    if device is cuda: torch.cuda.synchronize()
    t1 = time.perf_counter()
    record sample (t1-t0) + timing buckets + memory + config/weight hashes
```

What warm-up warms: kernel/algorithm selection, allocator and cache state,
worker/thread startup. What it contributes to: **nothing reported** — all
samples come from runs restored to identical initial state (§1.1), so
warm-up weight drift can never leak into measured numbers.

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
- Warm-up iterations excluded from all statistics (and see state-restore
  rule above).
- GPU memory counters reset (`torch.cuda.reset_peak_memory_stats()`) before
  each measured run; CPU RSS sampling window restarts with each run.
- All run state (model/optimizer/scheduler/RNG) restored from the §1.1
  snapshot before warm-up **and** before each measured unit.

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
Weight hash: <sha256 of the §1.1 initial-state snapshot — must match across compared runs>
Epochs: N, batch: B, repeats: R (warmup W batches, state-restore)

                CPU          GPU
train time (s)  x.xx ± y.yy  x.xx ± y.yy
throughput      x.xx img/s   x.xx img/s
peak mem        RSS x.x GB   alloc x.x GB
val mAP@0.5     x.xxx        x.xxx
speedup (GPU)  = CPU_time / GPU_time = x.xx ×   [only in D]
```

---

## 9. Reproducibility checklist (before publishing any table)

- [ ] `config_hash` recorded and identical across all compared runs
- [ ] `weight_hash` recorded and identical across all compared runs (§1.1)
- [ ] initial-state snapshot taken **before** any warm-up or training
- [ ] `restore_run_state` applied before warm-up **and** before each
      measured unit (state-restore warm-up, §4)
- [ ] split manifest (seed 42, image lists) identical across runs
- [ ] DataLoader shuffle order controlled by a dedicated seeded generator
- [ ] `warmup_iterations` and `repeat_runs` values logged; warm-up samples
      excluded from every statistic
- [ ] every repeat recorded as an individual sample (mean ± std computed
      from the samples, not from a pre-aggregated number)
- [ ] CUDA: sync before/after timestamps; GPU peak stats reset per unit
- [ ] CPU: per-batch RSS sampling active; GPU memory fields null
- [ ] hardware report collected at runtime (no assumed GPU model/VRAM)
- [ ] speedup only for matched Experiment D pairs
- [ ] no parallel-efficiency number without a documented scaling study (§3)
- [ ] detection metrics (mAP) in a separate table from performance (§3)
- [ ] `dataset.approved` still `false` unless dataset approval is recorded
- [ ] no dataset files, weights, checkpoints or secrets committed
