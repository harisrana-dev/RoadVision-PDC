# Project Overview — RoadVision-PDC

## Identity

- **Project:** RoadVision-PDC — CPU–GPU Accelerated Road Damage Detection Using Deep Learning
- **Course:** Parallel & Distributed Computing, CS09645|11 (3 credit hours)
- **Instructor:** Nayab Gull, BSCS, The University of Lahore
- **Team size:** 3 students

## Research question

> How do CPU and GPU execution environments affect the computational
> performance and training efficiency of a deep-learning-based road damage
> detection system under controlled experimental conditions?

## Application domains

Computer Vision · Deep Learning · Intelligent Transportation Systems ·
Automotive Intelligence · Parallel Computing

## Technical direction

| Component | Choice | Status |
|---|---|---|
| Dataset | RDD2022 (six countries, Pascal VOC XML) | candidate; **instructor approval PENDING** |
| Task | Object detection (not classification) | fixed by brief |
| Classes | D00 longitudinal, D10 transverse, D20 alligator, D40 potholes | fixed |
| Model | Torchvision `ssdlite320_mobilenet_v3_large` | **initial candidate only** — feasibility test required before finalization |
| Frameworks | Python 3.11+, PyTorch, Torchvision, NumPy, Pandas, Pillow, Matplotlib, Seaborn, PyYAML, tqdm, OpenCV, pycocotools | pinned in `requirements.txt` |
| Local env | Windows laptop, **CPU-only** PyTorch (no CUDA assumed) | set up |
| Cloud env | Google Colab CPU + GPU runtimes | notebooks prepared |

## Controlled experiment structure

The performance study is deliberately structured so that the *only*
CPU-vs-GPU comparison with everything else held constant is:

- **Experiment D (primary):** Colab CPU vs Colab GPU **inside the same Colab
  GPU runtime**, identical model initialization, dataset split,
  preprocessing, batch size, optimizer, learning rate, and training
  configuration.

Supplementary comparisons (each documents its hardware differences):

- **Experiment A:** local laptop CPU (i5-8350U) — supplementary baseline.
- **Experiment B:** Colab CPU.
- **Experiment C:** Colab GPU.

## Honesty constraints (binding for all team members)

1. Never fabricate or estimate performance numbers — report only measured
   values with mean ± std and the protocol that produced them.
2. Never invent hardware specifications — collect them at runtime
   (`roadvision.benchmarks.profiler.collect_hardware_info`).
3. Never download the dataset before `DATASET_APPROVED` is enabled and
   `docs/DATASET_APPROVAL.md` records instructor approval.
4. Never commit raw data, archives, or checkpoints.
5. Never claim GPU speedup, parallel efficiency, or accuracy results that
   have not been produced by an actual run of the shared engine.

## Status

Foundation (Stages 1–5) complete: repository, CPU environment, dataset
infrastructure (download gated OFF), model factory, shared training engine,
benchmark utilities, tests, CI. **No dataset downloaded, no training run,
no performance numbers reported yet.**
