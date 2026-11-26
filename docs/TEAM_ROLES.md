# Team Roles & Collaboration — RoadVision-PDC

## 1. Purpose and team structure

This document defines who owns what, how work moves through Git, and what "done"
means. It exists so that three developers can contribute in parallel without
colliding on shared code, duplicating experiments, or weakening measurement
integrity.

| Member | Role | GitHub |
|---|---|---|
| Haris Rana | Team Lead & Core ML/Systems Engineer | [harisrana-dev](https://github.com/harisrana-dev) |
| Romaisa Irfan | Dataset & Evaluation Engineer | [romaisairfan](https://github.com/romaisairfan) |
| Mian Salman | GPU & Performance Engineer | [Miansalmannn](https://github.com/Miansalmannn) |

Haris is the technical lead and integration owner. Romaisa and Mian Salman each
implement, test, document, and defend their assigned work independently.

---

## 2. Haris Rana — Team Lead & Core ML/Systems Engineer

**Owns:**

- Experimental methodology and benchmark protocol
  ([EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)).
- Model architecture and configuration decisions
  (`src/roadvision/models/`, `configs/experiment.yaml`).
- **Shared training engine** (`src/roadvision/engine/`) — the single training code
  path used by both CPU and GPU runs.
- CPU training baseline (Experiment A).
- Benchmarking and performance-measurement framework
  (`src/roadvision/benchmarks/`).
- Experiment logging and reproducibility (seeds, config hashing, run artifacts).
- GPU implementation review and integration.
- CPU/GPU performance analysis and speedup calculations.
- Final technical integration, report, and research conclusions.

---

## 3. Romaisa Irfan — Dataset & Evaluation Engineer

**Owns:**

- Dataset inspection and acquisition preparation (post-approval execution of the
  gated downloader; never bypassing the approval gate).
- Pascal VOC annotation parsing and validation
  (`src/roadvision/data/annotations.py`).
- Dataset integrity and annotation-quality checks (exclusion logs, invalid-box
  audits).
- Preprocessing and data pipeline (`src/roadvision/data/preprocessing.py`).
- Train/validation splitting and leakage prevention
  (`src/roadvision/data/dataset.py`).
- Detection evaluation metrics (`src/roadvision/engine/evaluate.py` integration,
  mAP reporting, per-class breakdowns).
- Prediction visualizations and error analysis
  (`src/roadvision/visualization/`, evaluation notebooks).

---

## 4. Mian Salman — GPU & Performance Engineer

**Owns:**

- Google Colab GPU environment validation (notebooks 01/05 bootstrap).
- CUDA availability and hardware verification — runtime collection only, never
  hardcoded GPU assumptions.
- GPU training experiments executed **through the shared training engine**
  (Experiments B/C/D GPU sides).
- GPU timing correctness: CUDA synchronization around timers, warm-up exclusion.
- GPU memory profiling (`torch.cuda.max_memory_allocated` /
  `max_memory_reserved`, reset between runs).
- Throughput measurement on GPU runs.
- GPU experiment logging and the performance report inputs for Experiment D.

---

## 5. Shared responsibilities

All three members:

- Follow [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) when producing or quoting
  numbers.
- Keep the dataset approval gate closed until instructor approval is recorded in
  [DATASET_APPROVAL.md](DATASET_APPROVAL.md).
- Never commit datasets, checkpoints, `.env` files, or secrets.
- Never fabricate results, metrics, hardware specs, or test outcomes.
- Run relevant tests and lint before opening a PR.
- Review each other's PRs constructively and specifically.

---

## 6. Technical ownership boundaries

- **Haris owns the shared training engine and the experimental protocol.**
  Changes to `src/roadvision/engine/`, `src/roadvision/benchmarks/`, or
  `configs/experiment.yaml` are coordinated with and reviewed by Haris.
- **Romaisa owns dataset integrity and evaluation-pipeline contributions.**
  Changes under `src/roadvision/data/` and the evaluation path are reviewed by
  Romaisa (with Haris reviewing any interface impact on the engine).
- **Mian Salman owns GPU environment validation and GPU performance
  instrumentation.** GPU-specific measurement code and Colab experiment notebooks
  are his responsibility, built on top of the shared engine.

**There is exactly one training engine.** Do not create competing CPU and GPU
training implementations. GPU experiments configure the common engine with
`device="cuda"`; CPU runs use `device="cpu"`. Changes to shared interfaces must be
coordinated with and reviewed by Haris before merge.

---

## 7. Git branching strategy

Workflow:

```
dev → feature branch → pull request → dev
```

Release workflow:

```
dev → release pull request → main
```

Branch naming:

| Pattern | Use |
|---|---|
| `feature/haris-<short-description>` | Haris's work |
| `feature/romaisa-<short-description>` | Romaisa's work |
| `feature/salman-<short-description>` | Mian Salman's work |
| `fix/<short-description>` | bug fixes |
| `docs/<short-description>` | documentation only |

Rules:

1. **Never develop directly on `main`.**
2. **Never develop directly on `dev`.**
3. Start every feature branch from an **updated `dev`**
   (`git fetch origin && git checkout -b feature/... origin/dev`).
4. Open pull requests **targeting `dev`**.
5. `main` is reserved for approved stable releases; release PRs go
   `dev → main` and are opened by Haris.
6. Keep branches short-lived; rebase or merge updated `dev` back in before review.
7. Commit messages: imperative mood with a scope prefix, e.g.
   `data: add exclusion logging to VOC parser`, `docs: fix Colab clone URL`.

> Branch protection has **not** been configured or verified yet. Until it is,
> these rules are enforced by review discipline, not by GitHub settings.

---

## 8. Pull request and review requirements

Every PR must include:

- **Objective** — why this change exists.
- **Summary of changes.**
- **Files affected.**
- **Testing commands and actual results** (paste real output; never invent it).
- **Related issue number.**
- **Known limitations.**
- **Any effect on experimental methodology** (and confirmation that shared
  config/protocol changes were coordinated with Haris).

Review rules:

1. Every contribution arrives as a PR — including the lead's.
2. Haris reviews all shared-engine and benchmark changes.
3. Dataset/evaluation changes receive technical review from Romaisa; GPU work
   receives technical review from Mian Salman; cross-cutting changes are reviewed
   by Haris.
4. CI (ruff + pytest) must pass before merge.
5. Review the **actual diff** — do not merge on description alone.
6. Romaisa and Mian Salman must not merge their own work into `main`.
7. Performance numbers quoted in docs or PR descriptions must reference a run
  artifact under `results/` — no pasted-from-memory values.

---

## 9. Commit standards

- Imperative subject line, scoped prefix (`data:`, `engine:`, `gpu:`, `docs:`,
  `test:`, `bench:`).
- Explain *why* in the body when the change is non-obvious.
- One logical change per commit; no drive-by refactors inside a feature PR.
- Never commit: `.venv/`, `data/raw/`, `data/processed/`, `*.pt`, `*.pth`,
  `*.ckpt`, `.env`, credentials, generated result artifacts — `.gitignore`
  enforces most of this; verify with `git status` before every commit.
- Never flip `dataset.approved` to `true` in a committed file without an
  approval record in [DATASET_APPROVAL.md](DATASET_APPROVAL.md).

---

## 10. Data, model and secret-handling rules

- **Dataset:** acquisition only after recorded instructor approval; only official
  sources; never committed to Git in any form.
- **Model weights and checkpoints:** never committed; regenerate from configs.
- **Secrets:** no tokens, passwords, API keys, or `.env` files in the repository
  or in screenshots/logs attached to PRs or issues.
- **Results:** only measured, protocol-compliant numbers may be published;
  artifacts live under `results/` (git-ignored) and are referenced, not inlined,
  from documentation.
- **Hardware facts:** collected at runtime via
  `roadvision.benchmarks.profiler` — never guessed or hardcoded.

---

## 11. Definition of done

A piece of work is done when:

1. It is merged into `dev` through a reviewed PR (release to `main` is a separate,
   Haris-initiated step).
2. Relevant tests pass locally (`pytest`, `ruff check src tests scripts`) and CI is
   green.
3. Documentation is updated if behaviour, configuration, or workflow changed.
4. Any experimental claim is traceable to a config + run artifact, or is explicitly
   marked pending.
5. No scope beyond the assigned ownership boundary was changed without coordination.
6. The author can explain and defend the change independently (viva readiness).
