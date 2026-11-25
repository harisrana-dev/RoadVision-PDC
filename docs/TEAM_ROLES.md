# Team Roles & Git Workflow — RoadVision-PDC

## Roles

### Haris — Team Lead
- Experimental methodology and protocol compliance
  ([EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md))
- CPU implementation (local Experiment A)
- **Common training engine** (`src/roadvision/engine/`) — owns the single
  shared code path used by both devices
- Benchmark architecture (`src/roadvision/benchmarks/`)
- Integration, technical review, final report assembly

### Member 2 — GPU
- Google Colab CPU/GPU experiment execution (Experiments B, C, D)
- CUDA execution details: synchronization, memory stats, mixed-precision
  decisions (if any)
- GPU profiling and identifying host-side bottlenecks
- Fills the GPU columns of the Experiment D table

### Member 3 — Dataset & Evaluation
- Dataset engineering: download orchestration (post-approval), validation
  reports, split manifests
- Annotation processing and quality auditing (exclusion logs)
- Evaluation: pycocotools integration, mAP reporting, per-class/country
  breakdowns
- Visualization of detections and error analysis

All three: reviewable contributions via PRs; no direct pushes to `main`.

---

## Git branching guidelines

```
main            — protected; always green (tests pass); only via PR
├── feat/<short-desc>     new functionality
├── fix/<short-desc>      bug fixes
├── exp/<short-desc>      experiment scripts/notebooks (no library logic)
└── docs/<short-desc>     documentation only
```

Rules:

1. Branch from up-to-date `main`; keep branches short-lived (< 2 days).
2. Commit messages: imperative mood, scoped, e.g.
   `add VOC box parser with exclusion logging`.
3. Never commit: `.venv/`, `data/raw/`, `data/processed/`, `*.pt`,
   `*.pth`, `.env`, large outputs — the `.gitignore` enforces this;
   double-check `git status` before every commit.
4. Never flip `dataset.approved` to `true` in a committed file without an
   approval record in `docs/DATASET_APPROVAL.md`.

## Pull-request guidelines

1. **Every** contribution arrives as a PR — including the lead's.
2. PR description must state: what changed, why, how it was tested
   (command + result), and any protocol impact.
3. Minimum **one reviewer** from the other two members; the lead reviews
   all engine/benchmark changes; Member 3 reviews all data/eval changes.
4. CI (GitHub Actions: ruff + pytest) must pass before merge.
5. Performance numbers quoted in docs/PRs must reference a run artifact
   under `results/` — no pasted-from-memory values.
6. Reviews are constructive and specific (file + line where possible).

## Suggested milestone split (foundation phase)

| Owner | First deliverable |
|---|---|
| Haris | shared engine + benchmark profiler + CI |
| Member 2 | Colab notebooks 04/05 device checks + GPU memory instrumentation review |
| Member 3 | annotation validation report + split manifest once dataset approved |
