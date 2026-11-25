# RoadVision-PDC

**CPU–GPU Accelerated Road Damage Detection Using Deep Learning**

| | |
|---|---|
| Course | Parallel & Distributed Computing (CS09645\|11) |
| Instructor | Nayab Gull |
| Team | Haris (lead), Member 2 (GPU/CUDA), Member 3 (dataset/evaluation) |
| Dataset | RDD2022 (candidate — **approval PENDING**, see [docs/DATASET_APPROVAL.md](docs/DATASET_APPROVAL.md)) |
| Model candidate | Torchvision SSDLite320 + MobileNetV3-Large (not yet finalized) |

**Research question:** How do CPU and GPU execution environments affect the
computational performance and training efficiency of a deep-learning-based
road damage detection system under controlled experimental conditions?

> **Honesty rules for this repo:** no fabricated results, no invented
> hardware specs, no unapproved dataset downloads, no performance claims
> before measurements exist. Raw data and checkpoints are never committed.

---

## Repository layout

```
RoadVision-PDC/
├── configs/experiment.yaml     # single shared config for CPU & GPU runs
├── data/{raw,processed}/       # dataset roots (git-ignored, empty)
├── notebooks/                  # 01..06, runnable locally or in Colab
├── src/roadvision/
│   ├── config.py               # YAML config loading
│   ├── data/                   # VOC parser, dataset, split, gated download
│   ├── models/detector.py      # SSDLite320 factory
│   ├── engine/                 # ONE shared train/eval engine (device param)
│   ├── benchmarks/             # profiler + metrics (mean/std, speedup, RSS)
│   └── visualization/plots.py
├── scripts/                    # check_environment / validate_dataset / smoke_test
├── tests/                      # pytest, synthetic fixtures only
├── results/                    # benchmark CSVs + figures (git-ignored outputs)
└── docs/                       # overview, methodology, experiments, approval, roles
```

---

## Local setup (Windows PowerShell)

Requires **Python 3.11+** (developed on 3.13.5), Git.

```powershell
cd C:\Projects\RoadVision-PDC

# 1. Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. CPU-only PyTorch (no CUDA packages on the laptop)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu

# 3. Everything else + editable package install
pip install -r requirements.txt
pip install -e .
```

If PowerShell blocks script activation, run once (current user scope):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Verify the environment

```powershell
python scripts\check_environment.py     # prints OS, CPU, RAM, GPU/CUDA status, versions
python scripts\smoke_test.py            # synthetic model init + 1 train step on CPU
pytest                                  # full test suite (synthetic data only)
ruff check src tests scripts            # lint
```

### Dataset (disabled until approved)

The downloader **refuses to run** while `DATASET_APPROVED=false`
(`configs/experiment.yaml` + `.env`). See
[docs/DATASET_APPROVAL.md](docs/DATASET_APPROVAL.md) for the approval
procedure and archive sizes. Once approved:

```powershell
$env:DATASET_APPROVED = "true"
python -m roadvision.data.download --countries Czech India Japan
python scripts\validate_dataset.py
```

Nothing is downloaded without that explicit opt-in.

---

## Google Colab

See [notebooks/01_environment_check.ipynb](notebooks/01_environment_check.ipynb)
for the canonical Colab bootstrap. Summary:

```python
# clone your private repo (use a PAT or gdown for the zip if needed)
!git clone https://<you>@github.com/<you>/RoadVision-PDC.git
%cd RoadVision-PDC
!pip install -q -r requirements-colab.txt
# torch/torchvision: use Colab's preinstalled pair (GPU-enabled);
# do NOT install CPU wheels inside a GPU runtime.
```

Colab notebooks in this repo **never** hardcode a GPU model and never launch
long training automatically — pilot subsets and pilot epoch counts only.
Mount Drive only when persisting results.

---

## Experiment design (summary)

| ID | Comparison | Hardware | Role |
|----|------------|----------|------|
| A | Local laptop CPU | HP EliteBook, i5-8350U | supplementary |
| B | Colab CPU | Colab host CPU | controlled reference |
| C | Colab GPU | Colab GPU (varies) | controlled reference |
| **D** | **CPU vs GPU inside one Colab GPU runtime** | same host, same run | **primary controlled comparison** |

Experiment D holds architecture, initial weights, split, preprocessing,
batch size, optimizer, LR, and all other settings constant; only the device
varies. All other experiments document their hardware differences explicitly.

Full protocol (timing buckets, warm-up, mean ± std, memory metrics, and the
decision **not** to report parallel efficiency without a valid scaling
experiment): [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md).

---

## Team & Git workflow

[docs/TEAM_ROLES.md](docs/TEAM_ROLES.md) — role assignments, branch naming,
PR review rules. Short version: every contribution arrives as a reviewed PR;
nobody pushes directly to `main`.

## References

[docs/REFERENCES.md](docs/REFERENCES.md) — dataset paper, model papers,
torchvision docs, citation strings.
