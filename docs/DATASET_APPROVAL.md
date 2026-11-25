# Dataset Approval — RDD2022

> **Approval status: PENDING**
> The dataset downloader is hard-disabled while `DATASET_APPROVED=false`
> (set in `configs/experiment.yaml` and/or the environment). Do not flip the
> flag until the instructor's approval is recorded in this file.

## Dataset

- **Name:** RDD2022 — Road Damage Dataset 2022 (released via CRDDC'2022)
- **Task:** object detection of road damage in dashcam/vehicle imagery
- **Annotation format:** Pascal VOC XML (bounding boxes + class names)
- **Damage classes:** D00 (longitudinal crack), D10 (transverse crack),
  D20 (alligator crack), D40 (pothole)

## Official sources (only these are permitted)

| Resource | URL |
|---|---|
| Official code/repo | https://github.com/sekilab/RoadDamageDetector |
| Official full archive | https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip |
| Country archives (official S3) | `.../Country_Specific_Data_CRDDC2022/RDD2022_<Country>.zip` |
| FigShare mirror (organizer-published) | https://figshare.com/articles/dataset/21431547 |
| Dataset paper | arXiv:2209.08538 / Geoscience Data Journal 11(4):846–862 |

**Never** download from Kaggle re-uploads, blogs, or other unverified mirrors.

## Approximate country archive sizes (official README)

| Country archive | Size | Notes |
|---|---:|---|
| RDD2022_Czech.zip | 245.2 MB | train + test |
| RDD2022_India.zip | 502.3 MB | train + test |
| RDD2022_Japan.zip | 1022.9 MB | train + test |
| RDD2022_United_States.zip | 423.8 MB | train + test |
| RDD2022_China_MotorBike.zip | 183.1 MB | train + test |
| RDD2022_China_Drone.zip | 152.8 MB | train only |
| RDD2022_Norway.zip | 9.9 GB | train + test — excluded (disk budget) |
| RDD2022.zip (all-in-one) | several GB | not used |

## Proposed country subset (PENDING)

**Czech + India + Japan ≈ 1.77 GB total.**

Rationale:

- Three countries give multi-national variation (the dataset's headline
  feature) while staying well inside the laptop's ~18.8 GB free disk.
- Norway (9.9 GB) is excluded on disk-budget grounds.
- The subset is small enough for pilot training on Colab and local CPU
  smoke runs.

Once approved, adjust `dataset.countries` in `configs/experiment.yaml`
(keep `approved: true` only in your local config/env — do not commit a
flipped flag without the approval record below).

## License & attribution

- **Images:** Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0) — https://creativecommons.org/licenses/by-sa/4.0/
- **Code in sekilab/RoadDamageDetector:** MIT
- **Required attribution:** keep the CC BY-SA 4.0 notice with any
  redistributed images and cite the dataset paper (bibtex in
  [REFERENCES.md](REFERENCES.md)).
- Privacy: the organizers state faces/license plates are blurred where
  clearly visible.

## Official test set — important evaluation note

RDD2022's official **test images ship without ground-truth annotations**.
This project therefore:

1. splits the **annotated training images** into train/validation at the
   image level (seeded, leakage-checked — see `src/roadvision/data/dataset.py`),
2. reports detection quality on the **validation** split,
3. never assumes labels exist for the official test images.

## Recording instructor approval

When Nayab Gull approves the dataset, append below (do not delete history):

```
## Approval record
- Approved by: <instructor name>
- Date: <YYYY-MM-DD>
- Approved subset: Czech + India + Japan (or as amended)
- Evidence: <screenshot path / email / LMS note reference>
- Flag enabled by: <student name>, <YYYY-MM-DD>
```

Then set `DATASET_APPROVED=true` locally and run:

```powershell
python -m roadvision.data.download --countries Czech India Japan
```
