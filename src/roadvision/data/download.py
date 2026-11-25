"""RDD2022 download orchestrator — HARD-GATED on dataset approval.

Safety rules (binding):
* Refuses to download unless ``DATASET_APPROVED=true`` in the environment
  **and** ``dataset.approved: true`` in the config (double opt-in — a stale
  env var or a locally edited config alone cannot trigger a multi-GB fetch).
* Only official sources are used (sekilab S3 / FigShare) — never mirrors.
* Only the explicitly requested country subset is fetched.
* No data is ever written into git-tracked paths beyond data/raw (ignored).

Usage (after approval is recorded in docs/DATASET_APPROVAL.md)::

    $env:DATASET_APPROVED = "true"      # PowerShell
    python -m roadvision.data.download --countries Czech India Japan
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

from roadvision.config import DEFAULT_CONFIG_PATH, dataset_approved, load_config

logger = logging.getLogger(__name__)

# Official country archives from sekilab/RoadDamageDetector README.
# Base URL is fixed — no user-controlled hosts.
_OFFICIAL_BASE = (
    "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/"
    "RDD2022/Country_Specific_Data_CRDDC2022"
)

ARCHIVES: dict[str, dict[str, object]] = {
    "Czech": {"file": "RDD2022_Czech.zip", "size_mb": 245.2},
    "India": {"file": "RDD2022_India.zip", "size_mb": 502.3},
    "Japan": {"file": "RDD2022_Japan.zip", "size_mb": 1022.9},
    "United_States": {"file": "RDD2022_United_States.zip", "size_mb": 423.8},
    "China_MotorBike": {"file": "RDD2022_China_MotorBike.zip", "size_mb": 183.1},
    "China_Drone": {"file": "RDD2022_China_Drone.zip", "size_mb": 152.8},
    "Norway": {"file": "RDD2022_Norway.zip", "size_mb": 9900.0},
}

# Countries allowed by default subset (must match docs/DATASET_APPROVAL.md).
PROPOSED_SUBSET = ("Czech", "India", "Japan")


class DownloadBlocked(RuntimeError):
    """Raised when the approval gate is closed."""


def check_approval(config=None) -> None:
    """Raise :class:`DownloadBlocked` unless double approval is present."""
    if config is None:
        config = load_config(DEFAULT_CONFIG_PATH)
    if not dataset_approved(config):
        raise DownloadBlocked(
            "Dataset download is BLOCKED.\n"
            "  1. Record instructor approval in docs/DATASET_APPROVAL.md\n"
            "  2. Set dataset.approved: true in configs/experiment.yaml\n"
            "  3. Set env var DATASET_APPROVED=true\n"
            "Current state: config.dataset.approved="
            f"{config.get('dataset', {}).get('approved')} "
            "DATASET_APPROVED env="
            f"{__import__('os').environ.get('DATASET_APPROVED', '<unset>')!r}"
        )


def download_country(
    country: str, dest_root: str | Path, *, overwrite: bool = False
) -> Path:
    """Download one official country archive into ``dest_root``.

    Returns the local zip path. Streams to a ``.part`` file first so an
    interrupted download never leaves a half-written archive with the final
    name.
    """
    if country not in ARCHIVES:
        raise ValueError(
            f"Unknown country {country!r}. Available: {sorted(ARCHIVES)}"
        )
    info = ARCHIVES[country]
    url = f"{_OFFICIAL_BASE}/{info['file']}"
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / str(info["file"])
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and not overwrite:
        logger.info("Already present, skipping: %s", dest)
        return dest

    logger.info("Downloading %s (%.1f MB) from official source ...", country, info["size_mb"])
    logger.info("  URL: %s", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RoadVision-PDC/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        if part.exists():
            part.unlink()
        raise RuntimeError(f"Download failed for {country}: {exc}") from exc

    part.replace(dest)
    logger.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / (1 << 20))
    return dest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Download approved RDD2022 country subset (official source only)."
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=list(PROPOSED_SUBSET),
        choices=sorted(ARCHIVES),
        help="Country archives to fetch (default: proposed Czech India Japan).",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination root (default: dataset.root from config, usually data/raw).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--config", default=None, help="Path to experiment YAML (optional)."
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        check_approval(config)
    except DownloadBlocked as exc:
        logger.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    dest = args.dest or config["dataset"]["root"]
    total_mb = sum(float(ARCHIVES[c]["size_mb"]) for c in args.countries)
    logger.info("Approved. Requested subset: %s (%.1f MB total)", args.countries, total_mb)

    failures = 0
    for country in args.countries:
        try:
            download_country(country, dest, overwrite=args.overwrite)
        except (RuntimeError, ValueError) as exc:
            logger.error("%s", exc)
            failures += 1

    if failures:
        logger.error("%d download(s) failed.", failures)
        return 1
    logger.info("All requested archives present under %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
