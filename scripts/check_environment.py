"""Print a runtime environment inventory for RoadVision-PDC.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running without editable install (adds src/ to path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadvision.benchmarks.profiler import hardware_report  # noqa: E402


def main() -> int:
    info = hardware_report(device_used="cpu")
    print("=" * 60)
    print("RoadVision-PDC environment report (runtime-collected)")
    print("=" * 60)
    for key, value in info.items():
        print(f"  {key:24s}: {value}")
    print("=" * 60)

    # Explicit CUDA expectation for this project's local environment.
    if info["cuda_available"]:
        print("NOTE: CUDA is available on this machine — record it, but the")
        print("      laptop was documented as CPU-only; verify with nvidia-smi.")
    else:
        print("CUDA not available -> local runs are CPU-only (expected).")

    # Optional dependency checks
    print("\nOptional components:")
    try:
        import pycocotools  # noqa: F401

        print("  pycocotools           : OK")
    except ImportError:
        print("  pycocotools           : MISSING (evaluation will fail)")
    try:
        from roadvision.engine.evaluate import verify_pycocotools

        result = verify_pycocotools()
        print(f"  pycocotools known-answer check: OK (AP@0.5={result['map50']:.3f})")
    except Exception as exc:  # noqa: BLE001 - report any failure verbatim
        print(f"  pycocotools known-answer check: FAILED -> {exc}")
        return 1

    out = Path("results") / "benchmarks" / "environment_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
