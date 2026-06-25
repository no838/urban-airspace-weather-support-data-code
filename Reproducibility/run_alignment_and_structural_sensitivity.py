#!/usr/bin/env python3
"""Compatibility entry point for the current alignment and structural checks.

The canonical 2024 diagnostic analysis is now maintained in
``run_canonical_2024_analysis.py``.  This wrapper keeps the historical command
name usable without retaining obsolete V3 constants or writing stale output
schemas.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "Source_Data" / "canonical_2024_summary.json"
QA_OUT = ROOT / "QA" / "alignment_and_structural_sensitivity_v7.json"


def main() -> int:
    script = ROOT / "Reproducibility" / "run_canonical_2024_analysis.py"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    checks = {
        "primary_n": summary["primary_n"] == 651,
        "primary_countries": summary["primary_countries"] == 82,
        "country_fe_shared_n": summary["country_fe_shared_n"] == 447,
        "frequency_ge_0_50": summary["frequency_ge_0_50"] == 490,
        "frequency_ge_0_80": summary["frequency_ge_0_80"] == 214,
        "model_core_n": summary["model_core_n"] == 441,
        "strict_events": summary["strict_events"] == 152,
        "strict_matched_rows": summary["strict_matched_rows"] == 456,
        "strict_main_estimate": math.isclose(
            summary["strict_main_estimate"], 0.4062914210526314, abs_tol=1e-12
        ),
    }
    payload = {"source_script": str(script.relative_to(ROOT)), "checks": checks, "summary": summary}
    QA_OUT.parent.mkdir(parents=True, exist_ok=True)
    QA_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not all(checks.values()):
        raise SystemExit(f"Alignment/structural sensitivity self-check failed: {checks}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("FINAL_VERDICT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
