#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def count_rows(rel: str) -> int:
    with (ROOT / rel).open(newline="", encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def rows(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    required = [
        "README.md",
        ".zenodo.json",
        "checksums.sha256",
        "release_manifest.csv",
        "Source_Data/component_year_panel_full.csv",
        "Source_Data/canonical_2024_q75_components.csv",
        "Source_Data/country_within_between_model_summary_2024.csv",
        "Source_Data/multimodel_core_components_2024.csv",
        "Source_Data/screen_calibration_and_comparison_rules.csv",
        "Reproducibility/run_canonical_2024_analysis.py",
        "Reproducibility/build_weather_support_figures.py",
        "Reproducibility/build_weather_support_supplementary_figures.py",
    ]
    missing = [rel for rel in required if not (ROOT / rel).exists()]
    if missing:
        fail("missing required public-release files: " + ", ".join(missing))

    forbidden = [
        "Manuscript",
        "Cover_Letter",
        "Supplementary_Information",
        "highlights",
        "PUBLIC_RELEASE_SYNC_INSTRUCTIONS.md",
    ]
    present_forbidden = [rel for rel in forbidden if (ROOT / rel).exists()]
    if present_forbidden:
        fail("submission-facing files present in public release: " + ", ".join(present_forbidden))

    expected_counts = {
        "Source_Data/component_year_panel_full.csv": 114220,
        "Source_Data/canonical_2024_q75_components.csv": 651,
        "Source_Data/component_screen_stability_2024.csv": 11422,
        "Source_Data/multimodel_core_components_2024.csv": 441,
        "Source_Data/event_q75_aligned_true_source_panel.csv": 152,
        "Source_Data/event_q75_aligned_matched_controls.csv": 456,
        "Source_Data/event_q75_aligned_permutation_iterations.csv": 10000,
        "Source_Data/benchmark_random_null_iterations.csv": 10000,
        "Source_Data/source_data_Fig1.csv": 651,
    }
    for rel, expected in expected_counts.items():
        actual = count_rows(rel)
        print(f"{rel}: {actual} rows")
        if actual != expected:
            fail(f"{rel}: expected {expected}, got {actual}")

    if (ROOT / "Source_Data/v3_structural_summary.json").exists():
        fail("obsolete v3 structural summary remains")

    summary = json.loads((ROOT / "Source_Data/canonical_2024_summary.json").read_text(encoding="utf-8"))
    expected_summary = {
        "primary_n": 651,
        "primary_countries": 82,
        "country_fe_shared_n": 447,
        "model_core_n": 441,
        "frequency_ge_0_50": 490,
        "frequency_ge_0_80": 214,
        "strict_events": 152,
        "strict_matched_rows": 456,
    }
    for key, expected in expected_summary.items():
        if int(summary[key]) != expected:
            fail(f"summary {key}: expected {expected}, got {summary[key]}")

    matched = rows("Source_Data/event_q75_aligned_matched_controls.csv")
    event_counts = Counter(row["event_id"] for row in matched)
    if len(event_counts) != 152 or set(event_counts.values()) != {3}:
        fail("event_q75_aligned_matched_controls.csv does not contain exactly three controls per event")

    bench = rows("Source_Data/benchmark_fixed_size_comparison_2024.csv")
    if any(int(row["screen_size"]) != 651 for row in bench):
        fail("benchmark comparison is not fixed at 651 components")

    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.suffix.lower() in {".md", ".txt", ".py", ".json", ".csv"}
    )
    blocked = [
        "will be " + "deposited",
        "PREPARED" + "_NOT_EXECUTED",
        "/" + "Users/",
        "/" + "Volumes/",
        "api_" + "key",
        "pass" + "word",
    ]
    hits = [marker for marker in blocked if marker.lower() in public_text.lower()]
    if hits:
        fail("blocked public-release text markers: " + ", ".join(hits))

    print("Public release derived tables, scripts, boundaries and sanitization checks verified.")
    print("FINAL_VERDICT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
