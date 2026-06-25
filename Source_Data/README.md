# Source data archive — v7.0.1

This directory contains derived and figure-ready analytical objects for the public hourly weather-support diagnostic.

## Canonical diagnostic

- `component_year_panel_full.csv`: 114,220 component-year rows representing 11,422 stable components.
- `canonical_2024_screen_flags.csv`: fixed 2024 q75/q90/q95 calibration and positive residual support-gap score.
- `canonical_2024_q75_components.csv`: canonical 651-component main screen across 82 countries.
- `component_screen_stability_2024.csv`: continuous scores, canonical membership and 1,000-draw need-weight selection frequency.
- `country_within_between_model_summary_2024.csv`: pooled, within-country, between-country and country-FE need estimates.
- `screen_sensitivity_overlap_2024.csv`: available-field, within–between, country-FE and no-hazard screen overlap.
- `multimodel_core_components_2024.csv`: 441-component canonical–available-field–country-FE core.
- `benchmark_fixed_size_comparison_2024.csv`: exactly 651 components per comparator plus 10,000-draw random null.
- `screen_calibration_and_comparison_rules.csv`: unified canonical and model-comparison rules.

## Primary selected-case analysis

Files beginning `event_q75_aligned_` contain the exact 152-date canonical-q75 stream. `event_q75_aligned_cluster_robustness.csv` is the primary inferential table. The original 344-date stream is retained under `Broader_Selected_Event_Stream/` only as a sensitivity.

## Reproduction boundary

The released tables reproduce downstream standardisation, need aggregation, residual sign, threshold membership, model comparisons, selected-case diagnostics and figures. They do not contain original urban polygons, the licensed station-hour archive, complete station inclusion/deduplication code, full constituent-proxy formulas or the original GDP-inclusive expected-support fit.
