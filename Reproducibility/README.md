# Reproducibility archive — v6

Run from the package root:

```bash
python Reproducibility/run_canonical_2024_analysis.py
python Reproducibility/build_weather_support_figures.py
python Reproducibility/build_weather_support_supplementary_figures.py
python Reproducibility/run_reproduction_audit.py
```

The canonical analysis script rebuilds the 2024-calibrated screen, within–between country decomposition, country-FE and available-field sensitivities, 1,000-draw selection frequencies, no-hazard ablation, equal-size benchmark null and exactly aligned selected-case tables from packaged derived objects. Fixed seeds are recorded in the scripts.

The two figure scripts rebuild Main Figures 1–4 and Supplementary Figures 1, 2, 8, 10 and 11 from packaged source tables. Supplementary Figures 3–7 and 9 are supplied as figure artifacts with their corresponding source tables and provenance records; no separate legacy figure-building script is required or claimed.

Raw third-party station-hour records, original urban polygons, constituent-proxy generation pipelines and the original GDP-inclusive expected-support fit are not distributed. Their absence is documented in the provenance tables.
