# Urban airspace weather-support data and code

This repository contains derived analytical tables and reproducibility code for a global diagnostic of public hourly weather-information support gaps for low-altitude services. The materials support reproduction from the released analytical layer, including the canonical 2024 component screen, country-context sensitivities, definition-stability checks, selected-case precipitation consistency checks and figure-ready source data.

## Contents

- `Source_Data/`: derived analytical tables, figure-ready sources, data dictionaries, provenance boundaries and audit tables.
- `Reproducibility/`: Python scripts for rebuilding the canonical 2024 analysis, main figures, selected supplementary figures and release audit checks.

Raw third-party records are not redistributed because they remain subject to source-specific licensing and access terms. The release does not contain publication-facing narrative files, editorial correspondence, credentials, local caches or machine-local paths.

## Quick start

Run from the repository root:

```bash
python Reproducibility/run_canonical_2024_analysis.py
python Reproducibility/build_weather_support_figures.py
python Reproducibility/build_weather_support_supplementary_figures.py
python Reproducibility/run_reproduction_audit.py
```

The released layer reproduces standardisation, equal-weight need aggregation, residual sign, 2024 threshold membership, model sensitivities, selected-case analysis and figures. It does not reconstruct the licensed station-hour archive, original component polygons, complete upstream station processing, constituent-proxy generation formulas or the original GDP-inclusive expected-support fit.

## Version

Release candidate: `v7.0.1`.

## License

Code is released under the MIT License. Derived data are released under the Creative Commons Attribution 4.0 International License.
