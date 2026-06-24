# Public weather-support deficit diagnostic data and code

This repository contains derived data and lightweight analysis code for a public weather-information support deficit diagnostic in urban low-altitude service contexts.

## Contents

- `Source_Data/`: derived and figure-ready tables, data dictionaries, diagnostic ledgers and audit tables.
- `Reproducibility/`: environment notes and scripts for auditing the released derived tables and rebuilding figure outputs.
- `EXTERNAL_REVIEW_BRIEF.md`: a bounded review brief for independent scientific review of the released analytical package.

Raw third-party records are not redistributed because they remain subject to source-specific licensing and access terms. The released materials support verification of the derived analytical objects, figure-ready data, diagnostic ledgers and reproducibility checks. They do not support operational flight-risk prediction, hidden observing-capacity inference, route-level disruption assessment or direct three-dimensional low-altitude wind-field claims.

## Quick start

```bash
python Reproducibility/run_reproduction_audit.py
python Reproducibility/scripts/build_main_figures.py
```

The first command checks the presence and row counts of key released derived tables. The second command rebuilds figure outputs from the released derived data.

## Data policy

This release contains derived, non-raw analytical tables. It does not contain restricted raw observations, local caches, manuscript files, correspondence files, credentials or machine-local paths.

## License

Code is released under the MIT License. Derived data are released under the Creative Commons Attribution 4.0 International License.
