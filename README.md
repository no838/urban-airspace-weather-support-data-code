# Urban airspace weather-support data and code

This repository contains the derived data and lightweight analysis code supporting the manuscript on public weather-support deficits in urban low-altitude systems.

## Contents

- `Source_Data/`: derived and figure-ready tables, data dictionaries, diagnostic ledgers and audit tables.
- `Reproducibility/`: environment notes and scripts for auditing the released derived tables and rebuilding figure-ready outputs.

Raw third-party records are not redistributed because they remain subject to source-specific licensing and access terms. The released materials support verification of the derived analytical objects, figure-ready data, diagnostic ledgers and reproducibility checks used in the manuscript.

## Quick start

```bash
python Reproducibility/run_reproduction_audit.py
python Reproducibility/scripts/build_v44_top_journal_main_figures.py
```

The first command checks the presence and row counts of key released derived tables. The second command rebuilds the figure outputs from the released derived data.

## Data policy

This release contains derived, non-raw analytical tables. It does not contain restricted raw observations, local caches, manuscript submission files, cover letters, reviewer correspondence, credentials or machine-local paths.

## License

Code is released under the MIT License. Derived data are released under the Creative Commons Attribution 4.0 International License.
