# Reproducibility archive

This archive provides figure-ready source tables, diagnostic ledgers, field dictionaries, checksums and audit scripts for independent verification of the released analytical objects.

Run from the repository root:

```bash
python Reproducibility/run_reproduction_audit.py
```

The script checks that key derived source tables exist and validates selected row counts where available. Raw third-party records are not redistributed because they remain subject to source-specific licensing and access terms.

Figure rendering uses the figure-ready tables in `Source_Data/` and `Source_Data/Figure_Tables/`:

```bash
python Reproducibility/scripts/build_main_figures.py
```

The rendering script rebuilds main Figures 1-4 and selected supplementary figures from packaged derived/source tables only. It does not require raw third-party data.
