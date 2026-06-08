# Reproducibility archive

This clean submission archive provides figure-ready source tables, diagnostic ledgers, field dictionaries, checksums and audit scripts. It is designed for reviewer navigation and verification of the submitted figures and manuscript claims.

Run from the package root:

```bash
python Reproducibility/run_reproduction_audit.py
```

The script checks that the key manuscript- and SI-referenced source tables exist and validates selected headline values where available. Raw third-party records are not redistributed as a single archive because they remain subject to source-specific licensing and access terms.

Final figure rendering uses the figure-ready tables in `Source_Data/` and
`Source_Data/Figure_Tables/`:

```bash
python Reproducibility/scripts/build_v44_top_journal_main_figures.py
```

The rendering script rebuilds main Figures 1-4 and Supplementary Figures 6, 7
and 9 from packaged derived/source tables only. It does not require raw
third-party data.
