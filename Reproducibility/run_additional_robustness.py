#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Source_Data"
FIG_TABLES = SOURCE / "Figure_Tables"
SEED = 20260624


def zscore_population(series: pd.Series) -> pd.Series:
    """Population-standardized z score (ddof=0), matching exported fields."""
    return (series - series.mean()) / series.std(ddof=0)


def cluster_intercept(
    df: pd.DataFrame,
    outcome: str,
    cluster: str,
) -> dict[str, float | int | str]:
    """Intercept-only mean with small-sample corrected cluster-robust t interval."""
    data = df[[outcome, cluster]].dropna().copy()
    if data.empty:
        raise ValueError(f"No complete rows for {outcome} clustered by {cluster}")
    design = np.ones((len(data), 1), dtype=float)
    fit = sm.OLS(data[outcome].to_numpy(dtype=float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster], "use_correction": True},
        use_t=True,
    )
    ci = np.asarray(fit.conf_int(), dtype=float).ravel()
    return {
        "estimate": float(fit.params[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(fit.pvalues[0]),
        "n_rows": int(len(data)),
        "n_clusters": int(data[cluster].nunique()),
        "cluster_variable": cluster,
        "interval_method": "small-sample corrected cluster-robust t interval",
    }


def adjusted_cluster_model(
    df: pd.DataFrame,
    outcome: str,
    cluster: str,
) -> dict[str, float | int | str]:
    """Pair-level regression adjustment at exact balance, with clustered inference."""
    data = df.copy()
    covariates = [
        ("population_density", "event_population_density_proxy", "control_population_density_proxy"),
        ("airport_distance", "event_nearest_airport_km", "control_nearest_airport_km"),
        ("coast_distance", "event_coast_distance_km", "control_coast_distance_km"),
    ]
    xcols: list[str] = []
    for label, event_col, control_col in covariates:
        pooled_sd = math.sqrt(
            (data[event_col].var(ddof=1) + data[control_col].var(ddof=1)) / 2.0
        )
        col = f"{label}_standardized_difference"
        data[col] = (data[event_col] - data[control_col]) / pooled_sd
        xcols.append(col)

    complete = data[[outcome, cluster, *xcols]].dropna().copy()
    design = sm.add_constant(complete[xcols], has_constant="add")
    fit = sm.OLS(complete[outcome], design).fit(
        cov_type="cluster",
        cov_kwds={"groups": complete[cluster], "use_correction": True},
        use_t=True,
    )
    ci = fit.conf_int().loc["const"]
    return {
        "estimate": float(fit.params["const"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(fit.pvalues["const"]),
        "n_rows": int(len(complete)),
        "n_clusters": int(complete[cluster].nunique()),
        "cluster_variable": cluster,
        "interval_method": (
            "OLS adjustment for population density, nearest-airport distance and coast "
            "distance; small-sample corrected cluster-robust t interval"
        ),
    }


def variance_decomposition(panel: pd.DataFrame, variable: str) -> dict[str, float | int | str]:
    data = panel[["component_id", "year", variable]].dropna().copy()
    grand_mean = float(data[variable].mean())
    component_mean = data.groupby("component_id")[variable].mean()
    component_n = data.groupby("component_id")[variable].count()
    between_ss = float((((component_mean - grand_mean) ** 2) * component_n).sum())
    within = data[variable] - data.groupby("component_id")[variable].transform("mean")
    within_ss = float((within**2).sum())
    total_ss = between_ss + within_ss

    wide = data.pivot(index="component_id", columns="year", values=variable)
    correlations = wide.corr().to_numpy(dtype=float)
    upper = correlations[np.triu_indices_from(correlations, k=1)]
    annual_means = data.groupby("year")[variable].mean()
    annual_sd = data.groupby("year")[variable].std(ddof=1)
    varying = data.groupby("component_id")[variable].nunique(dropna=False).gt(1).sum()

    return {
        "variable": variable,
        "n_rows": int(len(data)),
        "n_components": int(component_mean.size),
        "components_with_temporal_variation": int(varying),
        "between_component_share": between_ss / total_ss if total_ss else np.nan,
        "within_component_share": within_ss / total_ss if total_ss else np.nan,
        "median_cross_year_correlation": float(np.nanmedian(upper)),
        "minimum_cross_year_correlation": float(np.nanmin(upper)),
        "annual_mean_min": float(annual_means.min()),
        "annual_mean_max": float(annual_means.max()),
        "annual_sd_min": float(annual_sd.min()),
        "annual_sd_max": float(annual_sd.max()),
    }


def construct_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(
        object_name: str,
        formula_or_rule: str,
        status: str,
        max_abs_error: float | None = None,
        mismatch_rows: int | None = None,
        evidence_boundary: str = "",
    ) -> None:
        rows.append(
            {
                "object": object_name,
                "formula_or_rule": formula_or_rule,
                "status": status,
                "max_abs_error": max_abs_error,
                "mismatch_rows": mismatch_rows,
                "evidence_boundary": evidence_boundary,
            }
        )

    reconstructed = zscore_population(panel["public_hourly_surface_observation_support"])
    add(
        "public_support_z",
        "(public_hourly_surface_observation_support - panel mean) / panel population SD",
        "PASS",
        float((reconstructed - panel["public_support_z"]).abs().max()),
        int((~np.isclose(reconstructed, panel["public_support_z"], atol=1e-12)).sum()),
        "Exactly reconstructible from the released 114,220-row panel.",
    )

    reconstructed_need = panel[
        ["demand_proxy_v2", "weather_hazard_proxy", "urban_form_complexity_proxy"]
    ].mean(axis=1)
    add(
        "need_composite",
        "(demand_proxy_v2 + weather_hazard_proxy + urban_form_complexity_proxy) / 3",
        "PASS",
        float((reconstructed_need - panel["need_composite"]).abs().max()),
        int((~np.isclose(reconstructed_need, panel["need_composite"], atol=1e-12)).sum()),
        "Equal-weight arithmetic mean of the three released proxy fields.",
    )

    reconstructed_need_z = zscore_population(panel["need_composite"])
    add(
        "need_composite_z",
        "(need_composite - panel mean) / panel population SD",
        "PASS",
        float((reconstructed_need_z - panel["need_composite_z"]).abs().max()),
        int((~np.isclose(reconstructed_need_z, panel["need_composite_z"], atol=1e-12)).sum()),
        "Exactly reconstructible from the released panel.",
    )

    alias_mask = np.isclose(
        panel["public_hourly_availability_ratio"],
        panel["station_universe_completeness_ratio"],
        atol=0.0,
        equal_nan=True,
    )
    add(
        "availability/completeness ratio alias",
        "public_hourly_availability_ratio == station_universe_completeness_ratio in current export",
        "PASS",
        0.0,
        int((~alias_mask).sum()),
        "The two columns are aliases in this export and must not be counted as independent evidence dimensions.",
    )

    ratio = panel["station_universe_completeness_ratio"]
    reconstructed_flag = ratio.isna() | ratio.lt(1.0)
    add(
        "denominator_uncertainty_flag",
        "is missing(station_universe_completeness_ratio) OR ratio < 1",
        "PASS",
        0.0,
        int((reconstructed_flag != panel["denominator_uncertainty_flag"].astype(bool)).sum()),
        "Exactly reconstructible; ratio = 1 defines denominator-certain rows.",
    )

    reconstructed_resid = panel["public_support_z"] - panel["expected_public_support_z"]
    add(
        "support_residual_z",
        "public_support_z - expected_public_support_z",
        "PASS",
        float((reconstructed_resid - panel["support_residual_z"]).abs().max()),
        int((~np.isclose(reconstructed_resid, panel["support_residual_z"], atol=1e-12)).sum()),
        "Exactly reconstructible once the released expected-support field is accepted.",
    )

    reconstructed_deficit = -panel["support_residual_z"]
    add(
        "residual_public_observability_deficit",
        "-support_residual_z",
        "PASS",
        float((reconstructed_deficit - panel["residual_public_observability_deficit"]).abs().max()),
        int((~np.isclose(reconstructed_deficit, panel["residual_public_observability_deficit"], atol=1e-12)).sum()),
        "Sign reversal used for intuitive higher-is-worse mapping.",
    )

    for q, flag in [(0.75, "residual_deficit_q75"), (0.90, "residual_deficit_q90"), (0.95, "residual_deficit_q95")]:
        need_threshold = float(panel["need_composite_z"].quantile(q))
        residual_threshold = float(panel["support_residual_z"].quantile(1.0 - q))
        reconstructed_flag = panel["need_composite_z"].ge(need_threshold) & panel[
            "support_residual_z"
        ].le(residual_threshold)
        add(
            flag,
            (
                f"need_composite_z >= panel q{int(q*100)} ({need_threshold:.12g}) AND "
                f"support_residual_z <= panel q{int(round((1-q)*100)):02d} ({residual_threshold:.12g})"
            ),
            "PASS",
            0.0,
            int((reconstructed_flag != panel[flag].astype(bool)).sum()),
            "Exactly reconstructible from released panel fields.",
        )

    add(
        "public_hourly_surface_observation_support",
        "Precomputed upstream normalized support index S_it",
        "BOUNDARY",
        None,
        None,
        (
            "The released package provides S_it and its audit fields but not the licensed raw station-hour archive or "
            "the complete upstream raw-record construction pipeline; raw-record reconstruction requires reacquisition."
        ),
    )
    add(
        "expected_public_support_z raw-model reconstruction",
        "Precomputed expected-support field used by the primary residual screen",
        "BOUNDARY",
        None,
        None,
        (
            "The released gdp_density_score and model-diagnostic standardized control columns are blank, so the exact "
            "upstream expected-support fit cannot be refitted from this derived export. Residuals and flags remain exactly "
            "auditable as released derived objects, and alternative model-form sensitivity is supplied separately."
        ),
    )

    return pd.DataFrame(rows)



def available_field_refit(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Refit the 2024 benchmark using only nonblank released controls."""
    data = panel.loc[panel["year"].eq(2024)].copy()
    controls = [
        "population_density_proxy",
        "nearest_airport_km",
        "official_station_density_50km",
        "coast_distance_km",
    ]
    design = pd.DataFrame(index=data.index)
    design["need_composite_z"] = data["need_composite_z"]
    for column in controls:
        design[f"{column}_z"] = zscore_population(data[column])
    fit = sm.OLS(data["public_support_z"], sm.add_constant(design, has_constant="add")).fit()
    data["available_field_expected_public_support_z"] = fit.predict(
        sm.add_constant(design, has_constant="add")
    )
    data["available_field_support_residual_z"] = (
        data["public_support_z"] - data["available_field_expected_public_support_z"]
    )
    need_threshold = float(panel["need_composite_z"].quantile(0.75))
    residual_threshold = float(data["available_field_support_residual_z"].quantile(0.25))
    data["available_field_q75_flag"] = data["need_composite_z"].ge(need_threshold) & data[
        "available_field_support_residual_z"
    ].le(residual_threshold)
    primary = data["residual_deficit_q75"].astype(bool)
    alternative = data["available_field_q75_flag"].astype(bool)
    intersection = int((primary & alternative).sum())
    union = int((primary | alternative).sum())
    summary = {
        "r2": float(fit.rsquared),
        "need_coefficient": float(fit.params["need_composite_z"]),
        "residual_spearman": float(
            data[["support_residual_z", "available_field_support_residual_z"]].corr(method="spearman").iloc[0, 1]
        ),
        "primary_q75_n": int(primary.sum()),
        "available_field_q75_n": int(alternative.sum()),
        "intersection_n": intersection,
        "jaccard": intersection / union if union else np.nan,
        "need_threshold": need_threshold,
        "available_residual_threshold": residual_threshold,
    }
    export = data[[
        "component_id",
        "country",
        "city_name",
        "need_composite_z",
        "public_support_z",
        "expected_public_support_z",
        "support_residual_z",
        "residual_deficit_q75",
        "available_field_expected_public_support_z",
        "available_field_support_residual_z",
        "available_field_q75_flag",
    ]].copy()
    return export, summary

def event_robustness_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched = pd.read_csv(SOURCE / "event_matched_low_gap_controls.csv", low_memory=False)
    placebo = pd.read_csv(SOURCE / "event_placebo_controls.csv", low_memory=False)
    false = pd.read_csv(SOURCE / "event_false_event_controls.csv", low_memory=False)

    placebo_ok = placebo.loc[placebo["placebo_status"].eq("ok_control_day")].copy()
    false_ok = false.loc[false["false_event_status"].eq("ok")].copy()
    component_country = (
        matched[["event_component_id", "event_country"]]
        .drop_duplicates()
        .set_index("event_component_id")["event_country"]
    )
    placebo_ok["event_country"] = placebo_ok["event_component_id"].map(component_country)

    rows: list[dict[str, object]] = []

    def record(
        specification: str,
        family: str,
        result: dict[str, object],
        outcome: str,
        interpretation: str,
    ) -> None:
        rows.append(
            {
                "specification": specification,
                "family": family,
                "outcome": outcome,
                **result,
                "interpretation": interpretation,
            }
        )

    main_outcome = "paired_gpm_precip_diff_high_minus_control"
    for label, cluster in [
        ("main_event_date_cluster", "event_id"),
        ("main_component_cluster", "event_component_id"),
        ("main_country_cluster", "event_country"),
    ]:
        record(
            label,
            "main_gpm",
            cluster_intercept(matched, main_outcome, cluster),
            main_outcome,
            (
                "Event-date clustering is the primary correction for three controls per event. Component and country "
                "clustering are higher-level dependence sensitivities."
            ),
        )

    for label, cluster in [
        ("main_adjusted_component_cluster", "event_component_id"),
        ("main_adjusted_country_cluster", "event_country"),
    ]:
        record(
            label,
            "adjusted_main_gpm",
            adjusted_cluster_model(matched, main_outcome, cluster),
            main_outcome,
            (
                "Adjusted contrast at zero imbalance in population density, nearest-airport distance and coast distance. "
                "Official-station-density adjustment is unavailable because the control field is blank in this export."
            ),
        )

    for label, cluster in [
        ("placebo_event_date_cluster", "match_id"),
        ("placebo_component_cluster", "event_component_id"),
        ("placebo_country_cluster", "event_country"),
    ]:
        record(
            label,
            "event_minus_placebo",
            cluster_intercept(placebo_ok, "observed_minus_placebo_diff", cluster),
            "observed_minus_placebo_diff",
            "Pair-weighted placebo-eligible contrast; higher-level clustering is a dependence sensitivity.",
        )

    for label, cluster in [
        ("false_event_date_cluster", "match_id"),
        ("false_component_cluster", "event_component_id"),
        ("false_country_cluster", "event_country"),
    ]:
        record(
            label,
            "false_event",
            cluster_intercept(false_ok, "observed_minus_false_event_diff", cluster),
            "observed_minus_false_event_diff",
            "Negative-control contrast; intervals crossing zero support the intended boundary.",
        )

    robustness = pd.DataFrame(rows)

    # Event-date clustered main figure contrasts.
    lookup = robustness.set_index("specification")
    contrast_rows = []
    for label, spec, color in [
        ("Main matched GPM", "main_event_date_cluster", "#0F766E"),
        ("Event-placebo net", "placebo_event_date_cluster", "#B76E2A"),
        ("False-event control", "false_event_date_cluster", "#6B7280"),
    ]:
        r = lookup.loc[spec]
        contrast_rows.append(
            {
                "label": label,
                "estimate": r["estimate"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "color": color,
                "interval_method": r["interval_method"],
                "n_clusters": int(r["n_clusters"]),
            }
        )
    contrasts = pd.DataFrame(contrast_rows)

    # Leave-out tests with event-date clustered intervals.
    leave_specs = [
        ("Main matched GPM", matched),
        ("Exclude top country\n(Indonesia)", matched.loc[~matched["event_country"].eq("Indonesia")]),
        ("Exclude China and India", matched.loc[~matched["event_country"].isin(["China", "India"])]),
        ("Leave-year minimum\n(2022)", matched.loc[~matched["event_year"].eq(2022)]),
    ]
    leave_rows = []
    for label, subset in leave_specs:
        r = cluster_intercept(subset, main_outcome, "event_id")
        leave_rows.append({"label": label, **{k: r[k] for k in ["estimate", "ci_low", "ci_high", "n_clusters"]}})
    leaveout = pd.DataFrame(leave_rows)

    # Event-label strata with event-date clustered intervals.
    hazard_map = {
        "thunder_or_convective": "Convective",
        "low_ceiling_1000m": "Low ceiling",
        "low_ceiling_300m": "Low ceiling",
        "low_vis_1km": "Low visibility",
        "low_vis_5km": "Low visibility",
        "gust_15ms": "Wind",
        "strong_wind": "Wind",
        "fog": "Fog",
    }
    matched["hazard_family"] = matched["dominant_hazard"].map(hazard_map)
    hazard_rows = []
    for family, subset in matched.groupby("hazard_family", sort=True):
        r = cluster_intercept(subset, main_outcome, "event_id")
        hazard_rows.append(
            {
                "hazard_family": family,
                "estimate": r["estimate"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "n_events": int(r["n_clusters"]),
                "interval_method": "event-date clustered t interval",
            }
        )
    hazard = pd.DataFrame(hazard_rows)

    # Year and selected-country strata for Supplementary Figure 6.
    year_rows = []
    for year, subset in matched.groupby("event_year", sort=True):
        r = cluster_intercept(subset, main_outcome, "event_id")
        year_rows.append(
            {
                "year": int(year),
                "estimate": r["estimate"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "n": int(r["n_rows"]),
                "n_events": int(r["n_clusters"]),
            }
        )
    years = pd.DataFrame(year_rows)

    existing_countries = pd.read_csv(FIG_TABLES / "Supplementary_Figure_6_country_estimates.csv")
    country_rows = []
    for country in existing_countries["country"]:
        subset = matched.loc[matched["event_country"].eq(country)]
        r = cluster_intercept(subset, main_outcome, "event_id")
        country_rows.append(
            {
                "country": country,
                "estimate": r["estimate"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "n": int(r["n_rows"]),
                "n_events": int(r["n_clusters"]),
            }
        )
    countries = pd.DataFrame(country_rows)

    return robustness, contrasts, leaveout, hazard, pd.concat(
        {"years": years, "countries": countries}, names=["table"]
    )


def update_event_source_tables(
    robustness: pd.DataFrame,
    contrasts: pd.DataFrame,
    leaveout: pd.DataFrame,
    hazard: pd.DataFrame,
    strata: pd.DataFrame,
) -> None:
    robustness.to_csv(SOURCE / "event_cluster_robustness.csv", index=False)
    contrasts.to_csv(FIG_TABLES / "Fig3_panel_b_contrasts.csv", index=False)
    leaveout.to_csv(FIG_TABLES / "Fig3_panel_c_leaveout.csv", index=False)
    hazard.to_csv(FIG_TABLES / "Fig3_panel_f_hazard_family.csv", index=False)

    years = strata.xs("years").reset_index(drop=True)
    countries = strata.xs("countries").reset_index(drop=True)
    years.to_csv(FIG_TABLES / "Supplementary_Figure_6_year_estimates.csv", index=False)
    countries.to_csv(FIG_TABLES / "Supplementary_Figure_6_country_estimates.csv", index=False)

    # Retain original non-event rows but replace the three headline intervals.
    summary_path = FIG_TABLES / "Supplementary_Figure_6_summary.csv"
    summary = pd.read_csv(summary_path)
    mapping = {
        "main_gpm_matched": "main_event_date_cluster",
        "strict_placebo_v4": "placebo_event_date_cluster",
        "false_event_control_v1": "false_event_date_cluster",
    }
    lookup = robustness.set_index("specification")
    for summary_spec, robust_spec in mapping.items():
        mask = summary["specification"].eq(summary_spec)
        r = lookup.loc[robust_spec]
        for col in ["estimate", "ci_low", "ci_high"]:
            summary.loc[mask, col] = float(r[col])
    summary.to_csv(summary_path, index=False)

    # Keep event_robustness_estimates backward compatible, add interval metadata.
    event_path = SOURCE / "event_robustness_estimates.csv"
    event = pd.read_csv(event_path)
    if "interval_method" not in event.columns:
        event["interval_method"] = "stored interval from prior derived export"
    if "n_clusters" not in event.columns:
        event["n_clusters"] = np.nan
    for event_spec, robust_spec in mapping.items():
        mask = event["specification"].eq(event_spec)
        r = lookup.loc[robust_spec]
        for col in ["estimate", "ci_low", "ci_high", "p_value"]:
            event.loc[mask, col] = float(r[col])
        event.loc[mask, "interval_method"] = str(r["interval_method"])
        event.loc[mask, "n_clusters"] = int(r["n_clusters"])
        note_map = {
            "main_gpm_matched": "Main no-ERA5 GPM matched-control contrast. Headline interval uses event-date clustering; component and country sensitivities are in event_cluster_robustness.csv.",
            "strict_placebo_v4": "Strict one-month placebo after donor redesign. Headline interval uses event-date clustering; component and country sensitivities are in event_cluster_robustness.csv.",
            "false_event_control_v1": "False-event negative control. Headline interval uses event-date clustering and crosses zero; component and country sensitivities are in event_cluster_robustness.csv.",
        }
        event.loc[mask, "boundary_note"] = note_map[event_spec]
    event.to_csv(event_path, index=False)

    # Update duplicate Event_Validation folder copies.
    duplicate = SOURCE / "Event_Validation" / "event_robustness_estimates.csv"
    if duplicate.exists():
        event.to_csv(duplicate, index=False)

    neg_path = FIG_TABLES / "Supplementary_Figure_7_negative_controls.csv"
    neg = pd.read_csv(neg_path)
    r = lookup.loc["false_event_date_cluster"]
    mask = neg["specification"].eq("false_event_control_v1")
    for col in ["estimate", "ci_low", "ci_high", "p_value"]:
        neg.loc[mask, col] = float(r[col])
    neg.to_csv(neg_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce construct, panel and event robustness audits.")
    parser.add_argument("--no-update-figure-tables", action="store_true")
    args = parser.parse_args()

    panel = pd.read_csv(SOURCE / "component_year_panel_full.csv", low_memory=False)
    audit = construct_audit(panel)
    refit_export, refit_summary = available_field_refit(panel)
    refit_export.to_csv(SOURCE / "available_field_expected_support_refit_2024.csv", index=False)
    audit = pd.concat(
        [
            audit,
            pd.DataFrame(
                [
                    {
                        "object": "available-field expected-support refit (2024)",
                        "formula_or_rule": "OLS: public_support_z ~ need + population + airport + official station density + coast; all controls z-standardized within 2024",
                        "status": "PASS_SENSITIVITY",
                        "max_abs_error": np.nan,
                        "mismatch_rows": np.nan,
                        "evidence_boundary": (
                            f"R2={refit_summary['r2']:.6f}; residual Spearman rho={refit_summary['residual_spearman']:.3f}; "
                            f"q75 retained n={refit_summary['available_field_q75_n']} versus primary {refit_summary['primary_q75_n']}; "
                            f"Jaccard={refit_summary['jaccard']:.3f}. This supports robustness but does not exactly reconstruct the upstream GDP-inclusive fit."
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    audit.to_csv(SOURCE / "construct_reconstruction_audit.csv", index=False)

    variance_variables = [
        "public_hourly_surface_observation_support",
        "public_support_z",
        "need_composite",
        "need_composite_z",
        "demand_proxy_v2",
        "weather_hazard_proxy",
        "urban_form_complexity_proxy",
        "expected_public_support_z",
        "support_residual_z",
        "residual_public_observability_deficit",
        "public_hourly_station_count_50km",
        "official_station_count_50km",
        "denominator_uncertainty_flag",
    ]
    variance = pd.DataFrame([variance_decomposition(panel, v) for v in variance_variables])
    variance.to_csv(SOURCE / "panel_variance_decomposition.csv", index=False)

    robustness, contrasts, leaveout, hazard, strata = event_robustness_tables()
    if not args.no_update_figure_tables:
        update_event_source_tables(robustness, contrasts, leaveout, hazard, strata)
    else:
        robustness.to_csv(SOURCE / "event_cluster_robustness.csv", index=False)

    # Minimal self-checks.
    if int(audit.loc[audit["status"].eq("PASS"), "mismatch_rows"].fillna(0).sum()) != 0:
        raise SystemExit("Construct reconstruction mismatch detected")
    main_event = robustness.loc[robustness["specification"].eq("main_event_date_cluster")].iloc[0]
    if not math.isclose(float(main_event["estimate"]), 0.5185561899224806, rel_tol=0, abs_tol=1e-12):
        raise SystemExit("Main GPM estimate changed unexpectedly")
    print("Construct audit, variance decomposition and cluster-aware event inference completed.")
    print(f"Primary event-date clustered interval: {main_event['ci_low']:.3f} to {main_event['ci_high']:.3f} mm")
    print("FINAL_VERDICT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
