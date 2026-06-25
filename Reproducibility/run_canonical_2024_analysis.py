#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Source_Data'
FIG_TABLES = SOURCE / 'Figure_Tables'
ROBUST = SOURCE / 'Robustness_Tables_Reorganized_SI'
SEED = 20260625
N_DRAWS = 1000
N_PERMUTATIONS = 10000
CONTROLS = [
    'population_density_proxy',
    'nearest_airport_km',
    'official_station_density_50km',
    'coast_distance_km',
]


def zscore(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f'Cannot standardize {series.name}: SD={sd}')
    return (series - series.mean()) / sd


def cluster_mean(df: pd.DataFrame, outcome: str, cluster: str) -> dict[str, float | int | str]:
    data = df[[outcome, cluster]].dropna().copy()
    if data.empty:
        raise ValueError(f'No complete rows for {outcome} clustered by {cluster}')
    fit = sm.OLS(data[outcome].to_numpy(float), np.ones((len(data), 1))).fit(
        cov_type='cluster', cov_kwds={'groups': data[cluster], 'use_correction': True}, use_t=True
    )
    ci = np.asarray(fit.conf_int(), float).ravel()
    return {
        'estimate': float(fit.params[0]),
        'ci_low': float(ci[0]),
        'ci_high': float(ci[1]),
        'p_value': float(fit.pvalues[0]),
        'n_rows': int(len(data)),
        'n_clusters': int(data[cluster].nunique()),
        'cluster_variable': cluster,
        'interval_method': 'small-sample corrected cluster-robust t interval',
    }


def event_level_adjustment(matched: pd.DataFrame, cluster: str | None) -> dict[str, float | int | str]:
    event = matched.groupby('event_id', as_index=False).agg(
        event_component_id=('event_component_id', 'first'),
        event_country=('event_country', 'first'),
        event_gpm=('event_gpm_event_subset_precip_mm_est', 'first'),
        control_gpm=('control_gpm_event_subset_precip_mm_est', 'mean'),
        event_population=('event_population_density_proxy', 'first'),
        control_population=('control_population_density_proxy', 'mean'),
        event_airport=('event_nearest_airport_km', 'first'),
        control_airport=('control_nearest_airport_km', 'mean'),
        event_coast=('event_coast_distance_km', 'first'),
        control_coast=('control_coast_distance_km', 'mean'),
    )
    event['outcome'] = event['event_gpm'] - event['control_gpm']
    pairs = {
        'population': ('event_population', 'control_population', 'event_population_density_proxy', 'control_population_density_proxy'),
        'airport': ('event_airport', 'control_airport', 'event_nearest_airport_km', 'control_nearest_airport_km'),
        'coast': ('event_coast', 'control_coast', 'event_coast_distance_km', 'control_coast_distance_km'),
    }
    xcols: list[str] = []
    for label, (event_col, control_col, pair_event_col, pair_control_col) in pairs.items():
        pooled = math.sqrt((matched[pair_event_col].var(ddof=1) + matched[pair_control_col].var(ddof=1)) / 2.0)
        out = f'{label}_standardized_difference'
        event[out] = (event[event_col] - event[control_col]) / pooled
        xcols.append(out)
    keep = ['outcome', *xcols] + ([cluster] if cluster else [])
    data = event[keep].dropna().copy()
    design = sm.add_constant(data[xcols], has_constant='add')
    if cluster:
        fit = sm.OLS(data['outcome'], design).fit(
            cov_type='cluster', cov_kwds={'groups': data[cluster], 'use_correction': True}, use_t=True
        )
    else:
        fit = sm.OLS(data['outcome'], design).fit()
    ci = fit.conf_int().loc['const']
    return {
        'estimate': float(fit.params['const']),
        'ci_low': float(ci.iloc[0]),
        'ci_high': float(ci.iloc[1]),
        'p_value': float(fit.pvalues['const']),
        'n_rows': int(len(data)),
        'n_clusters': int(data[cluster].nunique()) if cluster else int(len(data)),
        'cluster_variable': cluster or 'event_id (one row per event)',
        'interval_method': (
            'event-level OLS adjustment for population density, nearest-airport distance and coast distance; '
            + ('small-sample corrected cluster-robust t interval' if cluster else 'conventional t interval')
        ),
    }


def standardized_mean_difference(event: pd.Series, control: pd.Series) -> float:
    pooled = math.sqrt((event.var(ddof=1) + control.var(ddof=1)) / 2.0)
    return float((event.mean() - control.mean()) / pooled)


def available_field_design(data: pd.DataFrame, need_col: str) -> pd.DataFrame:
    x = pd.DataFrame(index=data.index)
    x[need_col] = data[need_col]
    for col in CONTROLS:
        x[f'{col}_z'] = zscore(data[col])
    return x


def fit_ols(data: pd.DataFrame, design: pd.DataFrame):
    x = sm.add_constant(design, has_constant='add')
    fit = sm.OLS(data['public_support_z'], x).fit()
    fit_cluster = sm.OLS(data['public_support_z'], x).fit(
        cov_type='cluster', cov_kwds={'groups': data['country'], 'use_correction': True}, use_t=True
    )
    residual = data['public_support_z'] - fit.predict(x)
    return fit, fit_cluster, fit.predict(x), residual


def canonical_flag(need: pd.Series, residual: pd.Series, need_threshold: float) -> tuple[pd.Series, float]:
    residual_threshold = float(residual.quantile(0.25))
    return need.ge(need_threshold) & residual.le(residual_threshold), residual_threshold


def event_permutation(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arrays = []
    for _, subset in matched.groupby('event_id', sort=True):
        event_value = float(subset['event_gpm_event_subset_precip_mm_est'].iloc[0])
        controls = subset['control_gpm_event_subset_precip_mm_est'].to_numpy(float)
        if len(controls) != 3:
            raise ValueError('Canonical q75 permutation expects exactly three controls per event')
        arrays.append(np.concatenate([[event_value], controls]))
    values = np.vstack(arrays)
    observed_by_event = values[:, 0] - values[:, 1:].mean(axis=1)
    observed = float(observed_by_event.mean())
    rng = np.random.default_rng(SEED)
    choices = rng.integers(0, 4, size=(N_PERMUTATIONS, len(values)))
    total = values.sum(axis=1)
    selected = values[np.arange(len(values))[None, :], choices]
    permuted_event = selected - (total[None, :] - selected) / 3.0
    null = permuted_event.mean(axis=1)
    q025, q50, q975 = np.quantile(null, [0.025, 0.5, 0.975])
    p_value = (1 + np.sum(np.abs(null) >= abs(observed))) / (N_PERMUTATIONS + 1)
    summary = pd.DataFrame([{
        'observed': observed,
        'null_q025': q025,
        'null_median': q50,
        'null_q975': q975,
        'p_value_two_sided': p_value,
        'iterations': N_PERMUTATIONS,
        'n_events': len(values),
        'permutation_unit': 'within event date: choose one of event plus three controls as pseudo-event',
        'random_seed': SEED,
    }])
    iterations = pd.DataFrame({'iteration': np.arange(N_PERMUTATIONS), 'null_mean_contrast_mm': null})
    return summary, iterations


def build_canonical_screen(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    data = panel.loc[panel['year'].eq(2024)].copy().reset_index(drop=True)
    thresholds: dict[str, float] = {}
    flags = panel[['component_id', 'year', 'need_composite_z', 'support_residual_z']].copy()
    flags['residual_public_support_gap_z'] = -flags['support_residual_z']
    for label, q_need, q_resid in [('q75', 0.75, 0.25), ('q90', 0.90, 0.10), ('q95', 0.95, 0.05)]:
        need_thr = float(data['need_composite_z'].quantile(q_need))
        residual_thr = float(data['support_residual_z'].quantile(q_resid))
        thresholds[f'{label}_need_threshold_z'] = need_thr
        thresholds[f'{label}_residual_threshold_z'] = residual_thr
        flags[f'canonical_2024_{label}'] = flags['need_composite_z'].ge(need_thr) & flags['support_residual_z'].le(residual_thr)
    flags.to_csv(SOURCE / 'canonical_2024_screen_flags.csv', index=False)

    merged = panel.merge(
        flags[['component_id', 'year', 'residual_public_support_gap_z', 'canonical_2024_q75', 'canonical_2024_q90', 'canonical_2024_q95']],
        on=['component_id', 'year'], how='left', validate='one_to_one'
    )
    d2024 = merged.loc[merged['year'].eq(2024)].copy().reset_index(drop=True)
    primary = d2024['canonical_2024_q75'].astype(bool)
    primary_n = int(primary.sum())
    primary_countries = int(d2024.loc[primary, 'country'].nunique())
    counts = d2024.loc[primary].groupby('country', as_index=False).size().rename(columns={'size': 'retained_components'})
    counts = counts.sort_values(['retained_components', 'country'], ascending=[False, True]).reset_index(drop=True)
    counts['rank'] = np.arange(1, len(counts) + 1)
    counts['share'] = counts['retained_components'] / primary_n
    counts['cumulative_share'] = counts['share'].cumsum()
    counts.to_csv(FIG_TABLES / 'Fig1_panel_b_country_counts.csv', index=False)
    counts[['country', 'retained_components', 'rank', 'cumulative_share']].to_csv(
        FIG_TABLES / 'Fig1_panel_c_country_concentration.csv', index=False
    )
    map_cols = [
        'component_id', 'year', 'country', 'city_name', 'centroid_lat', 'centroid_lon',
        'public_hourly_surface_observation_support', 'denominator_uncertainty_flag',
        'official_station_count_50km', 'public_hourly_station_count_50km',
        'need_composite_z', 'support_residual_z', 'residual_public_support_gap_z',
        'canonical_2024_q75', 'canonical_2024_q90', 'canonical_2024_q95',
    ]
    d2024.loc[primary, map_cols].sort_values('residual_public_support_gap_z', ascending=False).to_csv(
        SOURCE / 'figure_ready_source_data_Fig1_global_map.csv', index=False
    )
    d2024.loc[primary, map_cols].to_csv(SOURCE / 'canonical_2024_q75_components.csv', index=False)

    # Figure 2 panel A: 2024 need deciles.
    d2024['need_decile'] = pd.qcut(d2024['need_composite_z'], 10, labels=False, duplicates='drop') + 1
    dec = d2024.groupby('need_decile', as_index=False).agg(
        observed_median=('public_support_z', 'median'),
        observed_p10=('public_support_z', lambda s: s.quantile(0.10)),
        observed_p25=('public_support_z', lambda s: s.quantile(0.25)),
        expected_median=('expected_public_support_z', 'median'),
        n=('component_id', 'size'),
    )
    dec.to_csv(FIG_TABLES / 'Fig2_panel_a_need_support_deciles.csv', index=False)
    d2024[['component_id', 'residual_public_support_gap_z', 'canonical_2024_q75']].to_csv(
        FIG_TABLES / 'Fig2_panel_b_residual_distribution_summary.csv', index=False
    )
    d2024[['component_id', 'need_composite_z', 'residual_public_support_gap_z', 'canonical_2024_q75']].to_csv(
        FIG_TABLES / 'Fig2_panel_c_decision_plane_sample.csv', index=False
    )
    annual_rows = []
    for year, subset in merged.groupby('year', sort=True):
        annual_rows.append({
            'year': int(year), 'rows': int(len(subset)),
            'q75': int(subset['canonical_2024_q75'].sum()),
            'q90': int(subset['canonical_2024_q90'].sum()),
            'q95': int(subset['canonical_2024_q95'].sum()),
            'q75_share': float(subset['canonical_2024_q75'].mean()),
            'q90_share': float(subset['canonical_2024_q90'].mean()),
            'q95_share': float(subset['canonical_2024_q95'].mean()),
        })
    pd.DataFrame(annual_rows).to_csv(FIG_TABLES / 'Fig2_panel_d_annual_threshold_stability.csv', index=False)

    china_india = int(counts.loc[counts['country'].isin(['China', 'India']), 'retained_components'].sum())
    summary = {
        **thresholds,
        'primary_n': primary_n,
        'primary_countries': primary_countries,
        'china_india_n': china_india,
        'china_india_share': china_india / primary_n,
        'outside_china_india_share': 1 - china_india / primary_n,
    }
    return merged, summary


def build_structural_outputs(panel: pd.DataFrame, screen_summary: dict[str, float | int]) -> dict[str, float | int]:
    data = panel.loc[panel['year'].eq(2024)].copy().reset_index(drop=True)
    primary = data['canonical_2024_q75'].astype(bool)
    primary_array = primary.to_numpy()
    primary_n = int(primary.sum())
    primary_ids = set(data.loc[primary, 'component_id'])
    need_threshold = float(screen_summary['q75_need_threshold_z'])

    # Pooled available-field model.
    pooled_design = available_field_design(data, 'need_composite_z')
    pooled, pooled_cluster, pooled_fitted, pooled_residual = fit_ols(data, pooled_design)
    pooled_flag, pooled_residual_threshold = canonical_flag(data['need_composite_z'], pooled_residual, need_threshold)

    # Country fixed effects.
    dummies = pd.get_dummies(data['country'], prefix='country', drop_first=True, dtype=float)
    fe_design = pd.concat([pooled_design, dummies], axis=1)
    fe, fe_cluster, fe_fitted, fe_residual = fit_ols(data, fe_design)
    fe_flag, fe_residual_threshold = canonical_flag(data['need_composite_z'], fe_residual, need_threshold)

    # Mundlak within-between decomposition of need and all available controls.
    wb_design = pd.DataFrame(index=data.index)
    for col in ['need_composite_z', *CONTROLS]:
        s = data[col] if col == 'need_composite_z' else zscore(data[col])
        country_mean = s.groupby(data['country']).transform('mean')
        key = col if col == 'need_composite_z' else f'{col}_z'
        wb_design[f'{key}_within'] = s - country_mean
        wb_design[f'{key}_between'] = country_mean
    wb, wb_cluster, wb_fitted, wb_residual = fit_ols(data, wb_design)
    wb_flag, wb_residual_threshold = canonical_flag(data['need_composite_z'], wb_residual, need_threshold)

    # No-hazard ablation, calibrated only on 2024.
    nohaz_need = zscore(data[['demand_proxy_v2', 'urban_form_complexity_proxy']].mean(axis=1).rename('need_no_hazard_z'))
    nohaz_data = data.copy()
    nohaz_data['need_no_hazard_z'] = nohaz_need
    nohaz_design = available_field_design(nohaz_data, 'need_no_hazard_z')
    nohaz, nohaz_cluster, nohaz_fitted, nohaz_residual = fit_ols(nohaz_data, nohaz_design)
    nohaz_need_threshold = float(nohaz_need.quantile(0.75))
    nohaz_flag, nohaz_residual_threshold = canonical_flag(nohaz_need, nohaz_residual, nohaz_need_threshold)

    # 1,000 stored Dirichlet weights, each calibrated and refitted within 2024.
    draw_table = pd.read_csv(ROBUST / 'Supplementary_Table_R1_dirichlet_need_weight_sensitivity_draws.csv')
    weights = draw_table[['w_demand', 'w_hazard', 'w_urban_form']].to_numpy(float)
    if len(weights) != N_DRAWS:
        raise ValueError(f'Expected {N_DRAWS} weights, found {len(weights)}')
    proxy_matrix = data[['demand_proxy_v2', 'weather_hazard_proxy', 'urban_form_complexity_proxy']].to_numpy(float)
    control_matrix = np.column_stack([zscore(data[col]).to_numpy(float) for col in CONTROLS])
    y = data['public_support_z'].to_numpy(float)
    selection_count = np.zeros(len(data), dtype=int)
    draw_rows = []
    for draw, weight in enumerate(weights):
        need = proxy_matrix @ weight
        need = (need - need.mean()) / need.std(ddof=0)
        design = np.column_stack([np.ones(len(data)), need, control_matrix])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ beta
        flag = (need >= np.quantile(need, 0.75)) & (residual <= np.quantile(residual, 0.25))
        selection_count += flag
        intersection = int(np.sum(flag & primary_array))
        union = int(np.sum(flag | primary_array))
        draw_rows.append({
            'draw': draw,
            'retained_2024': int(flag.sum()),
            'countries_2024': int(data.loc[flag, 'country'].nunique()),
            'overlap_with_primary_count_2024': intersection,
            'jaccard_vs_primary_q75_2024': intersection / union if union else np.nan,
            'w_demand': float(weight[0]), 'w_hazard': float(weight[1]), 'w_urban_form': float(weight[2]),
        })
    frequency = selection_count / N_DRAWS
    pd.DataFrame(draw_rows).to_csv(SOURCE / 'dirichlet_2024_refit_draw_summary.csv', index=False)

    # Equal-size benchmark comparison and random null.
    benchmark = pd.read_csv(SOURCE / 'benchmark_screen_rankings.csv', low_memory=False)
    if len(benchmark) != len(data):
        raise ValueError('Benchmark and 2024 component tables differ in row count')
    benchmark_specs = [
        ('Population', 'population_rank'),
        ('Airport', 'airport_rank'),
        ('Station density', 'official_station_density_rank'),
        ('Climate', 'climate_rank'),
        ('Legacy demand', 'legacy_demand_rank'),
        ('Coastal demand', 'coastal_demand_rank'),
    ]
    rng = np.random.default_rng(SEED)
    random_overlap = rng.hypergeometric(primary_n, len(data) - primary_n, primary_n, size=N_PERMUTATIONS)
    random_q025, random_median, random_q975 = np.quantile(random_overlap, [0.025, 0.5, 0.975])
    benchmark_rows = []
    for name, rank_col in benchmark_specs:
        selected = benchmark.sort_values([rank_col, 'component_id'], ascending=[True, True]).head(primary_n)
        selected_ids = set(selected['component_id'])
        intersection = len(primary_ids & selected_ids)
        union = len(primary_ids | selected_ids)
        benchmark_rows.append({
            'benchmark': name,
            'screen_size': primary_n,
            'intersection': intersection,
            'overlap_with_q75_screen_pct': 100 * intersection / primary_n,
            'jaccard': intersection / union,
            'precision': intersection / primary_n,
            'recall': intersection / primary_n,
            'random_overlap_mean_pct': 100 * random_overlap.mean() / primary_n,
            'random_overlap_q025_pct': 100 * random_q025 / primary_n,
            'random_overlap_median_pct': 100 * random_median / primary_n,
            'random_overlap_q975_pct': 100 * random_q975 / primary_n,
            'tie_rule': f'rank ascending, component_id ascending; exactly {primary_n} selected',
        })
    benchmark_fixed = pd.DataFrame(benchmark_rows)
    benchmark_fixed.to_csv(SOURCE / 'benchmark_fixed_size_comparison_2024.csv', index=False)
    benchmark_fixed.to_csv(FIG_TABLES / 'Fig4_panel_a_benchmark_overlap_fixed_size.csv', index=False)
    pd.DataFrame({'iteration': np.arange(N_PERMUTATIONS), 'random_intersection': random_overlap}).to_csv(
        SOURCE / 'benchmark_random_null_iterations.csv', index=False
    )

    # Model coefficients and overlap summaries.
    def clustered_param(fit, name: str) -> tuple[float, float, float]:
        ci = fit.conf_int().loc[name]
        return float(fit.params[name]), float(ci.iloc[0]), float(ci.iloc[1])

    pcoef, plo, phi = clustered_param(pooled_cluster, 'need_composite_z')
    wcoef, wlo, whi = clustered_param(wb_cluster, 'need_composite_z_within')
    bcoef, blo, bhi = clustered_param(wb_cluster, 'need_composite_z_between')
    fecoef, felo, fehi = clustered_param(fe_cluster, 'need_composite_z')
    model_rows = [
        {'model': 'Pooled controls', 'coefficient_role': 'pooled need slope', 'estimate': pcoef, 'ci_low': plo, 'ci_high': phi, 'r2': pooled.rsquared, 'n': len(data)},
        {'model': 'Within-between', 'coefficient_role': 'within-country need slope', 'estimate': wcoef, 'ci_low': wlo, 'ci_high': whi, 'r2': wb.rsquared, 'n': len(data)},
        {'model': 'Within-between', 'coefficient_role': 'between-country need slope', 'estimate': bcoef, 'ci_low': blo, 'ci_high': bhi, 'r2': wb.rsquared, 'n': len(data)},
        {'model': 'Country fixed effects', 'coefficient_role': 'country-FE need slope', 'estimate': fecoef, 'ci_low': felo, 'ci_high': fehi, 'r2': fe.rsquared, 'n': len(data)},
    ]
    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(SOURCE / 'country_within_between_model_summary_2024.csv', index=False)
    model_summary.to_csv(FIG_TABLES / 'Fig4_panel_b_mundlak_coefficients.csv', index=False)

    def overlap_row(label: str, flag: pd.Series) -> dict[str, float | int | str]:
        arr = flag.astype(bool).to_numpy()
        intersection = int(np.sum(arr & primary_array))
        union = int(np.sum(arr | primary_array))
        return {
            'sensitivity': label,
            'retained_2024': int(arr.sum()),
            'shared_with_primary': intersection,
            'jaccard_with_primary': intersection / union,
            'share_of_primary_recovered': intersection / primary_n,
            'countries_2024': int(data.loc[arr, 'country'].nunique()),
        }

    sensitivity = pd.DataFrame([
        overlap_row('Available-field refit', pooled_flag),
        overlap_row('Within-between model', wb_flag),
        overlap_row('Country fixed effects', fe_flag),
        overlap_row('No-hazard need', nohaz_flag),
    ])
    sensitivity.to_csv(SOURCE / 'screen_sensitivity_overlap_2024.csv', index=False)
    sensitivity.to_csv(FIG_TABLES / 'Fig4_panel_c_sensitivity_overlap.csv', index=False)

    # Component-level diagnostic output and model core.
    component = data[[
        'component_id', 'country', 'city_name', 'centroid_lat', 'centroid_lon',
        'need_composite_z', 'support_residual_z', 'residual_public_support_gap_z',
        'canonical_2024_q75', 'denominator_uncertainty_flag',
    ]].copy()
    component['available_field_expected_support_z'] = pooled_fitted.to_numpy()
    component['available_field_support_residual_z'] = pooled_residual.to_numpy()
    component['available_field_q75'] = pooled_flag.to_numpy()
    component['within_between_expected_support_z'] = wb_fitted.to_numpy()
    component['within_between_support_residual_z'] = wb_residual.to_numpy()
    component['within_between_q75'] = wb_flag.to_numpy()
    component['country_fe_expected_support_z'] = fe_fitted.to_numpy()
    component['country_fe_support_residual_z'] = fe_residual.to_numpy()
    component['country_fe_q75'] = fe_flag.to_numpy()
    component['need_no_hazard_z'] = nohaz_need.to_numpy()
    component['no_hazard_support_residual_z'] = nohaz_residual.to_numpy()
    component['no_hazard_q75'] = nohaz_flag.to_numpy()
    component['dirichlet_selection_count'] = selection_count
    component['dirichlet_selection_frequency'] = frequency
    component['weight_stability_class'] = pd.cut(
        frequency, bins=[-1e-12, 0.20, 0.50, 0.80, 1.0000001],
        labels=['<0.20', '0.20-<0.50', '0.50-<0.80', '>=0.80'], right=False
    ).astype(str)
    component['model_core_primary_available_country_fe'] = (
        primary_array & pooled_flag.to_numpy() & fe_flag.to_numpy()
    )
    component['graded_core_primary_available_within_between_frequency'] = (
        primary_array & pooled_flag.to_numpy() & wb_flag.to_numpy() & (frequency >= 0.50)
    )
    component.to_csv(SOURCE / 'component_screen_stability_2024.csv', index=False)
    component.to_csv(FIG_TABLES / 'Fig4_panel_d_selection_frequency.csv', index=False)
    component.loc[component['model_core_primary_available_country_fe']].to_csv(
        SOURCE / 'multimodel_core_components_2024.csv', index=False
    )

    freq_summary = pd.DataFrame([{
        'population': 'primary canonical 2024 q75 components',
        'n_components': primary_n,
        'frequency_median': float(np.median(frequency[primary_array])),
        'frequency_p05': float(np.quantile(frequency[primary_array], 0.05)),
        'frequency_p95': float(np.quantile(frequency[primary_array], 0.95)),
        'n_frequency_ge_0_50': int(np.sum(primary_array & (frequency >= 0.50))),
        'n_frequency_ge_0_80': int(np.sum(primary_array & (frequency >= 0.80))),
        'n_frequency_lt_0_20': int(np.sum(primary_array & (frequency < 0.20))),
        'draws': N_DRAWS,
        'model_core_n': int(component['model_core_primary_available_country_fe'].sum()),
        'graded_core_n': int(component['graded_core_primary_available_within_between_frequency'].sum()),
        'definition': 'selection frequency across stored Dirichlet weights with draw-specific 2024 need q75 and model residual q25',
    }])
    freq_summary.to_csv(SOURCE / 'selection_frequency_summary_2024.csv', index=False)

    # Canonical comparison rules for Supplementary Table 16.
    rules = pd.DataFrame([
        {'record_type': 'primary calibration', 'branch_or_screen': 'q75', 'need_quantile_or_rule': 0.75, 'need_threshold_z': screen_summary['q75_need_threshold_z'], 'residual_quantile_or_rule': 0.25, 'residual_threshold_z': screen_summary['q75_residual_threshold_z'], 'reference_year': 2024, 'interpretation': 'fixed 2024 thresholds applied to each annual component'},
        {'record_type': 'primary calibration', 'branch_or_screen': 'q90', 'need_quantile_or_rule': 0.90, 'need_threshold_z': screen_summary['q90_need_threshold_z'], 'residual_quantile_or_rule': 0.10, 'residual_threshold_z': screen_summary['q90_residual_threshold_z'], 'reference_year': 2024, 'interpretation': 'fixed 2024 thresholds applied to each annual component'},
        {'record_type': 'primary calibration', 'branch_or_screen': 'q95', 'need_quantile_or_rule': 0.95, 'need_threshold_z': screen_summary['q95_need_threshold_z'], 'residual_quantile_or_rule': 0.05, 'residual_threshold_z': screen_summary['q95_residual_threshold_z'], 'reference_year': 2024, 'interpretation': 'fixed 2024 thresholds applied to each annual component'},
        {'record_type': 'comparison rule', 'branch_or_screen': 'Primary q75', 'need_quantile_or_rule': 'need_composite_z >= 2024 q75', 'need_threshold_z': np.nan, 'residual_quantile_or_rule': 'primary support_residual_z <= 2024 q25', 'residual_threshold_z': np.nan, 'reference_year': 2024, 'interpretation': 'canonical screen; fixed thresholds applied to annual audits'},
        {'record_type': 'comparison rule', 'branch_or_screen': 'Available-field pooled', 'need_quantile_or_rule': 'same 2024 q75 need convention', 'need_threshold_z': np.nan, 'residual_quantile_or_rule': 'model-specific 2024 residual q25', 'residual_threshold_z': pooled_residual_threshold, 'reference_year': 2024, 'interpretation': 'fully reproducible residual-model sensitivity'},
        {'record_type': 'comparison rule', 'branch_or_screen': 'Country fixed effects', 'need_quantile_or_rule': 'same 2024 q75 need convention', 'need_threshold_z': np.nan, 'residual_quantile_or_rule': 'model-specific 2024 residual q25', 'residual_threshold_z': fe_residual_threshold, 'reference_year': 2024, 'interpretation': 'country-intercept sensitivity'},
        {'record_type': 'comparison rule', 'branch_or_screen': 'Within-between', 'need_quantile_or_rule': 'same 2024 q75 need convention', 'need_threshold_z': np.nan, 'residual_quantile_or_rule': 'model-specific 2024 residual q25', 'residual_threshold_z': wb_residual_threshold, 'reference_year': 2024, 'interpretation': 'within- and between-country decomposition'},
        {'record_type': 'comparison rule', 'branch_or_screen': 'No-hazard need', 'need_quantile_or_rule': '2024 q75 of two-proxy no-hazard composite', 'need_threshold_z': nohaz_need_threshold, 'residual_quantile_or_rule': 'model-specific 2024 residual q25', 'residual_threshold_z': nohaz_residual_threshold, 'reference_year': 2024, 'interpretation': 'weather-hazard-proxy ablation'},
        {'record_type': 'comparison rule', 'branch_or_screen': 'Dirichlet weights', 'need_quantile_or_rule': 'draw-specific 2024 weighted-need q75', 'need_threshold_z': np.nan, 'residual_quantile_or_rule': 'draw-specific 2024 model residual q25', 'residual_threshold_z': np.nan, 'reference_year': 2024, 'interpretation': 'selection-frequency analysis'},
    ])
    rules.to_csv(SOURCE / 'screen_calibration_and_comparison_rules.csv', index=False)

    # Updated reconstruction audit: legacy flags are retained but not main-screen fields.
    old_audit = pd.read_csv(SOURCE / 'construct_reconstruction_audit.csv')
    old_audit.loc[old_audit['object'].eq('residual_deficit_q75'), 'object'] = 'legacy_panel_calibrated_residual_deficit_q75'
    old_audit.loc[old_audit['object'].eq('residual_deficit_q90'), 'object'] = 'legacy_panel_calibrated_residual_deficit_q90'
    old_audit.loc[old_audit['object'].eq('residual_deficit_q95'), 'object'] = 'legacy_panel_calibrated_residual_deficit_q95'
    old_audit.loc[old_audit['object'].str.startswith('legacy_panel_calibrated_', na=False), 'evidence_boundary'] = (
        'Exactly reconstructible legacy panel-calibrated flag retained for provenance; not the canonical 2024-calibrated main screen.'
    )
    old_audit = old_audit.loc[~old_audit['object'].eq('available-field expected-support refit (2024)')].copy()
    additions = pd.DataFrame([
        {'object': 'canonical_2024_q75', 'formula_or_rule': f"need_composite_z >= 2024 q75 ({screen_summary['q75_need_threshold_z']:.12g}) AND support_residual_z <= 2024 q25 ({screen_summary['q75_residual_threshold_z']:.12g})", 'status': 'PASS', 'max_abs_error': 0.0, 'mismatch_rows': 0, 'evidence_boundary': f"Canonical main screen; {primary_n} components in 2024 and fixed thresholds applied to annual stability audits."},
        {'object': 'canonical_2024_q90', 'formula_or_rule': f"need_composite_z >= 2024 q90 ({screen_summary['q90_need_threshold_z']:.12g}) AND support_residual_z <= 2024 q10 ({screen_summary['q90_residual_threshold_z']:.12g})", 'status': 'PASS', 'max_abs_error': 0.0, 'mismatch_rows': 0, 'evidence_boundary': 'Canonical strict sensitivity screen using fixed 2024 thresholds.'},
        {'object': 'canonical_2024_q95', 'formula_or_rule': f"need_composite_z >= 2024 q95 ({screen_summary['q95_need_threshold_z']:.12g}) AND support_residual_z <= 2024 q05 ({screen_summary['q95_residual_threshold_z']:.12g})", 'status': 'PASS', 'max_abs_error': 0.0, 'mismatch_rows': 0, 'evidence_boundary': 'Canonical strict sensitivity screen using fixed 2024 thresholds.'},
        {'object': 'available-field expected-support refit (2024 canonical)', 'formula_or_rule': 'OLS: public_support_z ~ need + population + airport + official station density + coast; controls z-standardized in 2024; same need q75 and model-specific residual q25', 'status': 'PASS_SENSITIVITY', 'max_abs_error': np.nan, 'mismatch_rows': np.nan, 'evidence_boundary': f"R2={pooled.rsquared:.6f}; retained n={int(pooled_flag.sum())}; shared with canonical primary={int((pooled_flag & primary).sum())}; Jaccard={sensitivity.loc[sensitivity.sensitivity.eq('Available-field refit'),'jaccard_with_primary'].iloc[0]:.3f}."},
        {'object': 'within-between expected-support sensitivity (2024)', 'formula_or_rule': 'Mundlak OLS with within-country deviations and country means for need and all available controls; same need q75 and model-specific residual q25', 'status': 'PASS_SENSITIVITY', 'max_abs_error': np.nan, 'mismatch_rows': np.nan, 'evidence_boundary': f"R2={wb.rsquared:.6f}; within-country need slope={wcoef:.3f}; between-country need slope={bcoef:.3f}; shared={int((wb_flag & primary).sum())}."},
    ])
    pd.concat([old_audit, additions], ignore_index=True).to_csv(SOURCE / 'construct_reconstruction_audit.csv', index=False)

    summary = {
        'pooled_r2': float(pooled.rsquared), 'pooled_need_coef': pcoef, 'pooled_need_ci_low': plo, 'pooled_need_ci_high': phi,
        'within_between_r2': float(wb.rsquared), 'within_need_coef': wcoef, 'within_need_ci_low': wlo, 'within_need_ci_high': whi,
        'between_need_coef': bcoef, 'between_need_ci_low': blo, 'between_need_ci_high': bhi,
        'country_fe_r2': float(fe.rsquared), 'country_fe_need_coef': fecoef, 'country_fe_need_ci_low': felo, 'country_fe_need_ci_high': fehi,
        'available_retained_n': int(pooled_flag.sum()), 'available_shared_n': int((pooled_flag & primary).sum()),
        'within_between_retained_n': int(wb_flag.sum()), 'within_between_shared_n': int((wb_flag & primary).sum()),
        'country_fe_retained_n': int(fe_flag.sum()), 'country_fe_shared_n': int((fe_flag & primary).sum()),
        'no_hazard_retained_n': int(nohaz_flag.sum()), 'no_hazard_shared_n': int((nohaz_flag & primary).sum()),
        'frequency_ge_0_50': int(np.sum(primary_array & (frequency >= 0.50))),
        'frequency_ge_0_80': int(np.sum(primary_array & (frequency >= 0.80))),
        'frequency_lt_0_20': int(np.sum(primary_array & (frequency < 0.20))),
        'model_core_n': int(component['model_core_primary_available_country_fe'].sum()),
        'graded_core_n': int(component['graded_core_primary_available_within_between_frequency'].sum()),
        'random_overlap_mean_pct': float(100 * random_overlap.mean() / primary_n),
        'random_overlap_q025_pct': float(100 * random_q025 / primary_n),
        'random_overlap_q975_pct': float(100 * random_q975 / primary_n),
        'fixed_size_max_overlap_pct': float(benchmark_fixed['overlap_with_q75_screen_pct'].max()),
    }
    return summary


def build_event_outputs(panel: pd.DataFrame) -> dict[str, float | int]:
    events = pd.read_csv(SOURCE / 'event_true_source_panel.csv', low_memory=False)
    matched = pd.read_csv(SOURCE / 'event_matched_low_gap_controls.csv', low_memory=False)
    placebo = pd.read_csv(SOURCE / 'event_placebo_controls.csv', low_memory=False)
    false = pd.read_csv(SOURCE / 'event_false_event_controls.csv', low_memory=False)

    aligned = events.merge(
        panel[['component_id', 'year', 'canonical_2024_q75']],
        left_on=['component_id', 'event_year'], right_on=['component_id', 'year'], how='left'
    )
    aligned['canonical_2024_q75'] = aligned['canonical_2024_q75'].fillna(False).astype(bool)
    strict_ids = set(aligned.loc[aligned['canonical_2024_q75'], 'event_id'])
    strict_events = aligned.loc[aligned['canonical_2024_q75']].copy()
    strict_matched = matched.loc[matched['event_id'].isin(strict_ids)].copy()
    strict_placebo = placebo.loc[placebo['match_id'].isin(strict_matched['match_id'])].copy()
    strict_false = false.loc[false['match_id'].isin(strict_matched['match_id'])].copy()

    event_country = strict_matched[['event_component_id', 'event_country']].drop_duplicates().set_index('event_component_id')['event_country']
    strict_placebo['event_country'] = strict_placebo['event_component_id'].map(event_country)

    strict_events.to_csv(SOURCE / 'event_q75_aligned_true_source_panel.csv', index=False)
    strict_matched.to_csv(SOURCE / 'event_q75_aligned_matched_controls.csv', index=False)
    strict_placebo.to_csv(SOURCE / 'event_q75_aligned_placebo_controls.csv', index=False)
    strict_false.to_csv(SOURCE / 'event_q75_aligned_false_event_controls.csv', index=False)

    main_outcome = 'paired_gpm_precip_diff_high_minus_control'
    rows = []
    for spec, cluster in [
        ('canonical_main_event_date_cluster', 'event_id'),
        ('canonical_main_component_cluster', 'event_component_id'),
        ('canonical_main_country_cluster', 'event_country'),
    ]:
        rows.append({'specification': spec, 'family': 'canonical_main_gpm', 'outcome': main_outcome, **cluster_mean(strict_matched, main_outcome, cluster), 'interpretation': 'Exactly canonical-q75-aligned selected-event subset; existing matched controls retained.'})
    for spec, cluster in [
        ('canonical_main_adjusted_event', None),
        ('canonical_main_adjusted_component_cluster', 'event_component_id'),
        ('canonical_main_adjusted_country_cluster', 'event_country'),
    ]:
        rows.append({'specification': spec, 'family': 'canonical_adjusted_main_gpm', 'outcome': main_outcome, **event_level_adjustment(strict_matched, cluster), 'interpretation': 'Event-level adjustment for available residual imbalance; station-density adjustment unavailable.'})

    placebo_ok = strict_placebo.loc[strict_placebo['placebo_status'].eq('ok_control_day')].copy()
    false_ok = strict_false.loc[strict_false['false_event_status'].eq('ok')].copy()
    for prefix, data, outcome, clusters in [
        ('canonical_placebo', placebo_ok, 'observed_minus_placebo_diff', [('event_date_cluster', 'match_id'), ('component_cluster', 'event_component_id'), ('country_cluster', 'event_country')]),
        ('canonical_false', false_ok, 'observed_minus_false_event_diff', [('event_date_cluster', 'match_id'), ('component_cluster', 'event_component_id'), ('country_cluster', 'event_country')]),
    ]:
        for suffix, cluster in clusters:
            rows.append({'specification': f'{prefix}_{suffix}', 'family': prefix, 'outcome': outcome, **cluster_mean(data, outcome, cluster), 'interpretation': 'Canonical-q75-aligned eligible subset; higher-level clustering is a dependence sensitivity.'})
    robust = pd.DataFrame(rows)
    robust.to_csv(SOURCE / 'event_q75_aligned_cluster_robustness.csv', index=False)

    # Main Figure 3 tables.
    lookup = robust.set_index('specification')
    cluster_rows = []
    for label, spec, color in [
        ('Event-date clusters', 'canonical_main_event_date_cluster', '#197E75'),
        ('Component clusters', 'canonical_main_component_cluster', '#B87333'),
        ('Country clusters', 'canonical_main_country_cluster', '#9A4F32'),
    ]:
        row = lookup.loc[spec]
        cluster_rows.append({'label': label, 'estimate': row.estimate, 'ci_low': row.ci_low, 'ci_high': row.ci_high, 'n_clusters': int(row.n_clusters), 'color': color})
    pd.DataFrame(cluster_rows).to_csv(FIG_TABLES / 'Fig3_panel_a_cluster_levels.csv', index=False)

    directional_rows = []
    for label, spec, color in [
        ('Aligned placebo', 'canonical_placebo_event_date_cluster', '#B87333'),
        ('False-event control', 'canonical_false_event_date_cluster', '#6B7280'),
    ]:
        row = lookup.loc[spec]
        directional_rows.append({'label': label, 'estimate': row.estimate, 'ci_low': row.ci_low, 'ci_high': row.ci_high, 'n_clusters': int(row.n_clusters), 'color': color})
    pd.DataFrame(directional_rows).to_csv(FIG_TABLES / 'Fig3_panel_b_controls.csv', index=False)

    top_country = strict_events['country'].value_counts().index[0]
    leave_year_rows = []
    for year in sorted(strict_matched['event_year'].unique()):
        result = cluster_mean(strict_matched.loc[~strict_matched['event_year'].eq(year)], main_outcome, 'event_id')
        leave_year_rows.append((int(year), result))
    min_year, _ = min(leave_year_rows, key=lambda item: item[1]['estimate'])
    leave_specs = [
        ('Main', strict_matched),
        (f'No {top_country}', strict_matched.loc[~strict_matched['event_country'].eq(top_country)]),
        ('No China-India', strict_matched.loc[~strict_matched['event_country'].isin(['China', 'India'])]),
        (f'No {min_year}', strict_matched.loc[~strict_matched['event_year'].eq(min_year)]),
    ]
    leaveout = []
    for label, subset in leave_specs:
        result = cluster_mean(subset, main_outcome, 'event_id')
        leaveout.append({'label': label, 'estimate': result['estimate'], 'ci_low': result['ci_low'], 'ci_high': result['ci_high'], 'n_clusters': result['n_clusters']})
    pd.DataFrame(leaveout).to_csv(FIG_TABLES / 'Fig3_panel_c_leaveout.csv', index=False)

    hazard_map = {
        'thunder_or_convective': 'Convective', 'low_ceiling_1000m': 'Low ceiling', 'low_ceiling_300m': 'Low ceiling',
        'low_vis_1km': 'Low visibility', 'low_vis_5km': 'Low visibility', 'gust_15ms': 'Wind', 'strong_wind': 'Wind', 'fog': 'Fog',
    }
    strict_events['hazard_family'] = strict_events['dominant_hazard'].map(hazard_map)
    stream = pd.crosstab(strict_events['event_year'], strict_events['hazard_family']).reindex(
        columns=['Low visibility', 'Low ceiling', 'Fog', 'Convective', 'Wind'], fill_value=0
    ).reset_index()
    stream.to_csv(FIG_TABLES / 'Fig3_panel_d_event_stream.csv', index=False)

    balance = []
    for label, event_col, control_col in [
        ('Coast distance', 'event_coast_distance_km', 'control_coast_distance_km'),
        ('Population density', 'event_population_density_proxy', 'control_population_density_proxy'),
        ('Airport distance', 'event_nearest_airport_km', 'control_nearest_airport_km'),
    ]:
        value = standardized_mean_difference(strict_matched[event_col], strict_matched[control_col])
        balance.append({'covariate': label, 'smd': value, 'absolute_smd': abs(value)})
    pd.DataFrame(balance).to_csv(FIG_TABLES / 'Fig3_panel_e_balance.csv', index=False)

    perm_summary, perm_iterations = event_permutation(strict_matched)
    perm_summary.to_csv(FIG_TABLES / 'Fig3_panel_f_permutation.csv', index=False)
    perm_summary.to_csv(SOURCE / 'event_q75_aligned_permutation_results.csv', index=False)
    perm_iterations.to_csv(SOURCE / 'event_q75_aligned_permutation_iterations.csv', index=False)

    # Supplemental year/country/hazard outputs.
    year_rows = []
    for year, subset in strict_matched.groupby('event_year', sort=True):
        result = cluster_mean(subset, main_outcome, 'event_id')
        year_rows.append({'year': int(year), 'estimate': result['estimate'], 'ci_low': result['ci_low'], 'ci_high': result['ci_high'], 'n_events': result['n_clusters']})
    pd.DataFrame(year_rows).to_csv(FIG_TABLES / 'Supplementary_Figure_6_year_estimates.csv', index=False)
    country_rows = []
    for country, subset in strict_matched.groupby('event_country', sort=True):
        if subset['event_id'].nunique() < 4:
            continue
        result = cluster_mean(subset, main_outcome, 'event_id')
        country_rows.append({'country': country, 'estimate': result['estimate'], 'ci_low': result['ci_low'], 'ci_high': result['ci_high'], 'n_events': result['n_clusters']})
    pd.DataFrame(country_rows).to_csv(FIG_TABLES / 'Supplementary_Figure_6_country_estimates.csv', index=False)
    strict_matched['hazard_family'] = strict_matched['dominant_hazard'].map(hazard_map)
    hazard_rows = []
    for family, subset in strict_matched.groupby('hazard_family', sort=True):
        result = cluster_mean(subset, main_outcome, 'event_id')
        hazard_rows.append({'hazard_family': family, 'estimate': result['estimate'], 'ci_low': result['ci_low'], 'ci_high': result['ci_high'], 'n_events': result['n_clusters']})
    pd.DataFrame(hazard_rows).to_csv(FIG_TABLES / 'Supplementary_Figure_6_hazard_estimates.csv', index=False)

    # Alignment and stream summary.
    main = lookup.loc['canonical_main_event_date_cluster']
    summary_table = pd.DataFrame([
        {'stream_or_diagnostic': 'Canonical 2024-calibrated q75 main', 'event_dates': strict_events['event_id'].nunique(), 'matched_or_eligible_rows': len(strict_matched), 'estimate': main.estimate, 'ci_low': main.ci_low, 'ci_high': main.ci_high, 'role': 'Primary selected-case precipitation consistency check'},
        {'stream_or_diagnostic': 'Canonical q75 component-cluster sensitivity', 'event_dates': strict_events['event_id'].nunique(), 'matched_or_eligible_rows': len(strict_matched), 'estimate': lookup.loc['canonical_main_component_cluster'].estimate, 'ci_low': lookup.loc['canonical_main_component_cluster'].ci_low, 'ci_high': lookup.loc['canonical_main_component_cluster'].ci_high, 'role': 'Dependence sensitivity'},
        {'stream_or_diagnostic': 'Canonical q75 country-cluster sensitivity', 'event_dates': strict_events['event_id'].nunique(), 'matched_or_eligible_rows': len(strict_matched), 'estimate': lookup.loc['canonical_main_country_cluster'].estimate, 'ci_low': lookup.loc['canonical_main_country_cluster'].ci_low, 'ci_high': lookup.loc['canonical_main_country_cluster'].ci_high, 'role': 'Dependence sensitivity'},
        {'stream_or_diagnostic': 'Canonical q75 aligned placebo', 'event_dates': placebo_ok['match_id'].nunique(), 'matched_or_eligible_rows': len(placebo_ok), 'estimate': lookup.loc['canonical_placebo_event_date_cluster'].estimate, 'ci_low': lookup.loc['canonical_placebo_event_date_cluster'].ci_low, 'ci_high': lookup.loc['canonical_placebo_event_date_cluster'].ci_high, 'role': 'Placebo branch'},
        {'stream_or_diagnostic': 'Canonical q75 false-event control', 'event_dates': false_ok['match_id'].nunique(), 'matched_or_eligible_rows': len(false_ok), 'estimate': lookup.loc['canonical_false_event_date_cluster'].estimate, 'ci_low': lookup.loc['canonical_false_event_date_cluster'].ci_low, 'ci_high': lookup.loc['canonical_false_event_date_cluster'].ci_high, 'role': 'Negative-control branch'},
    ])
    summary_table.to_csv(SOURCE / 'event_stream_alignment_and_contrast_summary.csv', index=False)

    return {
        'strict_events': int(strict_events['event_id'].nunique()),
        'strict_components': int(strict_events['component_id'].nunique()),
        'strict_countries': int(strict_events['country'].nunique()),
        'strict_matched_rows': int(len(strict_matched)),
        'strict_main_estimate': float(main.estimate),
        'strict_main_ci_low': float(main.ci_low),
        'strict_main_ci_high': float(main.ci_high),
        'component_ci_low': float(lookup.loc['canonical_main_component_cluster'].ci_low),
        'component_ci_high': float(lookup.loc['canonical_main_component_cluster'].ci_high),
        'country_ci_low': float(lookup.loc['canonical_main_country_cluster'].ci_low),
        'country_ci_high': float(lookup.loc['canonical_main_country_cluster'].ci_high),
        'permutation_p': float(perm_summary.loc[0, 'p_value_two_sided']),
    }


def write_dictionary_and_summary(summary: dict[str, float | int]) -> None:
    dictionary = pd.read_csv(SOURCE / 'data_dictionary.csv')
    additions = pd.DataFrame([
        {'file': 'canonical_2024_screen_flags.csv', 'grain': 'component-year', 'purpose': 'Canonical 2024-calibrated q75/q90/q95 flags and positive residual public-support gap.'},
        {'file': 'canonical_2024_q75_components.csv', 'grain': '2024 component', 'purpose': 'Canonical 651-component main screen.'},
        {'file': 'country_within_between_model_summary_2024.csv', 'grain': 'model coefficient', 'purpose': 'Pooled, within-country, between-country and country-fixed-effect need estimates.'},
        {'file': 'screen_calibration_and_comparison_rules.csv', 'grain': 'screen rule', 'purpose': 'Unified 2024 calibration and model-specific comparison rules used throughout the manuscript.'},
        {'file': 'multimodel_core_components_2024.csv', 'grain': '2024 component', 'purpose': '441-component intersection of canonical primary, available-field and country-fixed-effect screens.'},
    ])
    dictionary = pd.concat([dictionary, additions], ignore_index=True).drop_duplicates('file', keep='last')
    dictionary.to_csv(SOURCE / 'data_dictionary.csv', index=False)
    (SOURCE / 'canonical_2024_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main() -> int:
    panel = pd.read_csv(SOURCE / 'component_year_panel_full.csv', low_memory=False)
    panel2, screen = build_canonical_screen(panel)
    structural = build_structural_outputs(panel2, screen)
    event = build_event_outputs(panel2)
    summary = {**screen, **structural, **event}
    write_dictionary_and_summary(summary)

    checks = [
        summary['primary_n'] == 651,
        summary['primary_countries'] == 82,
        summary['country_fe_shared_n'] == 447,
        summary['frequency_ge_0_50'] == 490,
        summary['frequency_ge_0_80'] == 214,
        summary['model_core_n'] == 441,
        summary['strict_events'] == 152,
        summary['strict_matched_rows'] == 456,
        math.isclose(summary['strict_main_estimate'], 0.4062914210526314, abs_tol=1e-12),
    ]
    if not all(checks):
        raise SystemExit(f'V5 self-check failed: {checks}\n{json.dumps(summary, indent=2)}')
    print(json.dumps(summary, indent=2))
    print('FINAL_VERDICT=PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
