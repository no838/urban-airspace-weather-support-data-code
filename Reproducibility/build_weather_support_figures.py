#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import geopandas as gpd
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Source_Data'
FT = SOURCE / 'Figure_Tables'
MAIN = ROOT / 'Figures' / 'Main'
SUPP = ROOT / 'Figures' / 'Supplementary'
TEAL = '#197E75'
TEAL_DARK = '#0F5F59'
TEAL_LIGHT = '#BFDCD7'
BROWN = '#B87333'
BROWN_DARK = '#9A4F32'
GREY = '#C9D3DF'
DARK = '#1F2937'
LIGHT = '#E5EAF0'
BLUE = '#3B78A0'


def setup():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.7, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'figure.dpi': 160, 'savefig.dpi': 450,
    })


def clean(ax, zero_x=False, zero_y=False):
    ax.grid(False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#27313A'); ax.spines['bottom'].set_color('#27313A')
    if zero_x: ax.axvline(0, color='#27313A', lw=0.75, zorder=0)
    if zero_y: ax.axhline(0, color='#27313A', lw=0.75, zorder=0)


def panel(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left', color='#111827', clip_on=False)


def save(fig, directory: Path, stem: str):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f'{stem}.png', bbox_inches='tight', pad_inches=0.02, dpi=450)
    fig.savefig(directory / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(directory / f'{stem}.svg', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(directory / f'{stem}.tiff', bbox_inches='tight', pad_inches=0.02, dpi=450)
    plt.close(fig)


def forest(ax, df, label_col='label', xlim=None, xlabel='GPM contrast (mm)', colors=None):
    d = df.iloc[::-1].reset_index(drop=True)
    if colors is None:
        colors = d.get('color', pd.Series([TEAL] * len(d))).tolist()[::-1]
    elif isinstance(colors, str):
        colors = [colors] * len(d)
    else:
        colors = list(colors)[::-1]
    ax.axvline(0, color='#27313A', lw=0.75)
    for i, row in d.iterrows():
        ax.plot([row.ci_low, row.ci_high], [i, i], color=colors[i], lw=1.35)
        ax.scatter(row.estimate, i, color=colors[i], s=20, zorder=3)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d[label_col])
    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(*xlim)
    clean(ax)


def figure1():
    data = pd.read_csv(SOURCE / 'figure_ready_source_data_Fig1_global_map.csv')
    counts = pd.read_csv(FT / 'Fig1_panel_b_country_counts.csv')
    conc = pd.read_csv(FT / 'Fig1_panel_c_country_concentration.csv')
    top10 = counts.head(10).iloc[::-1]
    p99 = float(data['residual_public_support_gap_z'].quantile(0.99))
    vals = data['residual_public_support_gap_z'].clip(upper=p99)
    vmin = float(data['residual_public_support_gap_z'].quantile(0.02))
    fig = plt.figure(figsize=(7.2, 4.8))
    ax_map = fig.add_axes([0.025, 0.22, 0.64, 0.62])
    ax_b = fig.add_axes([0.735, 0.59, 0.235, 0.31])
    ax_c = fig.add_axes([0.735, 0.18, 0.235, 0.31])
    cax = fig.add_axes([0.08, 0.105, 0.52, 0.032])
    # Natural Earth is optional because some modern GeoPandas/Pyogrio installs
    # no longer bundle the old test-fixture shapefile.
    shp_candidates = [
        Path('/opt/pyvenv/lib/python3.13/site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp'),
        ROOT / 'Source_Data' / 'naturalearth_lowres' / 'naturalearth_lowres.shp',
    ]
    shp = next((candidate for candidate in shp_candidates if candidate.exists()), None)
    if shp is not None:
        world = gpd.read_file(shp)
        robin = CRS.from_proj4('+proj=robin +datum=WGS84 +units=m +no_defs')
        world = world.to_crs(robin)
        world.plot(ax=ax_map, facecolor='#F7F8F8', edgecolor='#D5DCE1', linewidth=0.28)
        transformer = Transformer.from_crs('EPSG:4326', robin, always_xy=True)
        x, y = transformer.transform(data['centroid_lon'].to_numpy(), data['centroid_lat'].to_numpy())
    else:
        ax_map.set_facecolor('#F7F8F8')
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-60, 85)
        ax_map.set_aspect('equal')
        x, y = data['centroid_lon'].to_numpy(), data['centroid_lat'].to_numpy()
    cmap = mpl.colors.LinearSegmentedColormap.from_list('gap', [TEAL, '#A5CFC8', '#E3C17D', '#E68613'])
    uncertain = data['denominator_uncertainty_flag'].astype(bool).to_numpy()
    sc = ax_map.scatter(np.asarray(x)[~uncertain], np.asarray(y)[~uncertain], c=vals.to_numpy()[~uncertain], cmap=cmap, vmin=vmin, vmax=p99, s=13, marker='o', lw=0.25, edgecolor='#263238', alpha=0.93, zorder=3)
    ax_map.scatter(np.asarray(x)[uncertain], np.asarray(y)[uncertain], c=vals.to_numpy()[uncertain], cmap=cmap, vmin=vmin, vmax=p99, s=18, marker='^', lw=0.25, edgecolor='#263238', alpha=0.93, zorder=3)
    ax_map.set_axis_off(); ax_map.set_aspect('equal'); panel(ax_map, 'a', x=-0.02, y=0.98)
    cbar = fig.colorbar(sc, cax=cax, orientation='horizontal', extend='max')
    cbar.set_label('Residual public-support gap (z; capped at p99)'); cbar.outline.set_visible(False)
    ax_map.scatter([], [], marker='o', s=18, facecolor='white', edgecolor=DARK, label='denominator certain')
    ax_map.scatter([], [], marker='^', s=22, facecolor='white', edgecolor=DARK, label='denominator uncertain')
    ax_map.legend(loc='lower left', bbox_to_anchor=(0.00, -0.01), frameon=False, handletextpad=0.3)
    ax_b.barh(top10.country, top10.retained_components, color=TEAL, height=0.7)
    clean(ax_b); ax_b.set_xlabel('Retained components'); ax_b.set_xlim(0, max(150, counts.retained_components.max() * 1.1)); panel(ax_b, 'b', x=-0.18, y=1.12)
    share = counts.loc[counts.country.isin(['China','India']), 'retained_components'].sum() / counts.retained_components.sum()
    ax_b.text(0.97, 0.06, f'China + India\n{share*100:.1f}%', transform=ax_b.transAxes, ha='right', va='bottom', color=DARK)
    ax_c.plot(conc['rank'], conc['cumulative_share'], color=TEAL_DARK, lw=1.7)
    ax_c.scatter(conc['rank'].head(20), conc['cumulative_share'].head(20), s=10, color=TEAL)
    clean(ax_c); ax_c.set_xlabel('Country rank'); ax_c.set_ylabel('Cumulative share'); ax_c.set_ylim(0,1.03); panel(ax_c, 'c', x=-0.18, y=1.12)
    ax_c.axhline(share, color='#6B7280', lw=0.65, ls=':')
    ax_c.text(0.98, 0.15, f'{(1-share)*100:.1f}% outside\nChina + India', transform=ax_c.transAxes, ha='right', va='bottom', color=DARK)
    save(fig, MAIN, 'Figure_1')

def figure2():
    dec = pd.read_csv(FT / 'Fig2_panel_a_need_support_deciles.csv')
    dist = pd.read_csv(FT / 'Fig2_panel_b_residual_distribution_summary.csv')
    plane = pd.read_csv(FT / 'Fig2_panel_c_decision_plane_sample.csv')
    annual = pd.read_csv(FT / 'Fig2_panel_d_annual_threshold_stability.csv')
    rules = pd.read_csv(SOURCE / 'screen_calibration_and_comparison_rules.csv')
    q75 = rules.loc[(rules.record_type=='primary calibration') & (rules.branch_or_screen=='q75')].iloc[0]
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = GridSpec(2,2,figure=fig,wspace=0.38,hspace=0.48)
    axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[1,0]); axd=fig.add_subplot(gs[1,1])
    axa.plot(dec.need_decile,dec.observed_median,color=TEAL_DARK,lw=1.8,marker='o',ms=3.6,label='Observed median')
    axa.plot(dec.need_decile,dec.expected_median,color=BROWN,lw=1.4,marker='s',ms=3.2,label='Expected median')
    axa.fill_between(dec.need_decile,dec.observed_p10,dec.observed_p25,color=TEAL_LIGHT,alpha=.55,lw=0,label='Observed p10-p25')
    clean(axa,zero_y=True); axa.set_xlabel('Need decile (2024)'); axa.set_ylabel('Public support (z)'); axa.set_xticks(dec.need_decile); axa.legend(frameon=False,loc='lower right'); panel(axa,'a')
    allv=dist.residual_public_support_gap_z; keep=dist.canonical_2024_q75.astype(bool); bins=np.linspace(allv.min(),allv.max(),35)
    axb.hist(allv,bins=bins,color=LIGHT,edgecolor='white',lw=.25,label='All components')
    axb.hist(allv[keep],bins=bins,color=TEAL,alpha=.9,edgecolor='white',lw=.25,label='q75 retained')
    gap_thr=-float(q75.residual_threshold_z); axb.axvline(gap_thr,color=BROWN_DARK,lw=.9,ls='--'); axb.text(gap_thr,1.02,'q25 residual threshold',transform=axb.get_xaxis_transform(),ha='center',va='bottom',color=BROWN_DARK)
    clean(axb); axb.set_xlabel('Residual public-support gap (z)'); axb.set_ylabel('Components'); axb.legend(frameon=False); panel(axb,'b')
    rng=np.random.default_rng(44); sample=plane.sample(n=min(5000,len(plane)),random_state=44); k=sample.canonical_2024_q75.astype(bool)
    axc.scatter(sample.loc[~k,'need_composite_z'],sample.loc[~k,'residual_public_support_gap_z'],s=4,color='#B9C3CF',alpha=.27,lw=0)
    axc.scatter(sample.loc[k,'need_composite_z'],sample.loc[k,'residual_public_support_gap_z'],s=7,color=TEAL_DARK,alpha=.75,lw=0)
    axc.axvline(float(q75.need_threshold_z),color=BROWN,lw=.8,ls='--'); axc.axhline(gap_thr,color=BROWN_DARK,lw=.8,ls='--')
    clean(axc); axc.set_xlabel('Need composite (z)'); axc.set_ylabel('Residual public-support gap (z)'); axc.text(.70,.90,'q75 retained',transform=axc.transAxes,color=TEAL_DARK); panel(axc,'c')
    for col,color,marker,label in [('q75_share',TEAL_DARK,'o','q75'),('q90_share',BROWN,'s','q90'),('q95_share',BROWN_DARK,'^','q95')]:
        axd.plot(annual.year,annual[col]*100,color=color,lw=1.6,marker=marker,ms=3.2)
        axd.text(2024.35,annual[col].iloc[-1]*100,label,color=color,va='center')
    clean(axd); axd.set_xlabel('Year'); axd.set_ylabel('Retained components (%)'); axd.set_xticks(annual.year[::2]); axd.set_xlim(2014.55,2024.85); axd.set_ylim(-.1,6.3); axd.text(.03,.73,'Fixed 2024 thresholds',transform=axd.transAxes,color=DARK); panel(axd,'d')
    save(fig,MAIN,'Figure_2')


def figure3():
    clusters=pd.read_csv(FT/'Fig3_panel_a_cluster_levels.csv')
    controls=pd.read_csv(FT/'Fig3_panel_b_controls.csv')
    leave=pd.read_csv(FT/'Fig3_panel_c_leaveout.csv')
    stream=pd.read_csv(FT/'Fig3_panel_d_event_stream.csv')
    balance=pd.read_csv(FT/'Fig3_panel_e_balance.csv')
    perm=pd.read_csv(FT/'Fig3_panel_f_permutation.csv').iloc[0]
    fig=plt.figure(figsize=(7.6,5.5))
    axa=fig.add_axes([.10,.60,.22,.31]); axb=fig.add_axes([.43,.60,.20,.31]); axc=fig.add_axes([.74,.60,.23,.31])
    axd=fig.add_axes([.11,.16,.20,.25]); axe=fig.add_axes([.44,.16,.23,.25]); axf=fig.add_axes([.77,.16,.18,.25])
    forest(axa,clusters,xlim=(-.3,1.05)); panel(axa,'a',x=-.28,y=1.08); axa.text(.32,.76,'152 selected event dates',transform=axa.transAxes,color=DARK)
    forest(axb,controls,xlim=(-.65,1.05),xlabel='Directional contrast (mm)'); panel(axb,'b',x=-.28,y=1.08)
    forest(axc,leave,xlim=(-.25,1.05)); panel(axc,'c',x=-.25,y=1.08)
    fams=['Low visibility','Low ceiling','Fog','Convective','Wind']; cols=[TEAL,BROWN,'#AAB4C0',BLUE,'#6B7280']; bottom=np.zeros(len(stream))
    for fam,col in zip(fams,cols):
        vals=stream.get(fam,pd.Series(np.zeros(len(stream)))).to_numpy(); axd.bar(stream.event_year,vals,bottom=bottom,color=col,width=.7,label=fam); bottom+=vals
    clean(axd); axd.set_xlabel('Event year'); axd.set_ylabel('Selected event dates'); axd.set_xticks(stream.event_year); axd.tick_params(axis='x',rotation=38); axd.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.35),handlelength=.8,columnspacing=.8); panel(axd,'d',x=-.22,y=1.10)
    bal=balance.sort_values('absolute_smd'); axe.barh(bal.covariate,bal.absolute_smd,color='#70B7AE',height=.62); axe.axvline(.1,color=BROWN,lw=.8); clean(axe); axe.set_xlabel('Absolute standardized mean difference'); axe.text(.1,1.02,'0.1 reference',transform=axe.get_xaxis_transform(),ha='center',va='bottom',color=BROWN); panel(axe,'e',x=-.25,y=1.10)
    axf.axvspan(perm.null_q025,perm.null_q975,color=GREY,alpha=.55,lw=0,label='null 95%'); axf.axvline(0,color='#27313A',lw=.75); axf.scatter([perm.observed],[0],color=TEAL,s=25,zorder=3,label='observed'); axf.set_yticks([]); axf.set_xlabel('Permutation (mm)'); axf.set_xlim(-.35,.8); clean(axf); axf.legend(frameon=False,loc='upper right'); axf.text(.52,.06,f'p = {perm.p_value_two_sided:.4f}',transform=axf.transAxes,color=DARK); panel(axf,'f',x=-.25,y=1.10)
    save(fig,MAIN,'Figure_3')


def figure4():
    bench=pd.read_csv(FT/'Fig4_panel_a_benchmark_overlap_fixed_size.csv')
    coef=pd.read_csv(FT/'Fig4_panel_b_mundlak_coefficients.csv')
    sens=pd.read_csv(FT/'Fig4_panel_c_sensitivity_overlap.csv')
    comp=pd.read_csv(FT/'Fig4_panel_d_selection_frequency.csv')
    primary=comp.canonical_2024_q75.astype(bool); freq=comp.loc[primary,'dirichlet_selection_frequency']
    fig=plt.figure(figsize=(7.2,4.8)); gs=GridSpec(2,2,figure=fig,wspace=.52,hspace=.62); axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[1,0]); axd=fig.add_subplot(gs[1,1])
    b=bench.sort_values('overlap_with_q75_screen_pct'); axa.barh(b.benchmark,b.overlap_with_q75_screen_pct,color=TEAL,height=.68); axa.axvspan(b.random_overlap_q025_pct.iloc[0],b.random_overlap_q975_pct.iloc[0],color=GREY,alpha=.45,label='random 95%'); axa.axvline(b.random_overlap_median_pct.iloc[0],color='#6B7280',lw=.8,ls=':'); clean(axa); axa.set_xlabel('Overlap with 651-component q75 screen (%)'); axa.legend(frameon=False,loc='lower right'); panel(axa,'a',x=-.15,y=1.10)
    rows=coef.loc[coef.coefficient_role.isin(['pooled need slope','within-country need slope','between-country need slope'])].copy(); labelmap={'pooled need slope':'Pooled need slope','within-country need slope':'Within-country need slope','between-country need slope':'Between-country need slope'}; rows['label']=rows.coefficient_role.map(labelmap); colors=[TEAL,BROWN_DARK,TEAL_DARK]; forest(axb,rows,label_col='label',xlim=(-.22,.65),xlabel='Need coefficient',colors=colors); axb.text(.98,.40,'Pooled R² 0.285\nWithin-between R² 0.358\nCountry-FE R² 0.454',transform=axb.transAxes,ha='right',va='center',color=DARK); panel(axb,'b',x=-.18,y=1.10)
    order=['Available-field refit','Within-between model','Country fixed effects','No-hazard need']; ss=sens.set_index('sensitivity').loc[order].reset_index(); ypos=np.arange(len(ss))[::-1]; cols=[BROWN if x=='No-hazard need' else TEAL for x in ss.sensitivity]; axc.barh(ypos,ss.jaccard_with_primary,color=cols,height=.68); axc.set_yticks(ypos); axc.set_yticklabels(ss.sensitivity); clean(axc); axc.set_xlabel('Jaccard with primary 2024 q75'); axc.set_xlim(0,.92); panel(axc,'c',x=-.15,y=1.10)
    for yval,(_,row) in zip(ypos,ss.iterrows()): axc.text(row.jaccard_with_primary+.012,yval,f'{int(row.shared_with_primary)}',va='center',color=DARK)
    bins=np.linspace(0,1,21); axd.hist(freq,bins=bins,color=TEAL,alpha=.9,edgecolor='white',lw=.3); axd.axvline(.5,color=BROWN,lw=.9); axd.axvline(.8,color=BROWN_DARK,lw=.9,ls=':'); clean(axd); axd.set_xlabel('Selection frequency (1,000 weight draws)'); axd.set_ylabel('Primary q75 components'); axd.text(.98,.96,f'{int((freq>=.5).sum())} ≥ 0.50\n{int((freq>=.8).sum())} ≥ 0.80\nmodel core: {int(comp.model_core_primary_available_country_fe.sum())}',transform=axd.transAxes,ha='right',va='top',color=DARK); panel(axd,'d',x=-.18,y=1.10)
    save(fig,MAIN,'Figure_4')


def supplementary6():
    years=pd.read_csv(FT/'Supplementary_Figure_6_year_estimates.csv')
    countries=pd.read_csv(FT/'Supplementary_Figure_6_country_estimates.csv')
    hazards=pd.read_csv(FT/'Supplementary_Figure_6_hazard_estimates.csv')
    robust=pd.read_csv(SOURCE/'event_q75_aligned_cluster_robustness.csv')
    fig=plt.figure(figsize=(7.2,4.9)); gs=GridSpec(2,2,figure=fig,wspace=.58,hspace=.58); axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[1,0]); axd=fig.add_subplot(gs[1,1])
    y=years.copy(); y['label']=y.year.astype(str); forest(axa,y,label_col='label',xlim=(-.6,1.7),xlabel='GPM contrast (mm)',colors=TEAL); panel(axa,'a',x=-.2,y=1.10)
    c=countries.sort_values('estimate'); c['label']=c.country.str.replace('México','Mexico',regex=False); forest(axb,c,label_col='label',xlim=(-2.2,5.4),xlabel='GPM contrast (mm)',colors=[BROWN_DARK if v<0 else TEAL for v in c.estimate]); panel(axb,'b',x=-.2,y=1.10)
    h=hazards.sort_values('estimate'); h['label']=h.hazard_family; forest(axc,h,label_col='label',xlim=(-1.3,3.2),xlabel='GPM contrast (mm)',colors=[BROWN_DARK if v<0 else TEAL for v in h.estimate]); panel(axc,'c',x=-.2,y=1.10)
    specs=['canonical_main_event_date_cluster','canonical_main_component_cluster','canonical_main_country_cluster']; rr=robust.set_index('specification').loc[specs].reset_index(); rr['label']=['Event dates','Components','Countries']; forest(axd,rr,label_col='label',xlim=(-.3,1.05),xlabel='GPM contrast (mm)',colors=[TEAL,BROWN,BROWN_DARK]); panel(axd,'d',x=-.2,y=1.10)
    save(fig,SUPP,'Supplementary_Figure_6')


def supplementary7():
    controls=pd.read_csv(FT/'Fig3_panel_b_controls.csv')
    perm=pd.read_csv(FT/'Fig3_panel_f_permutation.csv').iloc[0]
    robust=pd.read_csv(SOURCE/'event_q75_aligned_cluster_robustness.csv')
    fig=plt.figure(figsize=(7.2,4.7)); gs=GridSpec(2,2,figure=fig,wspace=.52,hspace=.55); axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[1,0]); axd=fig.add_subplot(gs[1,1])
    forest(axa,controls.iloc[[0]],xlim=(-.3,.8),xlabel='Directional contrast (mm)',colors=BROWN); panel(axa,'a',x=-.2,y=1.10)
    forest(axb,controls.iloc[[1]],xlim=(-.7,1.1),xlabel='Directional contrast (mm)',colors='#6B7280'); panel(axb,'b',x=-.2,y=1.10)
    axc.axvspan(perm.null_q025,perm.null_q975,color=GREY,alpha=.55,lw=0); axc.axvline(0,color='#27313A',lw=.75); axc.scatter([perm.observed],[0],color=TEAL,s=26); axc.set_yticks([]); axc.set_xlabel('Within-event permutation (mm)'); axc.text(.98,.88,f'p = {perm.p_value_two_sided:.4f}',transform=axc.transAxes,ha='right',color=DARK); clean(axc); panel(axc,'c',x=-.2,y=1.10)
    specs=['canonical_main_adjusted_event','canonical_main_adjusted_component_cluster','canonical_main_adjusted_country_cluster']; rr=robust.set_index('specification').loc[specs].reset_index(); rr['label']=['Adjusted event','Adjusted component','Adjusted country']; forest(axd,rr,label_col='label',xlim=(-.2,1.25),xlabel='Adjusted GPM contrast (mm)',colors=[TEAL,BROWN,BROWN_DARK]); panel(axd,'d',x=-.2,y=1.10)
    save(fig,SUPP,'Supplementary_Figure_7')


def supplementary9():
    fig,ax=plt.subplots(figsize=(7.2,3.7)); ax.axis('off')
    stages=[
        ('Stage 1','Global\nscreening','Completed\nhere'),
        ('Stage 2','Local source and\nmetadata audit','Required\nlocally'),
        ('Stage 3','Variable-specific\nvalidation','Requires task-specific\nobservations'),
        ('Stage 4','Decision-facing\nservice evaluation','Requires users and\noperational endpoints'),
    ]
    xs=[.025,.267,.509,.751]
    box_w=.205
    for i,(stage,title,note) in enumerate(stages):
        box=FancyBboxPatch((xs[i],.34),box_w,.38,boxstyle='round,pad=0.010,rounding_size=.014',facecolor='#EFF6F5' if i==0 else '#F7F8FA',edgecolor=TEAL if i==0 else '#9CA3AF',lw=1.2)
        ax.add_patch(box)
        cx=xs[i]+box_w/2
        ax.text(cx,.62,stage,ha='center',va='center',fontweight='bold',fontsize=9,color=TEAL_DARK if i==0 else DARK)
        ax.text(cx,.51,title,ha='center',va='center',fontsize=7.7,linespacing=1.15,color=DARK)
        ax.text(cx,.385,note,ha='center',va='center',fontsize=6.3,linespacing=1.12,color='#4B5563')
        if i<3:
            ax.add_patch(FancyArrowPatch((xs[i]+box_w+.004,.53),(xs[i+1]-.006,.53),arrowstyle='-|>',mutation_scale=10,color='#6B7280',lw=.9))
    ax.text(.5,.88,'Interpretation ladder for the public weather-information support diagnostic',ha='center',fontweight='bold',fontsize=10,color='#111827')
    ax.text(.5,.16,'Supported evidence ceiling: released-layer diagnostic and selected-case precipitation consistency.\nBlocked: operational flight risk, route disruption, hidden capacity, causal station failure, or direct wind/visibility/ceiling validation.',ha='center',va='center',fontsize=7.6,color=DARK)
    ax.set_xlim(0,1); ax.set_ylim(0,1); save(fig,SUPP,'Supplementary_Figure_9')


def main():
    setup(); figure1(); figure2(); figure3(); figure4(); supplementary6(); supplementary7(); supplementary9()
    print('Built Figures 1-4 and Supplementary Figures 6, 7, 9')

if __name__=='__main__': main()
