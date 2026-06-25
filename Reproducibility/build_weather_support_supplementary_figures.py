#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT=Path(__file__).resolve().parents[1]
SD=ROOT/'Source_Data'; FT=SD/'Figure_Tables'; SUPP=ROOT/'Figures'/'Supplementary'
TEAL='#197E75'; TEAL_DARK='#0F5F59'; TEAL_LIGHT='#BFDCD7'; BROWN='#B87333'; BROWN_DARK='#9A4F32'; GREY='#C9D3DF'; DARK='#1F2937'; BLUE='#3B78A0'

def setup():
 mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.7,'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':450})
def clean(ax,zero_x=False,zero_y=False):
 ax.grid(False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
 ax.spines['left'].set_color('#27313A'); ax.spines['bottom'].set_color('#27313A')
 if zero_x: ax.axvline(0,color='#27313A',lw=.75)
 if zero_y: ax.axhline(0,color='#27313A',lw=.75)
def panel(ax,label,x=-.12,y=1.08): ax.text(x,y,label,transform=ax.transAxes,fontsize=10,fontweight='bold',va='top',ha='left',color='#111827',clip_on=False)
def save(fig,stem):
 SUPP.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf','svg','tiff']:
  kw={'dpi':450} if ext in ['png','tiff'] else {}
  fig.savefig(SUPP/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.02,**kw)
 plt.close(fig)
def forest(ax,d,label='label',xlim=None,xlabel='Estimate',colors=None):
 d=d.iloc[::-1].reset_index(drop=True); ax.axvline(0,color='#27313A',lw=.75)
 if colors is None: colors=[TEAL]*len(d)
 elif isinstance(colors,str): colors=[colors]*len(d)
 else: colors=list(colors)[::-1]
 for i,r in d.iterrows():
  ax.plot([r.ci_low,r.ci_high],[i,i],color=colors[i],lw=1.3); ax.scatter(r.estimate,i,color=colors[i],s=18,zorder=3)
 ax.set_yticks(range(len(d))); ax.set_yticklabels(d[label]); ax.set_xlabel(xlabel)
 if xlim: ax.set_xlim(*xlim)
 clean(ax)

def figure1():
 annual=pd.read_csv(FT/'Fig2_panel_d_annual_threshold_stability.csv')
 pdata=pd.read_csv(SD/'component_year_panel_full.csv',usecols=['year','country','denominator_uncertainty_flag'])
 rows=pdata.groupby('year').size(); countries=pdata.groupby('year').country.nunique(); uncert=pdata.groupby('year').denominator_uncertainty_flag.mean()*100
 fig=plt.figure(figsize=(7.2,4.8)); gs=GridSpec(2,2,figure=fig,wspace=.42,hspace=.50)
 a,b,c,d=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
 a.bar(rows.index,rows.values,color=TEAL); clean(a); a.set_ylabel('Component-year rows'); a.set_xlabel('Year'); a.set_ylim(0,12500); panel_label=lambda ax,l: panel(ax,l)
 panel(a,'a')
 b.plot(countries.index,countries.values,color=BROWN,marker='o',ms=3); clean(b); b.set_ylabel('Countries'); b.set_xlabel('Year'); b.set_ylim(180,202); panel(b,'b')
 c.plot(annual.year,annual.q75,color=TEAL_DARK,marker='o',ms=3); clean(c); c.set_ylabel('q75 retained components'); c.set_xlabel('Year'); c.set_ylim(640,690); panel(c,'c')
 d.plot(uncert.index,uncert.values,color=BROWN,marker='s',ms=3,label='uncertain'); d.plot(uncert.index,100-uncert.values,color=TEAL,marker='o',ms=3,label='certain'); clean(d); d.set_ylabel('Annual rows (%)'); d.set_xlabel('Year'); d.set_ylim(45,55); d.legend(frameon=False); panel(d,'d')
 save(fig,'Supplementary_Figure_1')

def figure2():
 corr=pd.read_csv(FT/'Supplementary_Figure_2_corr.csv',index_col=0)
 assoc=pd.read_csv(FT/'Supplementary_Figure_2_proxy_assoc.csv')
 annual=pd.read_csv(FT/'Fig2_panel_d_annual_threshold_stability.csv')
 paneldata=pd.read_csv(SD/'component_year_panel_full.csv',usecols=['year','need_composite_z','denominator_uncertainty_flag'])
 fig=plt.figure(figsize=(7.2,4.9)); gs=GridSpec(2,2,figure=fig,wspace=.42,hspace=.52)
 a,b,c,d=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
 im=a.imshow(corr.values,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto'); a.set_xticks(range(len(corr.columns))); a.set_xticklabels(['Demand','Hazard','Urban form','Population','NTL','Vertiport','Heliport','Policy','Verti/heli','Need'],rotation=55,ha='right'); a.set_yticks(range(len(corr.index))); a.set_yticklabels(['Demand','Hazard','Urban form','Population','NTL','Vertiport','Heliport','Policy','Verti/heli','Need']); fig.colorbar(im,ax=a,fraction=.046,pad=.04,label='Correlation'); panel(a,'a',x=-.18)
 aa=assoc.sort_values('spearman_rho'); b.barh(aa.proxy,aa.spearman_rho,color=[BROWN_DARK if x<0 else TEAL for x in aa.spearman_rho]); b.axvline(0,color='#27313A',lw=.7); clean(b); b.set_xlabel('Spearman rho'); panel(b,'b')
 for col,color,marker,label in [('q75',TEAL_DARK,'o','q75'),('q90',BROWN,'s','q90'),('q95',BROWN_DARK,'^','q95')]: c.plot(annual.year,annual[col],color=color,marker=marker,ms=3,label=label)
 clean(c); c.set_xlabel('Year'); c.set_ylabel('Retained components'); c.legend(frameon=False); panel(c,'c')
 x0=paneldata.loc[~paneldata.denominator_uncertainty_flag.astype(bool),'need_composite_z'].dropna(); x1=paneldata.loc[paneldata.denominator_uncertainty_flag.astype(bool),'need_composite_z'].dropna(); d.boxplot([x0,x1],tick_labels=['certain','uncertain'],patch_artist=True,boxprops={'facecolor':TEAL_LIGHT},medianprops={'color':BROWN_DARK}); clean(d); d.set_ylabel('Need composite (z)'); panel(d,'d')
 save(fig,'Supplementary_Figure_2')

def figure8():
 primary=pd.read_csv(SD/'canonical_2024_q75_components.csv')
 counts=primary.groupby('country').size().sort_values(ascending=False); conc=counts.cumsum()/counts.sum(); top=counts.head(8).sort_values()
 panel2024=pd.read_csv(SD/'component_year_panel_full.csv'); panel2024=panel2024[panel2024.year==2024].copy(); flags=pd.read_csv(SD/'canonical_2024_screen_flags.csv'); flags=flags[flags.year==2024][['component_id','canonical_2024_q75']]; panel2024=panel2024.merge(flags,on='component_id',how='left')
 # coastal <=50 km vs inland, with leave-out scopes
 rows=[]
 for label,mask in [('global',pd.Series(True,index=panel2024.index)),('exclude China',panel2024.country!='China'),('exclude India',panel2024.country!='India'),('exclude China+India',~panel2024.country.isin(['China','India']))]:
  x=panel2024[mask]; coast=x.coast_distance_km<=50
  rows.append((label,100*x.loc[coast,'canonical_2024_q75'].mean(),100*x.loc[~coast,'canonical_2024_q75'].mean()))
 den=panel2024.groupby('denominator_uncertainty_flag').canonical_2024_q75.sum()
 fig=plt.figure(figsize=(7.2,4.8)); gs=GridSpec(2,2,figure=fig,wspace=.44,hspace=.50); a,b,c,d=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
 a.plot(np.arange(1,len(conc)+1),conc.values,color=TEAL_DARK,lw=1.7); a.scatter(np.arange(1,min(21,len(conc)+1)),conc.values[:20],color=TEAL,s=10); clean(a); a.set_xlabel('Countries ranked by retained components'); a.set_ylabel('Cumulative share'); panel(a,'a')
 b.barh(top.index,top.values,color=TEAL); clean(b); b.set_xlabel('Retained components'); panel(b,'b')
 rr=pd.DataFrame(rows,columns=['scope','coastal','inland']); xx=np.arange(len(rr)); w=.34; c.bar(xx-w/2,rr.coastal,w,color=TEAL,label='coastal ≤50 km'); c.bar(xx+w/2,rr.inland,w,color=BROWN,label='inland'); c.set_xticks(xx); c.set_xticklabels(rr.scope,rotation=22,ha='right'); c.set_ylabel('q75 prevalence (%)'); c.legend(frameon=False,fontsize=6); clean(c); panel(c,'c')
 d.bar(['certain','uncertain'],[int(den.get(False,0)),int(den.get(True,0))],color=[TEAL,BROWN]); clean(d); d.set_ylabel('Retained components'); panel(d,'d')
 save(fig,'Supplementary_Figure_8')

def figure10():
 draws=pd.read_csv(SD/'dirichlet_2024_refit_draw_summary.csv')
 sens=pd.read_csv(SD/'screen_sensitivity_overlap_2024.csv')
 r3=pd.read_csv(SD/'Robustness_Tables_Reorganized_SI/Supplementary_Table_R3_model_form_robustness.csv')
 r6=pd.read_csv(SD/'Robustness_Tables_Reorganized_SI/Supplementary_Table_R6_conditional_event_robustness_current_selected_events.csv')
 # canonical 2024 threshold grid
 p=pd.read_csv(SD/'component_year_panel_full.csv'); x=p[p.year==2024].copy(); grid=[]
 for nq in [.70,.75,.80,.85,.90,.95]:
  nth=x.need_composite_z.quantile(nq)
  for rq in [.25,.20,.15,.10,.05]:
   rth=x.support_residual_z.quantile(rq); n=int(((x.need_composite_z>=nth)&(x.support_residual_z<=rth)).sum()); grid.append({'need_quantile':nq,'residual_quantile':rq,'retained_2024':n,'share_pct':100*n/len(x)})
 grid=pd.DataFrame(grid); grid.to_csv(SD/'canonical_threshold_grid_2024.csv',index=False)
 mat=grid.pivot(index='need_quantile',columns='residual_quantile',values='retained_2024').sort_index(ascending=False)
 fig=plt.figure(figsize=(7.6,5.1)); gs=GridSpec(2,3,figure=fig,wspace=.72,hspace=.55)
 a,b,c,d,e,f=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]]
 a.hist(draws.jaccard_vs_primary_q75_2024,bins=25,color=TEAL,edgecolor='white',lw=.3); a.axvline(draws.jaccard_vs_primary_q75_2024.median(),color=BROWN,lw=.9); clean(a); a.set_xlabel('Jaccard with canonical q75'); a.set_ylabel('Weight draws'); panel(a,'a')
 b.scatter(draws.retained_2024,draws.countries_2024,s=10,color=TEAL,alpha=.45); b.axvline(651,color=BROWN,lw=.8); b.axhline(82,color=BROWN_DARK,lw=.8); clean(b); b.set_xlabel('Retained components (2024)'); b.set_ylabel('Countries (2024)'); panel(b,'b')
 ss=sens.sort_values('jaccard_with_primary'); yy=np.arange(len(ss)); c.scatter(ss.jaccard_with_primary,yy,color=TEAL,s=24,label='Jaccard'); c.scatter(ss.share_of_primary_recovered,yy,color=BROWN,marker='x',s=32,label='Primary recovered'); c.set_yticks(yy); c.set_yticklabels(ss.sensitivity); c.set_xlim(0,1); c.set_xlabel('Canonical agreement'); c.legend(frameon=False,loc='lower right'); clean(c); panel(c,'c')
 im=d.imshow(mat.values,cmap='viridis',aspect='auto'); d.set_xticks(range(len(mat.columns))); d.set_xticklabels([f'q{int(v*100)}' for v in mat.columns]); d.set_yticks(range(len(mat.index))); d.set_yticklabels([f'q{int(v*100)}' for v in mat.index]); d.set_xlabel('Lower-support residual threshold'); d.set_ylabel('Need threshold'); fig.colorbar(im,ax=d,fraction=.040,pad=.025); panel(d,'d')
 picks=['primary_q75_stored','ols_need_only','ols_quadratic_controls','ols_log_transformed_controls']; ev=r6[r6.specification.isin(picks)].copy(); ev['label']=ev.specification.map({'primary_q75_stored':'Canonical q75','ols_need_only':'Need only','ols_quadratic_controls':'Quadratic','ols_log_transformed_controls':'Log controls'}); ev=ev.sort_values('matched_gpm_estimate'); ev2=pd.DataFrame({'label':ev.label,'estimate':ev.matched_gpm_estimate,'ci_low':ev.matched_gpm_ci_low,'ci_high':ev.matched_gpm_ci_high}); forest(e,ev2,label='label',xlim=(-.15,.9),xlabel='Conditional GPM contrast (mm)',colors=TEAL); panel(e,'e')
 vals=[114220,152,456,0]; labs=['component-year panel','canonical events','matched rows','complete event universe']; f.bar(range(4),vals,color=TEAL); f.set_yscale('symlog',linthresh=1); f.set_xticks(range(4)); f.set_xticklabels(labs,rotation=30,ha='right'); f.set_ylabel('Rows (symlog)'); clean(f); panel(f,'f');
 for i,v in enumerate(vals): f.text(i,v if v else .3,str(v),ha='center',va='bottom',fontsize=7)
 save(fig,'Supplementary_Figure_10')

def figure11():
 grid=pd.read_csv(SD/'canonical_threshold_grid_2024.csv') if (SD/'canonical_threshold_grid_2024.csv').exists() else None
 if grid is None: return
 mat=grid.pivot(index='need_quantile',columns='residual_quantile',values='share_pct').sort_index(ascending=False)
 fig,ax=plt.subplots(figsize=(7.2,4.8)); im=ax.imshow(mat.values,cmap='viridis',aspect='auto'); ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels([f'q{int(v*100)}' for v in mat.columns]); ax.set_yticks(range(len(mat.index))); ax.set_yticklabels([f'q{int(v*100)}' for v in mat.index]); ax.set_xlabel('Lower-support residual threshold'); ax.set_ylabel('Need threshold')
 for i in range(mat.shape[0]):
  for j in range(mat.shape[1]): ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='black' if mat.iloc[i,j]<5.5 else 'white')
 fig.colorbar(im,ax=ax,label='2024 retained share (%)'); fig.tight_layout(); save(fig,'Supplementary_Figure_11')

def main():
 setup(); figure1(); figure2(); figure8(); figure10(); figure11(); print('Built supplementary figures 1,2,8,10,11')
if __name__=='__main__': main()
