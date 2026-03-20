"""
Professional Figures — Industry Grade
All fonts large, all panels readable, no wasted space
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# ── Colors ────────────────────────────────────────────────
NAVY   = '#1F3864'
BLUE   = '#2E75B6'
LBLUE  = '#BDD7EE'
GREEN  = '#375623'
ORANGE = '#C55A11'
RED    = '#C00000'
GRAY   = '#595959'
WHITE  = '#FFFFFF'

COOLING_COLORS = {
    'Air-side':     '#2E75B6',
    'Evaporative':  '#C55A11',
    'Chilled Water':'#375623',
    'Seawater':     '#7030A0',
    'Liquid':       '#C00000',
}

# ── Global font settings ──────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         18,
    'axes.titlesize':    20,
    'axes.titleweight':  'bold',
    'axes.titlepad':     18,
    'axes.labelsize':    17,
    'axes.labelweight':  'bold',
    'axes.grid':         True,
    'axes.facecolor':    '#FAFAFA',
    'figure.facecolor':  WHITE,
    'xtick.labelsize':   15,
    'ytick.labelsize':   15,
    'legend.fontsize':   15,
    'legend.framealpha': 0.92,
    'lines.linewidth':   2.8,
    'lines.markersize':  9,
})

# ── Data ─────────────────────────────────────────────────
df = pd.read_csv('data/data_centre_wue.csv')
FEATURES = ['Cooling_Type_Encoded','Ambient_Temperature_C',
            'Relative_Humidity_Pct','IT_Load_MW',
            'Server_Utilisation_Pct','Climate_Zone_Encoded']
TARGET = 'WUE_Litres_Per_kWh'
y = df[TARGET].values

df['Cooling_Label'] = df['Cooling_Type_Encoded'].map(
    {1:'Air-side',2:'Evaporative',3:'Chilled Water',4:'Seawater',5:'Liquid'})
df['Climate_Label'] = df['Climate_Zone_Encoded'].map(
    {1:'Tropical',2:'Subtropical',3:'Temperate',4:'Continental',
     5:'Oceanic',6:'Subarctic',7:'Mediterranean',8:'Desert'})

X = df[FEATURES].values
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ════════════════════════════════════════════════════════
# FIGURE 1 — EDA (separate panels, large and clear)
# Layout: Row1=violin(full), Row2=temp+utilisation,
#         Row3=climate zone, Row4=correlation matrix
# ════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 30))
fig.suptitle(
    'Predicting Cooling Water Consumption in Data Centres\n'
    'Exploratory Data Analysis',
    fontsize=24, fontweight='bold', color=NAVY,
    y=0.998, x=0.5)

gs = gridspec.GridSpec(4, 2, figure=fig,
    hspace=0.48, wspace=0.35,
    top=0.960, bottom=0.03,
    left=0.08, right=0.97)

# ── A: Violin — full width ────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
order  = ['Air-side','Seawater','Liquid','Chilled Water','Evaporative']
colors = [COOLING_COLORS[c] for c in order]
data   = [df[df['Cooling_Label']==c][TARGET].values for c in order]
bp = ax0.violinplot(data, positions=range(len(order)),
    showmedians=True, showextrema=True, widths=0.72)
for patch, color in zip(bp['bodies'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.80)
bp['cmedians'].set_color(WHITE)
bp['cmedians'].set_linewidth(3)
bp['cmaxes'].set_color(GRAY)
bp['cmins'].set_color(GRAY)
bp['cbars'].set_color(GRAY)
ax0.set_xticks(range(len(order)))
ax0.set_xticklabels(order, fontsize=16, fontweight='bold')
ax0.set_ylabel('Cooling Water Consumption\n(litres / kWh)', fontsize=17)
ax0.set_xlabel('Cooling System Type', fontsize=17)
ax0.set_title('A.  Cooling Water Consumption by Cooling System Type',
    loc='left', fontsize=20, fontweight='bold', color=NAVY)
ax0.axhline(1.8, color=ORANGE, ls='--', lw=2.5, alpha=0.9,
    label='Industry average: 1.8 L/kWh')
ax0.axhline(0.5, color=GREEN, ls='--', lw=2.5, alpha=0.9,
    label='Best practice target: 0.5 L/kWh')
ax0.legend(fontsize=15, loc='upper left',
    framealpha=0.92, edgecolor='#CCCCCC')
ax0.set_facecolor('#F8FAFD')

# ── B: Temperature vs consumption ────────────────────
ax1 = fig.add_subplot(gs[1, 0])
for cooling, color in COOLING_COLORS.items():
    mask = df['Cooling_Label'] == cooling
    ax1.scatter(df[mask]['Ambient_Temperature_C'],
        df[mask][TARGET],
        c=color, alpha=0.55, s=45, label=cooling,
        edgecolors='none')
r_temp = np.corrcoef(df['Ambient_Temperature_C'], y)[0,1]
ax1.set_xlabel('Ambient Temperature (°C)', fontsize=17)
ax1.set_ylabel('Cooling Water\nConsumption (L/kWh)', fontsize=17)
ax1.set_title(f'B.  Temperature vs Consumption\n(r = {r_temp:+.3f})',
    loc='left', fontsize=18, fontweight='bold', color=NAVY)
ax1.legend(fontsize=13, markerscale=1.8,
    framealpha=0.92, edgecolor='#CCCCCC')

# ── C: Utilisation vs consumption ────────────────────
ax2 = fig.add_subplot(gs[1, 1])
ax2.scatter(df['Server_Utilisation_Pct'], y,
    c=df['Cooling_Type_Encoded'],
    cmap='tab10', alpha=0.50, s=45, edgecolors='none')
r_util = np.corrcoef(df['Server_Utilisation_Pct'], y)[0,1]
ax2.set_xlabel('Server Utilisation (%)', fontsize=17)
ax2.set_ylabel('Cooling Water\nConsumption (L/kWh)', fontsize=17)
ax2.set_title(f'C.  Server Utilisation vs Consumption\n(r = {r_util:+.3f})',
    loc='left', fontsize=18, fontweight='bold', color=NAVY)

# ── D: Climate zone boxplot — full width ─────────────
ax3 = fig.add_subplot(gs[2, :])
climate_order = ['Tropical','Desert','Subtropical',
    'Mediterranean','Temperate','Oceanic','Continental','Subarctic']
climate_data = [df[df['Climate_Label']==c][TARGET].dropna().values
    for c in climate_order]
bp2 = ax3.boxplot(climate_data, patch_artist=True,
    widths=0.55,
    medianprops={'color':WHITE,'linewidth':3},
    whiskerprops={'color':GRAY,'linewidth':1.8},
    capprops={'color':GRAY,'linewidth':1.8},
    flierprops={'marker':'o','markersize':5,
        'alpha':0.35,'markeredgewidth':0})
palette = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(climate_order)))
for patch, color in zip(bp2['boxes'], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax3.set_xticks(range(1, len(climate_order)+1))
ax3.set_xticklabels(climate_order, fontsize=15, fontweight='bold')
ax3.set_ylabel('Cooling Water Consumption\n(litres / kWh)', fontsize=17)
ax3.set_title('D.  Consumption Distribution by Climate Zone',
    loc='left', fontsize=20, fontweight='bold', color=NAVY)
ax3.axhline(0.5, color=GREEN, ls='--', lw=2.0, alpha=0.7,
    label='Best practice: 0.5 L/kWh')
ax3.legend(fontsize=14)

# ── E: Correlation matrix — full width ───────────────
ax4 = fig.add_subplot(gs[3, :])
col_labels = [
    'Cooling\nSystem', 'Ambient\nTemp (°C)',
    'Humidity\n(%)', 'IT Load\n(MW)',
    'Utilisation\n(%)', 'Climate\nZone',
    'Consumption\n(L/kWh)']
cols   = FEATURES + [TARGET]
corr_m = df[cols].corr()
cbar_kw = {'label':'Pearson r','shrink':0.55,
           'format':'%.2f'}
hm = sns.heatmap(corr_m, annot=True, fmt='.2f',
    cmap='RdYlBu_r', center=0, ax=ax4,
    square=False, linewidths=2.0, linecolor='white',
    annot_kws={'size':16, 'weight':'bold'},
    xticklabels=col_labels,
    yticklabels=col_labels,
    cbar_kws=cbar_kw)
ax4.set_title('E.  Feature Correlation Matrix (Pearson r)',
    loc='left', fontsize=20, fontweight='bold', color=NAVY)
ax4.tick_params(axis='both', labelsize=15)
# Make colorbar label larger
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Pearson r', fontsize=15)

plt.savefig('outputs/fig1_eda.png',
    dpi=180, bbox_inches='tight', facecolor=WHITE)
plt.close()
print("✓ Figure 1: EDA (22x30, large fonts, 4-row layout)")


# ════════════════════════════════════════════════════════
# FIGURE 2 — Model Comparison (tall bars, large labels)
# ════════════════════════════════════════════════════════
models_dict = {
    'KNN\n(K=5)': Pipeline([('s',StandardScaler()),
        ('m',KNeighborsRegressor(n_neighbors=5,weights='distance'))]),
    'Linear\nRegression': Pipeline([('s',StandardScaler()),
        ('m',LinearRegression())]),
    'Ridge\nRegression': Pipeline([('s',StandardScaler()),
        ('m',Ridge(alpha=1.0))]),
    'Random\nForest': Pipeline([('s',StandardScaler()),
        ('m',RandomForestRegressor(n_estimators=100,random_state=42))]),
}
results = {}
for name, pipe in models_dict.items():
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    results[name] = {
        'R2':   r2_score(y_test, yp),
        'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
        'MAE':  mean_absolute_error(y_test, yp),
    }

fig, axes = plt.subplots(1, 3, figsize=(22, 9))
fig.suptitle('Model Comparison: KNN vs Alternative Approaches',
    fontsize=24, fontweight='bold', color=NAVY, y=1.02)

names  = list(results.keys())
r2s    = [results[n]['R2']   for n in names]
rmses  = [results[n]['RMSE'] for n in names]
maes   = [results[n]['MAE']  for n in names]
bar_c  = [BLUE if 'KNN' in n else '#D0D0D0' for n in names]
edge_c = [NAVY if 'KNN' in n else GRAY      for n in names]

for ax, vals, title, unit, better in [
    (axes[0], r2s,   'R² Score',      '',        'higher is better ↑'),
    (axes[1], rmses, 'RMSE',          ' (L/kWh)','lower is better ↓'),
    (axes[2], maes,  'Mean Absolute\nError', ' (L/kWh)','lower is better ↓'),
]:
    bars = ax.bar(names, vals, color=bar_c,
        edgecolor=edge_c, linewidth=2.0, width=0.52)
    best = max(vals) if 'R²' in title else min(vals)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+max(vals)*0.028,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=16, fontweight='bold',
            color=NAVY if val==best else GRAY)
    ax.set_title(f'{title}\n({better})',
        fontsize=18, fontweight='bold', color=NAVY)
    ax.set_ylabel(f'{title}{unit}', fontsize=16)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, max(vals)*1.22)
    bars[0].set_edgecolor(NAVY)
    bars[0].set_linewidth(3.0)

plt.tight_layout()
plt.savefig('outputs/fig2_model_comparison.png',
    dpi=180, bbox_inches='tight', facecolor=WHITE)
plt.close()
print("✓ Figure 2: Model comparison (22x9)")


# ════════════════════════════════════════════════════════
# FIGURE 3 — GridSearchCV (large, clear elbow)
# ════════════════════════════════════════════════════════
param_grid = {
    'model__n_neighbors': list(range(1, 31)),
    'model__weights':     ['uniform','distance'],
    'model__metric':      ['euclidean','manhattan'],
}
knn_pipe = Pipeline([('scaler',StandardScaler()),
    ('model',KNeighborsRegressor(algorithm='ball_tree'))])
gs_cv = GridSearchCV(knn_pipe, param_grid,
    cv=KFold(n_splits=5,shuffle=True,random_state=42),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, refit=True, verbose=0)
gs_cv.fit(X_train, y_train)

best_k = gs_cv.best_params_['model__n_neighbors']
best_w = gs_cv.best_params_['model__weights']
best_m = gs_cv.best_params_['model__metric']

cv_df   = pd.DataFrame(gs_cv.cv_results_)
cv_best = cv_df[
    (cv_df['param_model__weights']==best_w) &
    (cv_df['param_model__metric']==best_m)
].sort_values('param_model__n_neighbors')

ks      = cv_best['param_model__n_neighbors'].values
rmse_cv = -cv_best['mean_test_score'].values
std_cv  =  cv_best['std_test_score'].values

r2_vals = []
for k in ks:
    pp = Pipeline([('s',StandardScaler()),
        ('m',KNeighborsRegressor(n_neighbors=int(k),
            weights=best_w, metric=best_m))])
    pp.fit(X_train, y_train)
    r2_vals.append(r2_score(y_test, pp.predict(X_test)))

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle(
    f'Hyperparameter Tuning — GridSearchCV (5-Fold CV)\n'
    f'Best: K={best_k}, weights={best_w}, metric={best_m}',
    fontsize=22, fontweight='bold', color=NAVY, y=1.03)

ax1 = axes[0]
ax1.plot(ks, rmse_cv, color=BLUE, lw=3.0, marker='o',
    ms=8, label='Mean CV-RMSE (5-Fold)')
ax1.fill_between(ks, rmse_cv-std_cv, rmse_cv+std_cv,
    alpha=0.20, color=BLUE, label='±1 Std Dev')
ax1.axvline(best_k, color=RED, ls='--', lw=3.0,
    label=f'Optimal K = {best_k}')
ax1.scatter([best_k],[rmse_cv[list(ks).index(best_k)]],
    color=RED, s=200, zorder=6,
    edgecolors=WHITE, linewidths=2.0)
ax1.set_xlabel('Number of Neighbours (K)', fontsize=17)
ax1.set_ylabel('Cross-Validation RMSE (L/kWh)', fontsize=17)
ax1.set_title('A.  RMSE vs K — Elbow Plot',
    loc='left', fontsize=20, fontweight='bold', color=NAVY)
ax1.legend(fontsize=15)
ax1.tick_params(labelsize=15)
ax1.set_xticks(range(0, 32, 5))

ax2 = axes[1]
ax2.plot(ks, r2_vals, color=GREEN, lw=3.0, marker='s',
    ms=8, label='Test Set R²')
ax2.axvline(best_k, color=RED, ls='--', lw=3.0,
    label=f'Optimal K = {best_k}')
ax2.scatter([best_k],[r2_vals[list(ks).index(best_k)]],
    color=RED, s=200, zorder=6,
    edgecolors=WHITE, linewidths=2.0)
ax2.set_xlabel('Number of Neighbours (K)', fontsize=17)
ax2.set_ylabel('R² Score', fontsize=17)
ax2.set_title('B.  R² Score vs K',
    loc='left', fontsize=20, fontweight='bold', color=NAVY)
ax2.legend(fontsize=15)
ax2.tick_params(labelsize=15)
ax2.set_xticks(range(0, 32, 5))

plt.tight_layout()
plt.savefig('outputs/fig3_gridsearchcv.png',
    dpi=180, bbox_inches='tight', facecolor=WHITE)
plt.close()
print("✓ Figure 3: GridSearchCV (22x9)")


# ════════════════════════════════════════════════════════
# FIGURE 4 — Final Results (2x2, large)
# ════════════════════════════════════════════════════════
best_model = gs_cv.best_estimator_
y_pred     = best_model.predict(X_test)
residuals  = y_test - y_pred

import joblib
joblib.dump(best_model,
    'models/knn_wue_model.pkl')

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
fig.suptitle(
    f'Model Evaluation Results — KNN Regressor\n'
    f'K={best_k}  |  weights={best_w}  |  metric={best_m}  |  R²={r2_score(y_test,y_pred):.4f}',
    fontsize=22, fontweight='bold', color=NAVY, y=1.01)

# A: Actual vs Predicted
ax1 = axes[0, 0]
sc = ax1.scatter(y_test, y_pred,
    c=residuals, cmap='RdYlBu',
    alpha=0.65, s=80, edgecolors='none')
cb = plt.colorbar(sc, ax=ax1, shrink=0.85)
cb.set_label('Residual (L/kWh)', fontsize=15)
cb.ax.tick_params(labelsize=13)
mn = min(y_test.min(), y_pred.min())-0.05
mx = max(y_test.max(), y_pred.max())+0.05
ax1.plot([mn,mx],[mn,mx], color=NAVY, lw=2.5, ls='--',
    label='Perfect prediction (y = x)')
ax1.set_xlabel('Actual Consumption (L/kWh)', fontsize=17)
ax1.set_ylabel('Predicted Consumption (L/kWh)', fontsize=17)
ax1.set_title(f'A.  Actual vs Predicted\nR² = {r2_score(y_test,y_pred):.4f}',
    loc='left', fontsize=18, fontweight='bold', color=NAVY)
ax1.legend(fontsize=14)
ax1.tick_params(labelsize=15)

# B: Residuals vs Predicted
ax2 = axes[0, 1]
ax2.scatter(y_pred, residuals, c=residuals,
    cmap='RdYlBu', alpha=0.65, s=80, edgecolors='none')
ax2.axhline(0, color=NAVY, lw=2.5, ls='--',
    label='Zero residual line')
ax2.axhline(residuals.mean(), color=ORANGE, lw=2.5,
    label=f'Mean residual: {residuals.mean():.4f}')
ax2.set_xlabel('Predicted Consumption (L/kWh)', fontsize=17)
ax2.set_ylabel('Residual (L/kWh)', fontsize=17)
ax2.set_title('B.  Residual Plot\n(No systematic pattern = no bias)',
    loc='left', fontsize=18, fontweight='bold', color=NAVY)
ax2.legend(fontsize=14)
ax2.tick_params(labelsize=15)

# C: Residual distribution
ax3 = axes[1, 0]
ax3.hist(residuals, bins=32, color=BLUE,
    alpha=0.82, edgecolor=WHITE, linewidth=0.8)
ax3.axvline(0, color=NAVY, lw=2.5, ls='--',
    label='Zero residual')
ax3.axvline(residuals.mean(), color=ORANGE, lw=2.5,
    label=f'Mean = {residuals.mean():.4f}')
ax3.axvline(np.median(residuals), color=GREEN, lw=2.5,
    label=f'Median = {np.median(residuals):.4f}')
ax3.set_xlabel('Residual (L/kWh)', fontsize=17)
ax3.set_ylabel('Frequency', fontsize=17)
ax3.set_title('C.  Residual Distribution',
    loc='left', fontsize=18, fontweight='bold', color=NAVY)
ax3.legend(fontsize=14)
ax3.tick_params(labelsize=15)

# D: Feature importance
ax4 = axes[1, 1]
feat_labels = ['Cooling\nSystem','Ambient\nTemp','Relative\nHumidity',
    'IT Load\n(MW)','Server\nUtilisation','Climate\nZone']
importances = [abs(np.corrcoef(df[f],y)[0,1]) for f in FEATURES]
sorted_idx  = np.argsort(importances)
sorted_imp  = [importances[i] for i in sorted_idx]
sorted_lab  = [feat_labels[i] for i in sorted_idx]
bar_colors  = [BLUE if i==sorted_idx[-1] else LBLUE
    for i in range(len(FEATURES))]
bar_colors  = [bar_colors[i] for i in sorted_idx]

bars = ax4.barh(sorted_lab, sorted_imp,
    color=bar_colors, edgecolor=WHITE,
    height=0.60, linewidth=0.8)
for bar, val in zip(bars, sorted_imp):
    ax4.text(val+0.005,
        bar.get_y()+bar.get_height()/2,
        f'{val:.3f}', va='center',
        fontsize=14, fontweight='bold', color=NAVY)
ax4.set_xlabel('|Pearson Correlation| with Target', fontsize=17)
ax4.set_title('D.  Feature Importance\n(Absolute Pearson Correlation)',
    loc='left', fontsize=18, fontweight='bold', color=NAVY)
ax4.set_xlim(0, max(importances)+0.12)
ax4.tick_params(labelsize=15)

plt.tight_layout(h_pad=3.5, w_pad=3.0)
plt.savefig('outputs/fig4_results.png',
    dpi=180, bbox_inches='tight', facecolor=WHITE)
plt.close()
print("✓ Figure 4: Final results (22x18)")


# ════════════════════════════════════════════════════════
# FIGURE 5 — Singapore Use Case
# ════════════════════════════════════════════════════════
knn_model = best_model.named_steps['model']
sc_model  = best_model.named_steps['scaler']

designs = [
    ('Air-side Cooling',         [1, 28, 82, 100, 80, 1]),
    ('Evaporative Cooling',      [2, 28, 82, 100, 80, 1]),
    ('Chilled Water (Standard)', [3, 28, 82, 100, 80, 1]),
    ('Liquid Cooling',           [5, 28, 82, 100, 80, 1]),
    ('Liquid (60% utilisation)', [5, 28, 82, 100, 60, 1]),
]
wues = []
lpds = []
for name, feat in designs:
    fs  = sc_model.transform(np.array([feat]))
    wue = knn_model.predict(fs)[0]
    wues.append(wue)
    lpds.append(wue * 100_000 * 24)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle(
    'Singapore Data Centre Design Analysis\n'
    'Conditions: 28°C  |  82% Humidity  |  100 MW IT Load  |  80% Utilisation',
    fontsize=22, fontweight='bold', color=NAVY, y=1.03)

dnames    = [d[0] for d in designs]
bar_clrs  = [GREEN if w<0.5 else ORANGE if w<1.0 else RED for w in wues]

ax1 = axes[0]
bars = ax1.barh(dnames, wues, color=bar_clrs,
    edgecolor=WHITE, height=0.52, linewidth=0.8)
ax1.axvline(0.4, color=GREEN, ls='--', lw=2.5,
    label='BCA Green Mark target: 0.4 L/kWh')
ax1.axvline(1.8, color=ORANGE, ls='--', lw=2.0, alpha=0.8,
    label='Industry average: 1.8 L/kWh')
for bar, val in zip(bars, wues):
    ax1.text(val+0.008,
        bar.get_y()+bar.get_height()/2,
        f'{val:.3f} L/kWh', va='center',
        fontsize=15, fontweight='bold', color=NAVY)
ax1.set_xlabel('Predicted Cooling Water Consumption\n(litres per kWh)',
    fontsize=17)
ax1.set_title('A.  Predicted Consumption by Cooling Design',
    loc='left', fontsize=20, fontweight='bold', color=NAVY)
ax1.legend(fontsize=15, loc='lower right',
    framealpha=0.92, edgecolor='#CCCCCC')
ax1.set_xlim(0, max(wues)*1.30)
ax1.tick_params(labelsize=15)

ax2 = axes[1]
bars2 = ax2.barh(dnames, [l/1e6 for l in lpds],
    color=bar_clrs, edgecolor=WHITE,
    height=0.52, linewidth=0.8)
for bar, val in zip(bars2, lpds):
    ax2.text(val/1e6+0.003,
        bar.get_y()+bar.get_height()/2,
        f'{val/1e6:.2f}M L/day', va='center',
        fontsize=15, fontweight='bold', color=NAVY)
ax2.set_xlabel('Projected Daily Water Consumption\n(million litres)',
    fontsize=17)
ax2.set_title('B.  Projected Daily Water Consumption',
    loc='left', fontsize=20, fontweight='bold', color=NAVY)

green_p  = mpatches.Patch(color=GREEN,  label='Meets BCA target  (< 0.5 L/kWh)')
orange_p = mpatches.Patch(color=ORANGE, label='Borderline  (0.5 – 1.0 L/kWh)')
red_p    = mpatches.Patch(color=RED,    label='Above target  (> 1.0 L/kWh)')
ax2.legend(handles=[green_p,orange_p,red_p],
    fontsize=14, loc='lower right',
    framealpha=0.92, edgecolor='#CCCCCC')
ax2.tick_params(labelsize=15)

plt.tight_layout()
plt.savefig('outputs/fig5_singapore.png',
    dpi=180, bbox_inches='tight', facecolor=WHITE)
plt.close()
print("✓ Figure 5: Singapore use case (22x10)")

print("\n✓ All 5 figures done — large fonts, no wasted space")
print(f"  Best K={best_k}, weights={best_w}, metric={best_m}")
print(f"  R²={r2_score(y_test,y_pred):.4f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_test,y_pred)):.4f}")
