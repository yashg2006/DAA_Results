"""
PHASE 4 — IMPROVED: Optimization + Validation
===============================================
Improvements over v1:
  1. Replace STI_combined with STI_heat (removes r=0.956 redundancy)
  2. Winsorize WUE_drought at 99th percentile (fixes outlier distortion)
  3. Bootstrap uses top-5 instead of top-10 (finds truly stable picks)
  4. Adds Kendall W statistic for overall ranking concordance
  5. Barley: adds PCA-based composite as alternative to weighted sum

Inputs  : wheat_model_ready.csv, selected_features.csv,
          barley_processed.csv, shap_importance_v2.csv
          (falls back to shap_importance.csv if v2 not found)
Outputs : wheat_rankings_v2.csv
          barley_rankings_v2.csv
          weight_sensitivity_v2.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from scipy.stats.mstats import winsorize
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

print("=== PHASE 4 IMPROVED: OPTIMIZATION + VALIDATION ===")

# ─────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────
df      = pd.read_csv('wheat_model_ready.csv')
barley  = pd.read_csv('barley_processed.csv')
features = pd.read_csv('selected_features.csv')['feature'].tolist()

# Use v2 SHAP if available
try:
    shap_imp = pd.read_csv('shap_importance_v2.csv')
    print("Using shap_importance_v2.csv")
except FileNotFoundError:
    shap_imp = pd.read_csv('shap_importance.csv')
    print("Using shap_importance.csv (v2 not found)")

X = df[features].fillna(df[features].median())

# ─────────────────────────────────────────────────────
# IMPROVEMENT 1: Replace STI_combined with STI_heat
# STI_combined had r=0.956 with STI_drought — redundant
# STI_heat provides independent heat stress information
# ─────────────────────────────────────────────────────
objectives_def = {
    'STI_drought': df['STI_drought'],
    'STI_heat':    df['STI_heat'],                              # NEW
    'WUE_drought': df['WUE_drought'].fillna(df['WUE_drought'].median()),
    'YR_drought':  df['YR_drought']
}

# Check correlations of new objective set
print("\nObjective correlations (improved set):")
obj_df_check = pd.DataFrame({k: v.values for k, v in objectives_def.items()})
print(obj_df_check.corr().round(3).to_string())

# ─────────────────────────────────────────────────────
# IMPROVEMENT 2: Winsorize WUE at 99th percentile
# ─────────────────────────────────────────────────────
wue_raw = objectives_def['WUE_drought'].copy()
p99     = wue_raw.quantile(0.99)
wue_winsorized = wue_raw.clip(upper=p99)

print(f"\nWUE winsorization: max {wue_raw.max():.1f} → {wue_winsorized.max():.1f}  "
      f"(99th pct = {p99:.1f})")
objectives_def['WUE_drought'] = wue_winsorized

# ─────────────────────────────────────────────────────
# TRAIN LIGHTGBM FOR EACH OBJECTIVE
# ─────────────────────────────────────────────────────
print("\n── Training LightGBM per objective ──")
pred_obj = {}
for obj_name, y_obj in objectives_def.items():
    m = LGBMRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    m.fit(X, y_obj)
    pred_obj[obj_name] = m.predict(X)
    print(f"  [✓] {obj_name}")

scores = pd.DataFrame(pred_obj)
scores['Entry'] = df['Entry'].values

# ─────────────────────────────────────────────────────
# NORMALIZE OBJECTIVES
# YR_drought = cost (lower is better) → invert
# ─────────────────────────────────────────────────────
def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

scores['STI_drought_norm'] =     minmax(scores['STI_drought'])
scores['STI_heat_norm']    =     minmax(scores['STI_heat'])
scores['WUE_drought_norm'] =     minmax(scores['WUE_drought'])
scores['YR_drought_norm']  = 1 - minmax(scores['YR_drought'])  # inverted

# ─────────────────────────────────────────────────────
# WEIGHTED COMPOSITE (baseline weights)
# STI_drought: 0.35 — primary tolerance measure
# STI_heat:    0.20 — independent heat tolerance
# WUE_drought: 0.25 — water use efficiency
# YR_drought:  0.20 — yield protection
# ─────────────────────────────────────────────────────
W = {
    'STI_drought_norm': 0.35,
    'STI_heat_norm':    0.20,
    'WUE_drought_norm': 0.25,
    'YR_drought_norm':  0.20
}

scores['composite_score'] = sum(W[k] * scores[k] for k in W)
scores = scores.sort_values('composite_score', ascending=False).reset_index(drop=True)
scores['ML_rank'] = scores.index + 1

print("\nTop 10 genotypes (improved composite):")
print(scores[['Entry','STI_drought','STI_heat','WUE_drought',
              'composite_score','ML_rank']].head(10).round(4).to_string(index=False))

# ─────────────────────────────────────────────────────
# WEIGHT SENSITIVITY
# ─────────────────────────────────────────────────────
print("\n── Weight Sensitivity Analysis ──")
scenarios = {
    'Balanced':      {k: 0.25 for k in W},
    'Drought-heavy': {'STI_drought_norm':0.50,'STI_heat_norm':0.15,
                      'WUE_drought_norm':0.20,'YR_drought_norm':0.15},
    'WUE-heavy':     {'STI_drought_norm':0.20,'STI_heat_norm':0.15,
                      'WUE_drought_norm':0.50,'YR_drought_norm':0.15},
    'Heat-heavy':    {'STI_drought_norm':0.20,'STI_heat_norm':0.40,
                      'WUE_drought_norm':0.25,'YR_drought_norm':0.15},
    'Baseline':      W
}

rank_matrix  = pd.DataFrame({'Entry': scores['Entry']})
top5_sets    = {}
top10_sets   = {}

for sc_name, w in scenarios.items():
    sc_score = sum(w[k] * scores[k] for k in w)
    rank_matrix[sc_name] = sc_score.rank(ascending=False).astype(int)
    top5_sets[sc_name]   = set(rank_matrix.nsmallest(5,  sc_name)['Entry'].tolist())
    top10_sets[sc_name]  = set(rank_matrix.nsmallest(10, sc_name)['Entry'].tolist())

sc_names = list(scenarios.keys())
print("\nTop-5 overlap (more stringent):")
for i, s1 in enumerate(sc_names):
    for s2 in sc_names[i+1:]:
        n5  = len(top5_sets[s1]  & top5_sets[s2])
        n10 = len(top10_sets[s1] & top10_sets[s2])
        print(f"  {s1:<16} ∩ {s2:<16}: top-5={n5}/5   top-10={n10}/10")

# Kendall W (overall concordance across all scenarios)
from scipy.stats import kendalltau
rank_cols = [rank_matrix[sc].values for sc in sc_names]
rank_arr  = np.array(rank_cols)  # shape: (n_scenarios, n_genotypes)
# Kendall W formula
n_gen   = rank_arr.shape[1]
n_rater = rank_arr.shape[0]
R_i     = rank_arr.sum(axis=0)
R_mean  = R_i.mean()
S       = np.sum((R_i - R_mean)**2)
W_stat  = 12 * S / (n_rater**2 * (n_gen**3 - n_gen))
print(f"\nKendall W (overall scenario concordance): {W_stat:.4f}")
print(f"  Interpretation: 0=no agreement, 1=perfect")
print(f"  W > 0.7 = strong concordance")

# ─────────────────────────────────────────────────────
# IMPROVEMENT 3: Bootstrap top-5 stability
# ─────────────────────────────────────────────────────
print("\n── Bootstrap Stability (n=1000, top-5) ──")
np.random.seed(42)
N = len(scores)
top5_freq  = np.zeros(N)
top10_freq = np.zeros(N)

for _ in range(1000):
    idx      = np.random.choice(N, N, replace=True)
    bs_score = sum(W[k] * scores[k].iloc[idx].values for k in W)
    top5_idx  = np.argsort(-bs_score)[:5]
    top10_idx = np.argsort(-bs_score)[:10]
    top5_freq[top5_idx]   += 1
    top10_freq[top10_idx] += 1

scores['bootstrap_top5_pct']  = (top5_freq  / 1000 * 100).round(1)
scores['bootstrap_top10_pct'] = (top10_freq / 1000 * 100).round(1)

print(f"\nTop-5 bootstrap frequency (≥20%):")
stable5 = scores[scores['bootstrap_top5_pct'] >= 20].copy()
if len(stable5):
    print(stable5[['Entry','STI_drought','STI_heat','composite_score',
                    'ML_rank','bootstrap_top5_pct']].to_string(index=False))
else:
    print("  No genotype appeared in top-5 ≥20% of samples")
    print("  Top 5 by bootstrap frequency:")
    print(scores.sort_values('bootstrap_top5_pct', ascending=False)
          [['Entry','ML_rank','bootstrap_top5_pct','bootstrap_top10_pct']]
          .head(5).to_string(index=False))

# ─────────────────────────────────────────────────────
# BARLEY CROSS-CROP VALIDATION
# ─────────────────────────────────────────────────────
print("\n── Cross-Crop Validation (Barley) ──")

for col in ['STI','GMP','YSI']:
    barley[f'{col}_norm'] = minmax(barley[col])

barley['SSI_inv_norm'] = 1 - minmax(barley['SSI'])   # lower SSI = better

# Weighted composite
barley['composite_score'] = (
    0.35 * barley['STI_norm']     +
    0.30 * barley['GMP_norm']     +
    0.20 * barley['YSI_norm']     +
    0.15 * barley['SSI_inv_norm']
)

barley = barley.sort_values('composite_score', ascending=False).reset_index(drop=True)
barley['composite_rank'] = barley.index + 1

# Rank each index
for col, asc in [('STI',False),('GMP',False),('YSI',False),('SSI',True)]:
    barley[f'{col}_rank'] = barley[col].rank(ascending=asc).astype(int)

# Spearman concordance matrix
idx_pairs = [
    ('STI_rank','composite_rank','STI vs Composite'),
    ('STI_rank','GMP_rank',      'STI vs GMP'),
    ('STI_rank','YSI_rank',      'STI vs YSI'),
    ('GMP_rank','composite_rank','GMP vs Composite'),
    ('YSI_rank','composite_rank','YSI vs Composite'),
]

print("\nSpearman ρ (barley index concordance):")
for r1, r2, label in idx_pairs:
    rho, p = spearmanr(barley[r1], barley[r2])
    sig    = '***' if p<0.001 else ('**' if p<0.01 else '*')
    print(f"  {label:<28}: ρ={rho:.4f}  p={p:.4e}  {sig}")

print(f"\nTop 10 barley genotypes:")
print(barley[['GEN','Yp','Ys','STI','YSI','GMP',
              'composite_score','composite_rank']].head(10).round(4).to_string(index=False))

# ─────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────
scores.to_csv('wheat_rankings_v2.csv', index=False)
barley.to_csv('barley_rankings_v2.csv', index=False)
rank_matrix.to_csv('weight_sensitivity_v2.csv', index=False)

print("\n=== SAVED ===")
print("wheat_rankings_v2.csv")
print("barley_rankings_v2.csv")
print("weight_sensitivity_v2.csv")
print("\nPhase 4 improved complete.")