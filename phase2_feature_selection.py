"""
PHASE 2: Feature Selection
==========================
Methods:
  1. Correlation filter    — remove one from each pair with r > 0.85
  2. LASSO (ElasticNetCV)  — penalized regression coefficients
  3. Random Forest         — impurity-based importance
  4. Final set             — union of top 20 from LASSO + RF

Input  : wheat_processed.csv
Outputs: selected_features.csv   (union set  — used for ML)
         core_features.csv       (intersection — most stable features)
         wheat_model_ready.csv   (X + targets combined, ready for Phase 3)

Requirements:
  pip install scikit-learn xgboost lightgbm
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────
OUT_DIR = r"c:\Users\G15\Downloads\DAA_project"

# ─────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────
df = pd.read_csv(f"{OUT_DIR}\\wheat_processed.csv")
print("=== PHASE 2: FEATURE SELECTION ===")
print(f"Loaded wheat_processed.csv: {df.shape}")

# ─────────────────────────────────────────────────────
# DEFINE TARGETS & CANDIDATE FEATURES
# Target for selection stage: STI_drought
# Exclude identifiers, all target columns, and index
# columns derived from yield (data leakage)
# ─────────────────────────────────────────────────────
targets = [
    'STI_drought', 'STI_heat', 'STI_combined',
    'Grain Yield_gm2_Drought', 'Grain Yield_gm2_Terminal heat',
    'YR_drought', 'YR_heat'
]

leakage_cols = [
    'SSI_drought', 'GMP_drought', 'MP_drought', 'TOL_drought',
    'YSI_drought', 'YI_drought',
    'SSI_heat',   'GMP_heat',   'MP_heat',   'TOL_heat',
    'YSI_heat',   'YI_heat',
    'Grain Yield_gm2_Irrigation'   # yield potential = direct leakage
]

exclude = ['Entry', 'CIMMYT ID', 'Cross Name'] + targets + leakage_cols

feature_cols = [c for c in df.columns if c not in exclude]
print(f"\nCandidate features before filtering : {len(feature_cols)}")

X_raw = df[feature_cols].copy()
y     = df['STI_drought'].copy()

# ─────────────────────────────────────────────────────
# STEP 1: Drop columns with >20% missing, fill rest
# ─────────────────────────────────────────────────────
X_raw = X_raw.dropna(axis=1, thresh=int(0.8 * len(X_raw)))
X_raw = X_raw.fillna(X_raw.median())
print(f"After NaN filter                    : {X_raw.shape[1]}")

# ─────────────────────────────────────────────────────
# STEP 2: Correlation filter  (r > 0.85)
# Keep the first column of each correlated pair
# ─────────────────────────────────────────────────────
corr_matrix = X_raw.corr().abs()
upper       = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop    = [col for col in upper.columns if any(upper[col] > 0.85)]
X_filt     = X_raw.drop(columns=to_drop)

print(f"After correlation filter (r>0.85)   : {X_filt.shape[1]} "
      f"({len(to_drop)} dropped)")

# ─────────────────────────────────────────────────────
# STEP 3: LASSO / ElasticNet importance
# ─────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_filt)

enet = ElasticNetCV(cv=5, max_iter=10000, random_state=42, n_jobs=-1)
enet.fit(X_scaled, y)

enet_coef    = pd.Series(np.abs(enet.coef_), index=X_filt.columns)
enet_selected = enet_coef[enet_coef > 0].sort_values(ascending=False)

print(f"\nElasticNet non-zero coefficients    : {len(enet_selected)}")
print("\nTop 15 by |coefficient|:")
print(enet_selected.head(15).round(5).to_string())

# ─────────────────────────────────────────────────────
# STEP 4: Random Forest importance
# ─────────────────────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_leaf=3,
    random_state=42, n_jobs=-1
)
rf.fit(X_filt, y)

rf_imp = pd.Series(rf.feature_importances_, index=X_filt.columns)
rf_top = rf_imp.sort_values(ascending=False)

print(f"\nTop 15 by Random Forest importance:")
print(rf_top.head(15).round(5).to_string())

# ─────────────────────────────────────────────────────
# STEP 5: Build final feature sets
# ─────────────────────────────────────────────────────
enet_top20 = set(enet_selected.head(20).index)
rf_top20   = set(rf_top.head(20).index)

union_set  = enet_top20.union(rf_top20)
inter_set  = enet_top20.intersection(rf_top20)

print(f"\nElasticNet top 20 : {len(enet_top20)}")
print(f"RF top 20         : {len(rf_top20)}")
print(f"Intersection      : {len(inter_set)}")
print(f"Union             : {len(union_set)}")

print(f"\nCore features (intersection — most stable):")
for f in sorted(inter_set):
    print(f"  {f}")

print(f"\nFull selected features (union — used for ML):")
for f in sorted(union_set):
    print(f"  {f}")

# ─────────────────────────────────────────────────────
# STEP 6: Save outputs
# ─────────────────────────────────────────────────────
final_features = sorted(union_set)
core_features  = sorted(inter_set)

pd.Series(final_features).to_csv(
    f"{OUT_DIR}\\selected_features.csv", index=False, header=['feature']
)
pd.Series(core_features).to_csv(
    f"{OUT_DIR}\\core_features.csv", index=False, header=['feature']
)

# Model-ready file: selected features + all target columns
keep_cols = ['Entry'] + final_features + targets
df_model  = df[keep_cols].copy()
df_model.to_csv(f"{OUT_DIR}\\wheat_model_ready.csv", index=False)

print("\n=== SAVED ===")
print(f"selected_features.csv  : {len(final_features)} features (union)")
print(f"core_features.csv      : {len(core_features)} features (intersection)")
print(f"wheat_model_ready.csv  : {df_model.shape}")
print("\nPhase 2 complete. Run phase3_ml_models.py next.")
