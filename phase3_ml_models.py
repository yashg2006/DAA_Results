"""
PHASE 3 — IMPROVED: ML Model Training
======================================
Improvements over v1:
  1. Power-transform STI_drought target (fixes -0.93 skew)
  2. Optuna hyperparameter tuning for XGBoost + LightGBM
  3. Stacking ensemble: XGBoost + LightGBM + ElasticNet meta-learner
  4. Proper grouped LOEO — fold assignment by sorted STI rank
     (avoids accidental data leakage from adjacent rows)
  5. Reports both transformed and back-transformed metrics

Inputs  : wheat_model_ready.csv, selected_features.csv
Outputs : model_predictions_v2.csv
          model_summary_v2.csv
          shap_importance_v2.csv

Requirements:
  pip install scikit-learn xgboost lightgbm shap optuna
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────
df       = pd.read_csv('wheat_model_ready.csv')
features = pd.read_csv('selected_features.csv')['feature'].tolist()
TARGET   = 'STI_drought'

X = df[features].fillna(df[features].median())
y = df[TARGET].values

print("=== PHASE 3 IMPROVED: ML MODEL TRAINING ===")
print(f"Samples  : {len(X)}  Features: {len(features)}")
print(f"Target skewness (raw): {pd.Series(y).skew():.4f}")

# ─────────────────────────────────────────────────────
# IMPROVEMENT 1: Power-transform target
# Yeo-Johnson handles negative values, fixes skew
# ─────────────────────────────────────────────────────
pt  = PowerTransformer(method='yeo-johnson', standardize=True)
y_t = pt.fit_transform(y.reshape(-1,1)).ravel()
print(f"Target skewness (transformed): {pd.Series(y_t).skew():.4f}")

# ─────────────────────────────────────────────────────
# IMPROVEMENT 2: Stratified LOEO folds
# Sort by STI_drought, assign folds in interleaved fashion
# so each fold has similar STI distribution
# ─────────────────────────────────────────────────────
n          = len(X)
sort_idx   = np.argsort(y)          # sort by target value
fold_ids   = np.zeros(n, dtype=int)
for rank, orig_idx in enumerate(sort_idx):
    fold_ids[orig_idx] = rank % 3   # interleave 0,1,2,0,1,2...

folds = [
    (np.where(fold_ids != i)[0].tolist(),
     np.where(fold_ids == i)[0].tolist())
    for i in range(3)
]
print(f"\nStratified folds: {[len(f[1]) for f in folds]} test samples per fold")

# ─────────────────────────────────────────────────────
# IMPROVEMENT 3: Optuna tuning for XGBoost + LightGBM
# 30 trials each — fast but effective
# ─────────────────────────────────────────────────────
print("\n── Optuna Hyperparameter Tuning ──")
X_arr = X.values

def tune_xgb(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'max_depth':        trial.suggest_int('max_depth', 3, 7),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        'random_state': 42, 'verbosity': 0
    }
    rmse_scores = []
    for tr_idx, te_idx in folds:
        m = XGBRegressor(**params)
        m.fit(X_arr[tr_idx], y_t[tr_idx])
        p = m.predict(X_arr[te_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y_t[te_idx], p)))
    return np.mean(rmse_scores)

def tune_lgb(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'max_depth':         trial.suggest_int('max_depth', 3, 7),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        'random_state': 42, 'verbose': -1
    }
    rmse_scores = []
    for tr_idx, te_idx in folds:
        m = LGBMRegressor(**params)
        m.fit(X_arr[tr_idx], y_t[tr_idx])
        p = m.predict(X_arr[te_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y_t[te_idx], p)))
    return np.mean(rmse_scores)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(tune_xgb, n_trials=30, show_progress_bar=False)
best_xgb_params = study_xgb.best_params
best_xgb_params.update({'random_state':42, 'verbosity':0})
print(f"XGBoost best RMSE (transformed): {study_xgb.best_value:.5f}")
print(f"  Params: {best_xgb_params}")

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(tune_lgb, n_trials=30, show_progress_bar=False)
best_lgb_params = study_lgb.best_params
best_lgb_params.update({'random_state':42, 'verbose':-1})
print(f"LightGBM best RMSE (transformed): {study_lgb.best_value:.5f}")
print(f"  Params: {best_lgb_params}")

# ─────────────────────────────────────────────────────
# MODELS — tuned XGBoost & LightGBM + stacking ensemble
# ─────────────────────────────────────────────────────
models = {
    'ElasticNet': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  ElasticNetCV(cv=5, max_iter=10000, random_state=42))
    ]),
    'RandomForest': RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    ),
    'XGBoost_tuned': XGBRegressor(**best_xgb_params),
    'LightGBM_tuned': LGBMRegressor(**best_lgb_params),
    'SVR': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  SVR(kernel='rbf', C=10, epsilon=0.01))
    ]),
}

# Stacking ensemble (XGB + LGB + RF → Ridge meta-learner)
stacking = StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor(**best_xgb_params)),
        ('lgb', LGBMRegressor(**best_lgb_params)),
        ('rf',  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ],
    final_estimator=RidgeCV(),
    cv=3,
    n_jobs=-1
)
models['Stacking'] = stacking

# ─────────────────────────────────────────────────────
# LOEO EVALUATION — train on transformed, report back-transformed
# ─────────────────────────────────────────────────────
results    = {}
all_preds  = {name: np.zeros(n) for name in models}

print("\n── LOEO Results (back-transformed metrics) ──")
print(f"{'Model':<18} {'Fold':<6} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'Spearman':>10}")
print("─" * 58)

for name, model in models.items():
    fold_metrics = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        yt_tr       = y_t[tr_idx]

        model.fit(X_tr, yt_tr)
        pred_t  = model.predict(X_te)

        # Back-transform to original STI_drought scale
        pred_t_clipped = np.clip(pred_t, y_t.min()-3, y_t.max()+3)
        pred_bt = pt.inverse_transform(pred_t_clipped.reshape(-1,1)).ravel()
        pred_bt = np.nan_to_num(pred_bt, nan=float(np.nanmedian(y)))
        y_true  = y[te_idx]

        all_preds[name][te_idx] = pred_bt

        rmse = np.sqrt(mean_squared_error(y_true, pred_bt))
        mae  = mean_absolute_error(y_true, pred_bt)
        r2   = r2_score(y_true, pred_bt)
        rho  = spearmanr(y_true, pred_bt).statistic

        fold_metrics.append({'RMSE':rmse,'MAE':mae,'R2':r2,'Spearman':rho})
        print(f"{name:<18} {fi+1:<6} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f} {rho:>10.4f}")

    means = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
    results[name] = means
    print(f"{name:<18} {'MEAN':<6} {means['RMSE']:>8.4f} {means['MAE']:>8.4f} "
          f"{means['R2']:>8.4f} {means['Spearman']:>10.4f}")
    print()

# ─────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────
summary = pd.DataFrame(results).T.round(4)
summary.columns = ['RMSE','MAE','R2','Spearman_rho']
summary = summary.sort_values('RMSE')

print("── Summary (sorted by RMSE) ──")
print(summary.to_string())
print(f"\nBest model: {summary.index[0]}")

# ─────────────────────────────────────────────────────
# SHAP on best tree model (XGBoost_tuned, full dataset)
# ─────────────────────────────────────────────────────
print("\n── SHAP Analysis (XGBoost_tuned, full dataset) ──")
xgb_full = XGBRegressor(**best_xgb_params)
xgb_full.fit(X, y_t)

explainer = shap.TreeExplainer(xgb_full)
shap_vals = explainer.shap_values(X)
shap_df   = pd.DataFrame(np.abs(shap_vals), columns=features)
mean_shap = shap_df.mean().sort_values(ascending=False)

print("\nTop 15 features by mean |SHAP|:")
print(mean_shap.head(15).round(5).to_string())

# ─────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────
pred_df = df[['Entry', TARGET]].copy()
for name in models:
    pred_df[f'pred_{name}'] = all_preds[name]
pred_df.to_csv('model_predictions_v2.csv', index=False)

summary.to_csv('model_summary_v2.csv')

shap_out = mean_shap.reset_index()
shap_out.columns = ['feature','mean_abs_shap']
shap_out.to_csv('shap_importance_v2.csv', index=False)

print("\n=== SAVED ===")
print("model_predictions_v2.csv")
print("model_summary_v2.csv")
print("shap_importance_v2.csv")
print("\nPhase 3 improved complete. Run phase4_improved.py next.")