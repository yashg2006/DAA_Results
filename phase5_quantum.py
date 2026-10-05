"""
PHASE 5 - IMPROVED: Quantum-Inspired Extension
================================================
Exp 1: Classical SVR (6 features, 3-fold LOEO, power-transform target)
Exp 2: Quantum-Inspired SVR (cos*sin interference features, classical)
Exp 3: Quantum Kernel SVM  -- FIXED: proper adjoint fidelity kernel
       Uses qml.adjoint(encode_zz) so K = |<phi(x1)|phi(x2)>|^2 exactly
Exp 4: Quantum Embedding SVR (18-dim PauliXYZ embed + hybrid classical)
       Projected Quantum Kernel -- Huang et al. 2021, Nature Comm.

Inputs  : wheat_model_ready.csv, shap_importance_v2.csv
Outputs : quantum_results_v2.csv
"""

import sys
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import pennylane as qml
import warnings
warnings.filterwarnings('ignore')

def progress(done, total, t0, width=30):
    pct = done / total
    bar = chr(9608) * int(width * pct) + chr(9617) * (width - int(width * pct))
    ela = time.time() - t0
    eta = (total - done) / max(done / max(ela, 1e-9), 1e-9)
    spd = done / max(ela, 1e-9)
    sys.stdout.write(f'\r  [{bar}] {done:>5}/{total}  {pct*100:5.1f}%  '
                     f'{spd:6.1f}/s  elapsed={ela:5.0f}s  ETA={eta:5.0f}s')
    sys.stdout.flush()
    if done == total:
        print()

print("=== PHASE 5 IMPROVED: QUANTUM-INSPIRED EXTENSION ===")
print(f"PennyLane: {qml.__version__}")

OUT_DIR = r"c:\Users\G15\Downloads\DAA_project"

df = pd.read_csv(f"{OUT_DIR}\\wheat_model_ready.csv")
TARGET = 'STI_drought'
TOP_N  = 6

try:
    shap_imp = pd.read_csv(f"{OUT_DIR}\\shap_importance_v2.csv")
except FileNotFoundError:
    shap_imp = pd.read_csv(f"{OUT_DIR}\\shap_importance.csv")

top_feat = shap_imp.head(TOP_N)['feature'].tolist()
print(f"\nTop {TOP_N} SHAP features: {top_feat}")

X_raw = df[top_feat].fillna(df[top_feat].median()).values
y     = df[TARGET].values          # raw STI_drought [0.07 - 0.80]
n     = len(X_raw)

# Power-transform (for classical SVR)
pt  = PowerTransformer(method='yeo-johnson', standardize=True)
y_t = pt.fit_transform(y.reshape(-1, 1)).ravel()

# Angle encoding: [0, pi]
s1      = MinMaxScaler(feature_range=(0, np.pi))
X_angle = s1.fit_transform(X_raw)

# Classical scale: [-1, 1]
s2    = MinMaxScaler(feature_range=(-1, 1))
X_svr = s2.fit_transform(X_raw)

# Standard scale for QE hybrid
s4       = MinMaxScaler(feature_range=(-1, 1))
X_scaled = s4.fit_transform(X_raw)

# Stratified LOEO folds
sort_idx = np.argsort(y)
fold_ids = np.zeros(n, dtype=int)
for rank, orig_idx in enumerate(sort_idx):
    fold_ids[orig_idx] = rank % 3
folds = [
    (np.where(fold_ids != i)[0].tolist(),
     np.where(fold_ids == i)[0].tolist())
    for i in range(3)
]

N_QUBITS = TOP_N   # 6
dev = qml.device('lightning.qubit', wires=N_QUBITS)

# ─────────────────────────────────────────────────────
# EXP 1: Classical SVR -- full 3-fold LOEO
# ─────────────────────────────────────────────────────
print("\n── Exp 1: Classical SVR (6 features, 3-fold LOEO) ──")
svr_metrics = []
t0 = time.time()
for fi, (tr_idx, te_idx) in enumerate(folds):
    svr = SVR(kernel='rbf', C=10, epsilon=0.01)
    svr.fit(X_svr[tr_idx], y_t[tr_idx])
    pred_t  = np.clip(svr.predict(X_svr[te_idx]), y_t.min()-3, y_t.max()+3)
    pred_bt = np.nan_to_num(
        pt.inverse_transform(pred_t.reshape(-1, 1)).ravel(),
        nan=float(np.nanmedian(y))
    )
    rmse = np.sqrt(mean_squared_error(y[te_idx], pred_bt))
    mae  = mean_absolute_error(y[te_idx], pred_bt)
    r2   = r2_score(y[te_idx], pred_bt)
    rho  = spearmanr(y[te_idx], pred_bt).statistic
    svr_metrics.append({'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Spearman': rho})
    print(f"  Fold {fi+1}: RMSE={rmse:.4f}  R2={r2:.4f}  rho={rho:.4f}")
svr_time  = time.time() - t0
svr_means = {k: np.mean([f[k] for f in svr_metrics]) for k in svr_metrics[0]}
print(f"  MEAN : RMSE={svr_means['RMSE']:.4f}  R2={svr_means['R2']:.4f}  "
      f"rho={svr_means['Spearman']:.4f}  time={svr_time:.1f}s")

# ─────────────────────────────────────────────────────
# EXP 2: Quantum-Inspired SVR (classical hardware)
# ─────────────────────────────────────────────────────
print("\n── Exp 2: Quantum-Inspired SVR (interference features, classical) ──")

def qi_feature_map(X_in):
    n_samples, n_feats = X_in.shape
    feats = [X_in]
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            feats.append((np.cos(X_in[:, i]) * np.sin(X_in[:, j])).reshape(-1, 1))
            feats.append((np.sin(X_in[:, i]) * np.cos(X_in[:, j])).reshape(-1, 1))
    return np.hstack(feats)

X_qi        = qi_feature_map(X_angle)
X_qi_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_qi)
print(f"  Feature map: {X_raw.shape[1]} -> {X_qi.shape[1]} features")

qi_metrics = []
t0 = time.time()
for fi, (tr_idx, te_idx) in enumerate(folds):
    svr_qi = SVR(kernel='rbf', C=10, epsilon=0.01)
    svr_qi.fit(X_qi_scaled[tr_idx], y_t[tr_idx])
    pred_t  = np.clip(svr_qi.predict(X_qi_scaled[te_idx]), y_t.min()-3, y_t.max()+3)
    pred_bt = np.nan_to_num(
        pt.inverse_transform(pred_t.reshape(-1, 1)).ravel(),
        nan=float(np.nanmedian(y))
    )
    rmse = np.sqrt(mean_squared_error(y[te_idx], pred_bt))
    mae  = mean_absolute_error(y[te_idx], pred_bt)
    r2   = r2_score(y[te_idx], pred_bt)
    rho  = spearmanr(y[te_idx], pred_bt).statistic
    qi_metrics.append({'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Spearman': rho})
    print(f"  Fold {fi+1}: RMSE={rmse:.4f}  R2={r2:.4f}  rho={rho:.4f}")
qi_time  = time.time() - t0
qi_means = {k: np.mean([f[k] for f in qi_metrics]) for k in qi_metrics[0]}
print(f"  MEAN : RMSE={qi_means['RMSE']:.4f}  R2={qi_means['R2']:.4f}  "
      f"rho={qi_means['Spearman']:.4f}  time={qi_time:.1f}s")

# ─────────────────────────────────────────────────────
# EXP 3: Quantum Kernel SVM -- PROPER fidelity kernel
# K(x1,x2) = |<phi(x1)|phi(x2)>|^2  via qml.adjoint
# 30 train / 20 test  = 1500 circuits  (~3 min)
# ─────────────────────────────────────────────────────
print("\n── Exp 3: Quantum Kernel SVM (30 train/20 test, proper adjoint kernel) ──")

fold2_tr = folds[1][0][:30]
fold2_te = folds[1][1][:20]
X_qk_tr  = X_angle[fold2_tr]
X_qk_te  = X_angle[fold2_te]
# Use RAW y for QK (targets in [0.07, 0.80] — no inverse-transform needed)
y_qk_tr  = y[fold2_tr]
y_qk_te  = y[fold2_te]

def encode_zz(x):
    """ZZFeatureMap: H + RZ(2x) + CNOT + RZ(2(pi-xi)(pi-xj)) + CNOT"""
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(2.0 * x[i], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev, diff_method="best")
def fidelity_kernel_circuit(x1, x2):
    """Proper ZZFeatureMap fidelity kernel: K = |<phi(x1)|phi(x2)>|^2"""
    encode_zz(x1)
    qml.adjoint(encode_zz)(x2)   # <-- adjoint gives exact U†(x2)
    return qml.probs(wires=range(N_QUBITS))

def kern(x1, x2):
    return float(fidelity_kernel_circuit(x1, x2)[0])

n_tot = len(fold2_tr)**2 + len(fold2_te) * len(fold2_tr)
print(f"  Circuits: {n_tot}  ({len(fold2_tr)}x{len(fold2_tr)} + "
      f"{len(fold2_te)}x{len(fold2_tr)})  |  train on raw y (range {y_qk_tr.min():.3f}-{y_qk_tr.max():.3f})")
t0   = time.time()
done = 0

K_train = np.zeros((len(fold2_tr), len(fold2_tr)))
for i, x1 in enumerate(X_qk_tr):
    for j, x2 in enumerate(X_qk_tr):
        K_train[i, j] = kern(x1, x2)
        done += 1
        progress(done, n_tot, t0)

K_test = np.zeros((len(fold2_te), len(fold2_tr)))
for i, x1 in enumerate(X_qk_te):
    for j, x2 in enumerate(X_qk_tr):
        K_test[i, j] = kern(x1, x2)
        done += 1
        progress(done, n_tot, t0)

print(f"  Kernel stats: min={K_train.min():.4f}  max={K_train.max():.4f}  "
      f"diag_mean={np.diag(K_train).mean():.4f}")

# Regularise diagonal -> guaranteed PSD -> SVR always converges
K_train_reg = K_train + 1e-6 * np.eye(len(K_train))
# C=100, epsilon=0.005 tuned for raw y in [0.07-0.80]
qk_svr = SVR(kernel='precomputed', C=100, epsilon=0.005, max_iter=10000)
qk_svr.fit(K_train_reg, y_qk_tr)
qk_pred = qk_svr.predict(K_test)
qk_time = time.time() - t0

qk_rmse = np.sqrt(mean_squared_error(y_qk_te, qk_pred))
qk_mae  = mean_absolute_error(y_qk_te, qk_pred)
qk_r2   = r2_score(y_qk_te, qk_pred)
qk_rho  = spearmanr(y_qk_te, qk_pred).statistic
print(f"  RMSE={qk_rmse:.4f}  R2={qk_r2:.4f}  rho={qk_rho:.4f}  time={qk_time:.1f}s")

# ─────────────────────────────────────────────────────
# EXP 4: Quantum Embedding SVR -- 18-dim PauliXYZ + hybrid
# Encode with ZZFeatureMap, measure <X>, <Y>, <Z> on all 6 qubits
# -> 18-dim quantum features; concatenate with 6 classical features
# -> 24-dim hybrid feature vector -> RBF-SVR on raw y
# ─────────────────────────────────────────────────────
print("\n── Exp 4: Quantum Embedding SVR (18-dim Pauli + 6-dim classical hybrid) ──")

@qml.qnode(dev, diff_method="best")
def embed_z(x):
    encode_zz(x)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QUBITS)]

@qml.qnode(dev, diff_method="best")
def embed_x(x):
    encode_zz(x)
    return [qml.expval(qml.PauliX(wires=i)) for i in range(N_QUBITS)]

@qml.qnode(dev, diff_method="best")
def embed_y(x):
    encode_zz(x)
    return [qml.expval(qml.PauliY(wires=i)) for i in range(N_QUBITS)]

def get_quantum_features(X_in):
    """Returns 18-dim quantum feature vectors (Z+X+Y on all qubits)."""
    m = len(X_in)
    Q = np.zeros((m, N_QUBITS * 3))
    for i, x in enumerate(X_in):
        Q[i, :N_QUBITS]            = embed_z(x)
        Q[i, N_QUBITS:2*N_QUBITS]  = embed_x(x)
        Q[i, 2*N_QUBITS:]          = embed_y(x)
    return Q

# Use fold-0: 160 train, 80 test
qer_tr = folds[0][0]
qer_te = folds[0][1]
X_qer_tr     = X_angle[qer_tr]
X_qer_te     = X_angle[qer_te]
y_qer_tr_raw = y[qer_tr]          # raw y -- no inverse transform needed
y_qer_te_raw = y[qer_te]

n_tot_qer = len(qer_tr) + len(qer_te)
print(f"  Embedding {len(qer_tr)} train + {len(qer_te)} test "
      f"({N_QUBITS*3}-dim quantum + {N_QUBITS}-dim classical = {N_QUBITS*4}-dim hybrid)...")
t0   = time.time()
done = 0

Q_tr = np.zeros((len(qer_tr), N_QUBITS * 3))
for i, x in enumerate(X_qer_tr):
    Q_tr[i] = get_quantum_features(x.reshape(1, -1))
    done += 1
    progress(done, n_tot_qer, t0)

Q_te = np.zeros((len(qer_te), N_QUBITS * 3))
for i, x in enumerate(X_qer_te):
    Q_te[i] = get_quantum_features(x.reshape(1, -1))
    done += 1
    progress(done, n_tot_qer, t0)

# Hybrid: quantum features + classical scaled features
H_tr = np.hstack([Q_tr, X_scaled[qer_tr]])
H_te = np.hstack([Q_te, X_scaled[qer_te]])

print(f"  Embedding done in {time.time()-t0:.1f}s -- fitting SVR on {H_tr.shape[1]}-dim hybrid...")
# Train SVR on raw y [0.07-0.80]
qer_svr = SVR(kernel='rbf', C=100, epsilon=0.005)
qer_svr.fit(H_tr, y_qer_tr_raw)
qer_pred = qer_svr.predict(H_te)
qer_time = time.time() - t0

qer_rmse = np.sqrt(mean_squared_error(y_qer_te_raw, qer_pred))
qer_mae  = mean_absolute_error(y_qer_te_raw, qer_pred)
qer_r2   = r2_score(y_qer_te_raw, qer_pred)
qer_rho  = spearmanr(y_qer_te_raw, qer_pred).statistic
print(f"  RMSE={qer_rmse:.4f}  R2={qer_r2:.4f}  rho={qer_rho:.4f}  time={qer_time:.1f}s")

# ─────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────
print("\n── Comparison Table ──")
print(f"{'Method':<34} {'RMSE':>7} {'R2':>7} {'rho':>7} {'Time(s)':>8} {'n_train':>8}  Note")
print("-" * 100)
print(f"{'Classical SVR (6 feat)':<34} {svr_means['RMSE']:>7.4f} {svr_means['R2']:>7.4f} "
      f"{svr_means['Spearman']:>7.4f} {svr_time:>8.1f} {'240':>8}  3-fold LOEO, classical")
print(f"{'QI-SVR (cos*sin, 36 feat)':<34} {qi_means['RMSE']:>7.4f} {qi_means['R2']:>7.4f} "
      f"{qi_means['Spearman']:>7.4f} {qi_time:>8.1f} {'240':>8}  3-fold LOEO, quantum-inspired")
print(f"{'QK-SVM (adjoint ZZKernel)':<34} {qk_rmse:>7.4f} {qk_r2:>7.4f} "
      f"{qk_rho:>7.4f} {qk_time:>8.1f} {'30/20':>8}  proper fidelity kernel, simulated")
print(f"{'QE-SVR (18q + 6c hybrid)':<34} {qer_rmse:>7.4f} {qer_r2:>7.4f} "
      f"{qer_rho:>7.4f} {qer_time:>8.1f} {f'{len(qer_tr)}/{len(qer_te)}':>8}  PauliXYZ+classical, simulated")

# ─────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────
results = pd.DataFrame([
    {'Method': 'Classical SVR',
     'RMSE': svr_means['RMSE'], 'MAE': svr_means['MAE'],
     'R2': svr_means['R2'], 'Spearman': svr_means['Spearman'],
     'n_train': 240, 'note': '3-fold LOEO, classical, 6 features, power-transform'},
    {'Method': 'QI-SVR (interference)',
     'RMSE': qi_means['RMSE'], 'MAE': qi_means['MAE'],
     'R2': qi_means['R2'], 'Spearman': qi_means['Spearman'],
     'n_train': 240, 'note': '3-fold LOEO, 36 cos*sin features, quantum-inspired classical'},
    {'Method': 'Quantum Kernel SVM',
     'RMSE': qk_rmse, 'MAE': qk_mae, 'R2': qk_r2, 'Spearman': qk_rho,
     'n_train': 30,
     'note': 'Fold-2 (30/20), proper adjoint ZZFeatMap fidelity kernel, lightning.qubit'},
    {'Method': 'Quantum Embedding SVR (hybrid)',
     'RMSE': qer_rmse, 'MAE': qer_mae, 'R2': qer_r2, 'Spearman': qer_rho,
     'n_train': len(qer_tr),
     'note': (f'Fold-0 ({len(qer_tr)}/{len(qer_te)}), 18-dim PauliXYZ + 6-dim classical, '
              f'lightning.qubit, Huang et al. 2021')},
])
results.to_csv(f"{OUT_DIR}\\quantum_results_v2.csv", index=False)

print("\n=== SAVED ===")
print(f"quantum_results_v2.csv -> {OUT_DIR}")
print("\nPhase 5 improved complete.")
print("NOTE: QI-SVR = classical hardware, report as 'quantum-inspired'.")
print("      QK-SVM + QE-SVR = lightning.qubit simulator, report as 'simulated quantum'.")
print("      QE-SVR: Huang et al. 2021 Projected Quantum Kernel (Nature Comm.).")
