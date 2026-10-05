"""
PHASE 1: Data Preparation & Stress Index Computation
=====================================================
Inputs:
  - Garcia-Barrios et al. 2025 (Wheat, CIMMYT) — BLUEs sheet
  - Mokarroma & Alam 2024 (Barley, Bangladesh)

Outputs:
  - wheat_processed.csv
  - barley_processed.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
# PATHS — corrected to match actual local file paths
# ─────────────────────────────────────────────────────
WHEAT_PATH  = r"c:\Users\G15\Downloads\DAA_project\Dataset1\Garcia-Barrios_et_al_2025_TPG_raw_BLUEs_datasets.xlsx"
BARLEY_PATH = r"c:\Users\G15\Downloads\DAA_project\Selection of drought tolerant barley genotypes using stress indices\Raw data.xlsx"

# Output directory (same folder as this script)
OUT_DIR = r"c:\Users\G15\Downloads\DAA_project"

# ─────────────────────────────────────────────────────
# STEP 1: Load wheat BLUEs
# ─────────────────────────────────────────────────────
blues = pd.read_excel(WHEAT_PATH, sheet_name='BLUEs')

print("=== WHEAT DATASET ===")
print(f"Shape      : {blues.shape}")
print(f"Environments: {blues['Environment'].unique().tolist()}")
print(f"Genotypes  : {blues['Entry'].nunique()}")
print(f"Missing    : {blues.isnull().sum().sum()} total NaN cells")

# ─────────────────────────────────────────────────────
# STEP 2: Pivot — one row per genotype, env as suffix
# ─────────────────────────────────────────────────────
id_cols   = ['Environment', 'Entry', 'CIMMYT ID', 'Cross Name']
trait_cols = [c for c in blues.columns if c not in id_cols]

pivot = blues.pivot_table(index='Entry', columns='Environment', values=trait_cols)
pivot.columns = ['_'.join(col).strip() for col in pivot.columns]
pivot = pivot.reset_index()

print(f"\nAfter pivot: {pivot.shape[0]} genotypes x {pivot.shape[1]} columns")

# ─────────────────────────────────────────────────────
# STEP 3: Stress index computation
# Formulas (Fischer & Maurer 1978; Rosielle & Hamblin 1981)
#   Yp = yield under irrigation (potential)
#   Ys = yield under stress (drought or heat)
#   D  = 1 - (mean_Ys / mean_Yp)   [stress intensity]
# ─────────────────────────────────────────────────────
Yp      = pivot['Grain Yield_gm2_Irrigation']
Ys_d    = pivot['Grain Yield_gm2_Drought']
Ys_h    = pivot['Grain Yield_gm2_Terminal heat']

Yp_mean  = Yp.mean()
Ysd_mean = Ys_d.mean()
Ysh_mean = Ys_h.mean()
D_d      = 1 - (Ysd_mean / Yp_mean)   # drought stress intensity
D_h      = 1 - (Ysh_mean / Yp_mean)   # heat stress intensity

print(f"\nStress intensity D:")
print(f"  Drought       : {D_d:.4f}")
print(f"  Terminal heat : {D_h:.4f}")

# -- Drought indices --
pivot['STI_drought']  = (Yp * Ys_d) / (Yp_mean ** 2)
pivot['SSI_drought']  = (1 - Ys_d / Yp) / D_d
pivot['GMP_drought']  = np.sqrt(Yp * Ys_d)
pivot['MP_drought']   = (Yp + Ys_d) / 2
pivot['TOL_drought']  = Yp - Ys_d
pivot['YSI_drought']  = Ys_d / Yp
pivot['YI_drought']   = Ys_d / Ysd_mean
pivot['YR_drought']   = ((Yp - Ys_d) / Yp) * 100   # % yield reduction

# -- Heat indices --
pivot['STI_heat']     = (Yp * Ys_h) / (Yp_mean ** 2)
pivot['SSI_heat']     = (1 - Ys_h / Yp) / D_h
pivot['GMP_heat']     = np.sqrt(Yp * Ys_h)
pivot['MP_heat']      = (Yp + Ys_h) / 2
pivot['TOL_heat']     = Yp - Ys_h
pivot['YSI_heat']     = Ys_h / Yp
pivot['YI_heat']      = Ys_h / Ysh_mean
pivot['YR_heat']      = ((Yp - Ys_h) / Yp) * 100

# -- Combined (mean of both stresses) --
pivot['STI_combined'] = (pivot['STI_drought'] + pivot['STI_heat']) / 2

print("\nStress index summary (wheat):")
idx_cols = ['STI_drought','SSI_drought','GMP_drought','MP_drought',
            'TOL_drought','YSI_drought','STI_heat','STI_combined']
print(pivot[idx_cols].describe().round(3).to_string())

# ─────────────────────────────────────────────────────
# STEP 4: Feature engineering
# ─────────────────────────────────────────────────────

# Water use efficiency proxies
pivot['WUE_drought']  = Ys_d  / pivot['E_mmolm2s1_Drought'].replace(0, np.nan)
pivot['WUE_irr']      = Yp    / pivot['E_mmolm2s1_Irrigation'].replace(0, np.nan)

# Intrinsic WUE (gs/E) per environment
for env in ['Irrigation', 'Drought', 'Terminal heat']:
    slug = env.replace(' ', '_')
    pivot[f'iWUE_{slug}'] = (
        pivot[f'gs_molm2s1_{env}'] /
        pivot[f'E_mmolm2s1_{env}'].replace(0, np.nan)
    )

# NDVI reduction under stress (vegetative stage)
pivot['dNDVI_drought'] = pivot['NDVI_vg_Irrigation'] - pivot['NDVI_vg_Drought']
pivot['dNDVI_heat']    = pivot['NDVI_vg_Irrigation'] - pivot['NDVI_vg_Terminal heat']

# Photosynthetic efficiency × grain filling rate composite
pivot['PhiGFR_irr']     = pivot['PhiPS2_Irrigation'] * pivot['GFR_gm2_Irrigation']
pivot['PhiGFR_drought'] = pivot['PhiPS2_Drought']    * pivot['GFR_gm2_Drought']
pivot['PhiGFR_heat']    = pivot['PhiPS2_Terminal heat'] * pivot['GFR_gm2_Terminal heat']

print("\nEngineered features added: WUE, iWUE, dNDVI, PhiGFR")

# ─────────────────────────────────────────────────────
# STEP 5: Load and clean barley dataset
# ─────────────────────────────────────────────────────
barley_raw = pd.read_excel(BARLEY_PATH)
barley_raw = barley_raw[['REP','GEN','Yp','Ys','TOL','MP','GMP',
                          'SSI','STI','YSI','YI']].copy()

# Average across 3 reps
barley = barley_raw.groupby('GEN').mean(numeric_only=True).reset_index()

# Recompute YR for consistency with wheat pipeline
barley['YR']           = ((barley['Yp'] - barley['Ys']) / barley['Yp']) * 100
barley['STI_combined'] = barley['STI']   # single-stress dataset
barley['WUE_proxy']    = barley['Ys'] / barley['Yp']  # YSI as proxy

print("\n=== BARLEY DATASET ===")
print(f"Genotypes  : {len(barley)}")
print(f"Missing    : {barley.isnull().sum().sum()} NaN cells")
print(barley[['GEN','Yp','Ys','STI','SSI','YSI','YR']].describe().round(3).to_string())

# ─────────────────────────────────────────────────────
# STEP 6: Save outputs
# ─────────────────────────────────────────────────────
wheat_out  = f"{OUT_DIR}\\wheat_processed.csv"
barley_out = f"{OUT_DIR}\\barley_processed.csv"

pivot.to_csv(wheat_out, index=False)
barley.to_csv(barley_out, index=False)

print("\n=== SAVED ===")
print(f"wheat_processed.csv   : {pivot.shape[0]} genotypes x {pivot.shape[1]} columns")
print(f"barley_processed.csv  : {barley.shape[0]} genotypes x {barley.shape[1]} columns")
print("\nPhase 1 complete. Run phase2_feature_selection.py next.")
