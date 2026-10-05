import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

out_dir = r"c:\Users\G15\Downloads\DAA_project\results_v2"
os.makedirs(out_dir, exist_ok=True)

# 1. Predicted vs Actual Scatter Plot (Stacking)
try:
    df_preds = pd.read_csv(os.path.join(out_dir, "model_predictions_v2.csv"))
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='STI_drought', y='pred_Stacking', data=df_preds, alpha=0.6, color='blue')
    
    # Perfect fit line
    min_val = min(df_preds['STI_drought'].min(), df_preds['pred_Stacking'].min())
    max_val = max(df_preds['STI_drought'].max(), df_preds['pred_Stacking'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit (y=x)')
    
    plt.title("Stacking Ensemble: Actual vs Predicted STI_drought")
    plt.xlabel("Actual STI_drought")
    plt.ylabel("Predicted STI_drought")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph_actual_vs_predicted.png"))
    plt.close()
    print("Generated graph_actual_vs_predicted.png")
except Exception as e:
    print(f"Error: {e}")

# 2. Residual Distribution Plot
try:
    df_preds['Residuals_Stacking'] = df_preds['STI_drought'] - df_preds['pred_Stacking']
    df_preds['Residuals_XGBoost'] = df_preds['STI_drought'] - df_preds['pred_XGBoost_tuned']
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_preds['Residuals_Stacking'], label='Stacking', fill=True, color='blue')
    sns.kdeplot(df_preds['Residuals_XGBoost'], label='XGBoost', fill=True, color='orange')
    
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Error Residuals Distribution (Actual - Predicted)")
    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph_residual_distribution.png"))
    plt.close()
    print("Generated graph_residual_distribution.png")
except Exception as e:
    print(f"Error: {e}")

print("Scatter plot and residual distribution plot generated!")
