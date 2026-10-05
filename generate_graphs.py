import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

out_dir = r"c:\Users\G15\Downloads\DAA_project\results_v2"
os.makedirs(out_dir, exist_ok=True)

# 1. Model Summary Graph
try:
    df_models = pd.read_csv(os.path.join(out_dir, "model_summary_v2.csv"))
    model_col = df_models.columns[0]
    df_models = df_models.sort_values('R2', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_models, x='R2', y=model_col, palette='viridis')
    plt.title("Model Performance (R² Score)")
    plt.xlabel("R² Score")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph_model_performance.png"))
    plt.close()
    print("Generated graph_model_performance.png")
except Exception as e:
    print(f"Error generating model graph: {e}")

# 2. SHAP Importance Graph
try:
    df_shap = pd.read_csv(os.path.join(out_dir, "shap_importance_v2.csv")).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_shap, x='mean_abs_shap', y='feature', palette='mako')
    plt.title("Top 10 Feature Importances (SHAP)")
    plt.xlabel("Mean |SHAP| Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph_shap_importance.png"))
    plt.close()
    print("Generated graph_shap_importance.png")
except Exception as e:
    print(f"Error generating SHAP graph: {e}")

# 3. Quantum Results Graph
try:
    df_q = pd.read_csv(os.path.join(out_dir, "quantum_results_v2.csv"))
    df_q = df_q.sort_values('Spearman', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_q, x='Spearman', y='Method', palette='rocket')
    plt.title("Quantum vs Classical Models (Spearman ρ)")
    plt.xlabel("Spearman Correlation (ρ)")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph_quantum_performance.png"))
    plt.close()
    print("Generated graph_quantum_performance.png")
except Exception as e:
    print(f"Error generating quantum graph: {e}")

# 4. Wheat Rankings Graph (Top 10 Composite Scores)
try:
    df_wheat = pd.read_csv(os.path.join(out_dir, "wheat_rankings_v2.csv")).head(10)
    df_wheat['Entry'] = df_wheat['Entry'].astype(str)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_wheat, x='composite_score', y='Entry', palette='crest')
    plt.title("Top 10 Wheat Genotypes (Composite Score)")
    plt.xlabel("Composite Score")
    plt.ylabel("Genotype Entry")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph_wheat_top10.png"))
    plt.close()
    print("Generated graph_wheat_top10.png")
except Exception as e:
    print(f"Error generating wheat rankings graph: {e}")

print("All graphs generated successfully!")
