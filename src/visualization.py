"""
Visualization Module for Cryptocurrency Risk Analysis
CS2235 - Module 4 & 5: Visualizing Clusters, PCA, and the Risk Frontier
Strictly follows syllabus: K-Means clusters, PCA 2D scatter, Elbow & Silhouette plots.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

OUTPUT_DIR = "docs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIER_COLORS = {
    "Ultra-Low Risk": "#2ca02c",       # Green
    "Low-Moderate Risk": "#1f77b4",   # Blue
    "Moderate-High Risk": "#ff7f0e",  # Orange
    "Extreme Risk": "#d62728"         # Red
}


def plot_volatility_vs_volume(df, save_path=os.path.join(OUTPUT_DIR, "01_volatility_vs_volume.png")):
    """Plots Annualized Volatility vs. Log Trading Volume (Core Problem Statement)."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    for tier, color in TIER_COLORS.items():
        subset = df[df["Risk_Tier"] == tier]
        ax.scatter(subset["Log_Avg_Volume_USD"], subset["Annualized_Volatility_%"],
                   c=color, label=tier, s=120, edgecolors="black", linewidths=0.8, alpha=0.9)
                   
    for _, row in df.iterrows():
        ax.annotate(row["Symbol"], 
                    (row["Log_Avg_Volume_USD"], row["Annualized_Volatility_%"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8, fontweight="bold", alpha=0.85)
                    
    ax.set_title("Cryptocurrency Risk Grouping: Volatility vs. Trading Volume Depth", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Log10 Average Daily Trading Volume (USD) [Trading Activity]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annualized Price Volatility (%) [Price Fluctuation]", fontsize=11, fontweight="bold")
    ax.legend(title="Risk Category", frameon=True, facecolor="white", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_feature_correlation(df, save_path=os.path.join(OUTPUT_DIR, "03_feature_correlation_heatmap.png")):
    """Plots correlation matrix of engineered features (Module 4)."""
    feature_cols = [
        "Annualized_Volatility_%", "Max_Drawdown_%", "Mean_Daily_Spread_%", 
        "Log_Avg_Volume_USD", "Vol_to_Volume_Ratio"
    ]
    
    corr = df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                square=True, linewidths=0.8, cbar_kws={"shrink": 0.8}, ax=ax)
                
    ax.set_title("Feature Correlation Matrix (CS2235 Module 4)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_optimal_k_metrics(metrics_json_path="data/models/metrics_summary.json", 
                           save_path=os.path.join(OUTPUT_DIR, "04_optimal_k_evaluation.png")):
    """Plots Elbow curve (WCSS / Inertia) and Silhouette score across k (Module 4)."""
    if not os.path.exists(metrics_json_path):
        return
        
    with open(metrics_json_path, "r") as f:
        summary = json.load(f)
        
    k_eval = summary.get("k_evaluation", {})
    k_vals = k_eval.get("k", [])
    inertias = k_eval.get("inertia", [])
    silhouettes = k_eval.get("silhouette", [])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # 1. Elbow Method (Inertia / WCSS)
    ax1.plot(k_vals, inertias, marker="o", color="#1f77b4", linewidth=2.5, markersize=7)
    ax1.set_title("Elbow Method: Within-Cluster Sum of Squares (Inertia)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Inertia (WCSS)", fontsize=10, fontweight="bold")
    ax1.axvline(4, color="red", linestyle="--", label="Elbow at k=4")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # 2. Silhouette Score
    ax2.plot(k_vals, silhouettes, marker="s", color="#2ca02c", linewidth=2.5, markersize=7, label="Silhouette Score")
    ax2.set_title("Model Evaluation: Silhouette Score vs. k", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Silhouette Score (Higher is better)", fontsize=10, fontweight="bold")
    ax2.axvline(4, color="red", linestyle="--", label="Optimal k=4 (Peak = 0.3868)")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_pca_clusters(df, save_path=os.path.join(OUTPUT_DIR, "05_pca_clusters_2d.png")):
    """Plots 2D Principal Component Analysis (PCA) projection (Module 4)."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    for tier, color in TIER_COLORS.items():
        subset = df[df["Risk_Tier"] == tier]
        ax.scatter(subset["PCA1"], subset["PCA2"], c=color, label=tier,
                   s=130, edgecolors="black", linewidths=0.9, alpha=0.9)
                   
    for _, row in df.iterrows():
        ax.annotate(row["Symbol"], (row["PCA1"], row["PCA2"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8, fontweight="bold", alpha=0.85)
                    
    ax.set_title("2D Principal Component Analysis (PCA) of K-Means Clusters", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Principal Component 1 (Price Volatility & Drawdown - 79.8% Variance)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Principal Component 2 (Trading Volume Depth - 17.1% Variance)", fontsize=10, fontweight="bold")
    ax.legend(title="Risk Category", frameon=True, facecolor="white", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_cluster_summary_bars(df, save_path=os.path.join(OUTPUT_DIR, "07_cluster_risk_comparison.png")):
    """Simple bar chart comparing mean Volatility and Volume across the 4 clusters."""
    metrics = ["Annualized_Volatility_%", "Max_Drawdown_%", "Log_Avg_Volume_USD"]
    metric_labels = ["Mean Volatility (%)", "Mean Max Drawdown (%)", "Log10 Daily Volume"]
    
    tier_means = df.groupby("Risk_Tier")[metrics].mean().reindex([
        "Ultra-Low Risk", "Low-Moderate Risk", "Moderate-High Risk", "Extreme Risk"
    ])
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        bars = ax.bar(tier_means.index, tier_means[metric], color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Tier 1\n(Ultra-Low)", "Tier 2\n(Low-Mod)", "Tier 3\n(Mod-High)", "Tier 4\n(Extreme)"], 
                           rotation=0, fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                        
    fig.suptitle("Comparison of Risk and Volume Across Discovered Categories", fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def generate_all_visualizations(clustered_csv="data/processed/crypto_clustered_results.csv"):
    if not os.path.exists(clustered_csv):
        raise FileNotFoundError(f"Clustered results not found: {clustered_csv}")
        
    df = pd.read_csv(clustered_csv)
    print("[*] Generating syllabus-aligned visual figures (Modules 4 & 5)...")
    plot_volatility_vs_volume(df)
    plot_feature_correlation(df)
    plot_optimal_k_metrics()
    plot_pca_clusters(df)
    plot_cluster_summary_bars(df)
    print(f"[+] All figures successfully generated and saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_visualizations()
