"""
Visualization Module for Cryptocurrency Risk Analysis
Generates high-resolution academic figures for reports, presentations, and notebooks.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Set aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
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
                   
    # Annotate key symbols
    for _, row in df.iterrows():
        ax.annotate(row["Symbol"], 
                    (row["Log_Avg_Volume_USD"], row["Annualized_Volatility_%"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8, fontweight="bold", alpha=0.85)
                    
    ax.set_title("Cryptocurrency Risk Grouping: Volatility vs. Trading Volume Depth", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Log10 Average Daily Trading Volume (USD)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annualized Price Volatility (%)", fontsize=11, fontweight="bold")
    ax.legend(title="Assigned Risk Tier", frameon=True, facecolor="white", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_risk_return(df, save_path=os.path.join(OUTPUT_DIR, "02_risk_vs_return.png")):
    """Plots Annualized Return vs. Annualized Volatility."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    for tier, color in TIER_COLORS.items():
        subset = df[df["Risk_Tier"] == tier]
        ax.scatter(subset["Annualized_Volatility_%"], subset["Annualized_Return_%"],
                   c=color, label=tier, s=120, edgecolors="black", linewidths=0.8, alpha=0.9)
                   
    for _, row in df.iterrows():
        ax.annotate(row["Symbol"], 
                    (row["Annualized_Volatility_%"], row["Annualized_Return_%"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=8, fontweight="bold", alpha=0.85)
                    
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Risk-Return Profile: Annualized Volatility vs. Annualized Return", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Annualized Volatility (%) [Risk Proxy]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annualized Return (%)", fontsize=11, fontweight="bold")
    ax.legend(title="Assigned Risk Tier", frameon=True, facecolor="white", loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_feature_correlation(df, save_path=os.path.join(OUTPUT_DIR, "03_feature_correlation_heatmap.png")):
    """Plots correlation matrix of engineered risk and volume metrics."""
    feature_cols = [
        "Annualized_Volatility_%", "Max_Drawdown_%", "VaR_95_%", "CVaR_95_%",
        "Downside_Deviation_%", "Mean_Daily_Spread_%", "Log_Avg_Volume_USD",
        "Volume_Volatility_CV", "Amihud_Illiquidity", "Vol_to_Volume_Ratio"
    ]
    
    corr = df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                square=True, linewidths=0.8, cbar_kws={"shrink": 0.8}, ax=ax)
                
    ax.set_title("Correlation Matrix of Quantitative Risk & Liquidity Metrics", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_optimal_k_metrics(metrics_json_path="data/models/metrics_summary.json", 
                           save_path=os.path.join(OUTPUT_DIR, "04_optimal_k_evaluation.png")):
    """Plots Elbow curve (Inertia) and Silhouette score comparison across k."""
    if not os.path.exists(metrics_json_path):
        return
        
    with open(metrics_json_path, "r") as f:
        summary = json.load(f)
        
    k_eval = summary.get("k_evaluation", {})
    k_vals = k_eval.get("k", [])
    inertias = k_eval.get("inertia", [])
    silhouettes = k_eval.get("silhouette", [])
    db_indices = k_eval.get("davies_bouldin", [])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # 1. Inertia / Elbow Method
    ax1.plot(k_vals, inertias, marker="o", color="#1f77b4", linewidth=2.5, markersize=7)
    ax1.set_title("Elbow Method: Within-Cluster Sum of Squares (Inertia)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Inertia (WCSS)", fontsize=10, fontweight="bold")
    ax1.axvline(4, color="red", linestyle="--", label="Chosen Optimal k=4")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # 2. Silhouette Score & Davies-Bouldin
    ax2.plot(k_vals, silhouettes, marker="s", color="#2ca02c", linewidth=2.5, markersize=7, label="Silhouette Score (Higher is better)")
    ax2.set_title("Cluster Validation: Silhouette Score vs. k", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Silhouette Score", fontsize=10, fontweight="bold")
    ax2.axvline(4, color="red", linestyle="--", label="Chosen Optimal k=4")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_pca_clusters(df, save_path=os.path.join(OUTPUT_DIR, "05_pca_clusters_2d.png")):
    """Plots 2D Principal Component Analysis (PCA) projection of risk clusters."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    for tier, color in TIER_COLORS.items():
        subset = df[df["Risk_Tier"] == tier]
        ax.scatter(subset["PCA1"], subset["PCA2"], c=color, label=tier,
                   s=140, edgecolors="black", linewidths=0.9, alpha=0.9)
                   
    for _, row in df.iterrows():
        ax.annotate(row["Symbol"], (row["PCA1"], row["PCA2"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8, fontweight="bold", alpha=0.85)
                    
    ax.set_title("2D Principal Component Analysis (PCA) of Cryptocurrency Risk Clusters", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Principal Component 1 (Volatility & Tail Risk Loading)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Principal Component 2 (Liquidity Depth & Turnover Loading)", fontsize=11, fontweight="bold")
    ax.legend(title="Assigned Risk Tier", frameon=True, facecolor="white", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_hierarchical_dendrogram(df, save_path=os.path.join(OUTPUT_DIR, "06_hierarchical_dendrogram.png")):
    """Generates Agglomerative Hierarchical Clustering Dendrogram."""
    from sklearn.preprocessing import StandardScaler
    feature_cols = [
        "Annualized_Volatility_%", "Max_Drawdown_%", "VaR_95_%", "CVaR_95_%",
        "Mean_Daily_Spread_%", "Log_Avg_Volume_USD", "Volume_Volatility_CV", "Vol_to_Volume_Ratio"
    ]
    X_scaled = StandardScaler().fit_transform(df[feature_cols].values)
    linkage_matrix = linkage(X_scaled, method="ward")
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    dendrogram(linkage_matrix, labels=df["Symbol"].values, leaf_rotation=90, leaf_font_size=9, ax=ax)
    
    ax.set_title("Agglomerative Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Cryptocurrency Asset", fontsize=11, fontweight="bold")
    ax.set_ylabel("Euclidean Distance (Dissimilarity)", fontsize=11, fontweight="bold")
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def plot_cluster_risk_radar(df, save_path=os.path.join(OUTPUT_DIR, "07_cluster_risk_comparison.png")):
    """Bar chart comparing mean risk metrics across the 4 clusters."""
    metrics = ["Annualized_Volatility_%", "Max_Drawdown_%", "VaR_95_%", "Log_Avg_Volume_USD"]
    metric_labels = ["Volatility (%)", "Max Drawdown (%)", "VaR 95% (%)", "Log Daily Volume ($)"]
    
    tier_means = df.groupby("Risk_Tier")[metrics].mean().reindex([
        "Ultra-Low Risk", "Low-Moderate Risk", "Moderate-High Risk", "Extreme Risk"
    ])
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        bars = ax.bar(tier_means.index, tier_means[metric], color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Tier 1\n(Ultra-Low)", "Tier 2\n(Low-Mod)", "Tier 3\n(Mod-High)", "Tier 4\n(Extreme)"], 
                           rotation=0, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        
        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
                        
    fig.suptitle("Comparison of Core Risk & Trading Characteristics Across Discovered Clusters", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {save_path}")


def generate_all_visualizations(clustered_csv="data/processed/crypto_clustered_results.csv"):
    """Generates all project visualizations in one execution."""
    if not os.path.exists(clustered_csv):
        raise FileNotFoundError(f"Clustered results not found: {clustered_csv}")
        
    df = pd.read_csv(clustered_csv)
    print("[*] Generating publication-quality visualization figures...")
    plot_volatility_vs_volume(df)
    plot_risk_return(df)
    plot_feature_correlation(df)
    plot_optimal_k_metrics()
    plot_pca_clusters(df)
    plot_hierarchical_dendrogram(df)
    plot_cluster_risk_radar(df)
    print(f"[+] All figures successfully generated and saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_visualizations()
