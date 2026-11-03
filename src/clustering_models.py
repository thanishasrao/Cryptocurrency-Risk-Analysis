"""
Machine Learning Clustering Pipeline for Cryptocurrency Risk Analysis
CS2235 - Module 4: Feature Scaling, PCA, K-Means Clustering & Silhouette Evaluation

Implements:
1. Feature Scaling (StandardScaler)
2. Dimensionality Reduction (PCA basics in 2D)
3. Optimal k determination via Elbow Method (WCSS/Inertia) and Silhouette Score
4. K-Means Clustering (Hyperparameter tuning for k)
5. Hierarchical Clustering (for Dendrogram visualization)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Core features taught in Module 4 representing price change and trading volume
FEATURE_COLS = [
    "Annualized_Volatility_%",
    "Max_Drawdown_%",
    "Mean_Daily_Spread_%",
    "Log_Avg_Volume_USD",
    "Vol_to_Volume_Ratio"
]


def evaluate_optimal_k(X_scaled, k_range=range(2, 8), random_state=42):
    """
    Evaluates cluster quality across k values using Elbow Method (Inertia)
    and Silhouette Score, exactly as taught in CS2235 Module 4.
    """
    results = {
        "k": list(k_range),
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": []
    }
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=random_state)
        labels = kmeans.fit_predict(X_scaled)
        
        results["inertia"].append(float(kmeans.inertia_))
        results["silhouette"].append(float(silhouette_score(X_scaled, labels)))
        results["davies_bouldin"].append(float(davies_bouldin_score(X_scaled, labels)))
        
    return results


def run_clustering_pipeline(input_path="data/processed/crypto_risk_features.csv",
                            output_path="data/processed/crypto_clustered_results.csv",
                            models_dir="data/models",
                            n_clusters=4,
                            random_state=42):
    """
    Runs the end-to-end K-Means clustering pipeline (Module 4 & 5).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Feature dataset not found: {input_path}")
        
    df = pd.read_csv(input_path)
    X = df[FEATURE_COLS].values
    
    # 1. Feature Scaling: StandardScaler (Z-Score normalization as taught in Module 4)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Dimensionality Reduction: PCA Basics (2 components for 2D visualization)
    pca_2d = PCA(n_components=2, random_state=random_state)
    coords_pca_2d = pca_2d.fit_transform(X_scaled)
    explained_var_2d = pca_2d.explained_variance_ratio_
    
    df["PCA1"] = np.round(coords_pca_2d[:, 0], 4)
    df["PCA2"] = np.round(coords_pca_2d[:, 1], 4)
    
    # 3. Optimal k Evaluation (Module 4)
    k_eval_results = evaluate_optimal_k(X_scaled, k_range=range(2, 8), random_state=random_state)
    
    # 4. Train K-Means with optimal k=4
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=25, random_state=random_state)
    raw_kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # 5. Simple Agglomerative Hierarchical Clustering (for dendrogram)
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    df["Raw_Cluster"] = raw_kmeans_labels
    df["Hierarchical_Cluster"] = hierarchical_labels
    
    # 6. Sort and Rank Clusters into Financial Risk Tiers
    # Rank clusters based on mean Annualized Volatility and Drawdown
    cluster_risk_rank = (
        df.groupby("Raw_Cluster")[["Annualized_Volatility_%", "Max_Drawdown_%"]]
        .mean()
        .assign(Composite_Risk=lambda d: d["Annualized_Volatility_%"] * 0.5 + d["Max_Drawdown_%"] * 0.5)
        .sort_values("Composite_Risk")
        .index.tolist()
    )
    
    tier_mapping = {
        cluster_risk_rank[0]: "Tier 1: Ultra-Low Risk (Pegged / Stablecoins)",
        cluster_risk_rank[1]: "Tier 2: Low-to-Moderate Risk (Blue-Chip / Deep Liquidity)",
        cluster_risk_rank[2]: "Tier 3: Moderate-to-High Risk (Established Altcoins / DeFi)",
        cluster_risk_rank[3]: "Tier 4: Extreme Speculative Risk (High Volatility / Meme Tokens)"
    }
    
    tier_short = {
        cluster_risk_rank[0]: "Ultra-Low Risk",
        cluster_risk_rank[1]: "Low-Moderate Risk",
        cluster_risk_rank[2]: "Moderate-High Risk",
        cluster_risk_rank[3]: "Extreme Risk"
    }
    
    rank_order = {old_c: rank for rank, old_c in enumerate(cluster_risk_rank)}
    
    df["Risk_Cluster_ID"] = df["Raw_Cluster"].map(rank_order)
    df["Risk_Category"] = df["Raw_Cluster"].map(tier_mapping)
    df["Risk_Tier"] = df["Raw_Cluster"].map(tier_short)
    
    # Model validation metrics (Module 4)
    sil_score = silhouette_score(X_scaled, df["Risk_Cluster_ID"])
    db_score = davies_bouldin_score(X_scaled, df["Risk_Cluster_ID"])
    
    metrics_summary = {
        "n_samples": len(df),
        "n_clusters": n_clusters,
        "silhouette_score": round(sil_score, 4),
        "davies_bouldin_index": round(db_score, 4),
        "pca_explained_variance_2d": [round(float(v), 4) for v in explained_var_2d],
        "pca_total_variance_2d": round(float(np.sum(explained_var_2d)), 4),
        "k_evaluation": k_eval_results,
        "cluster_distribution": df["Risk_Tier"].value_counts().to_dict()
    }
    
    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.joblib"))
    joblib.dump(pca_2d, os.path.join(models_dir, "pca_model.joblib"))
    
    with open(os.path.join(models_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("\n" + "="*60)
    print("CS2235 K-MEANS CLUSTERING PIPELINE SUMMARY")
    print("="*60)
    print(f"Total Cryptocurrencies Clustered: {len(df)}")
    print(f"Optimal Clusters (k): {n_clusters}")
    print(f"Silhouette Score (Module 4): {sil_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    print(f"PCA 2D Total Variance Explained (Module 4): {metrics_summary['pca_total_variance_2d']*100:.2f}%")
    print("\nCluster Member Counts:")
    for tier, count in metrics_summary["cluster_distribution"].items():
        print(f"  * {tier}: {count} assets")
    print("="*60)
    
    return df, metrics_summary

if __name__ == "__main__":
    run_clustering_pipeline()
