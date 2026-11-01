"""
Machine Learning Clustering Module for Cryptocurrency Risk Analysis
Implements K-Means, Agglomerative Hierarchical Clustering, DBSCAN, and GMM.
Computes optimal cluster metrics (Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz)
and profiles risk tiers based on price volatility and trading volume dynamics.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Default risk and liquidity features representing the core problem statement:
# "how much their prices change and how much they are traded"
FEATURE_COLS = [
    "Annualized_Volatility_%",
    "Max_Drawdown_%",
    "VaR_95_%",
    "CVaR_95_%",
    "Mean_Daily_Spread_%",
    "Log_Avg_Volume_USD",
    "Volume_Volatility_CV",
    "Vol_to_Volume_Ratio"
]


def evaluate_optimal_k(X_scaled, k_range=range(2, 9), random_state=42):
    """
    Evaluates cluster quality across a range of k values using multiple metrics.
    
    Returns:
    --------
    dict
        Dictionary containing metric curves for Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz.
    """
    results = {
        "k": list(k_range),
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": []
    }
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=random_state)
        labels = kmeans.fit_predict(X_scaled)
        
        results["inertia"].append(float(kmeans.inertia_))
        results["silhouette"].append(float(silhouette_score(X_scaled, labels)))
        results["davies_bouldin"].append(float(davies_bouldin_score(X_scaled, labels)))
        results["calinski_harabasz"].append(float(calinski_harabasz_score(X_scaled, labels)))
        
    return results


def run_clustering_pipeline(input_path="data/processed/crypto_risk_features.csv",
                            output_path="data/processed/crypto_clustered_results.csv",
                            models_dir="data/models",
                            n_clusters=4,
                            random_state=42):
    """
    Executes the full clustering and evaluation pipeline.
    
    Parameters:
    -----------
    input_path : str
        Path to processed feature CSV.
    output_path : str
        Path to save final clustered dataset.
    models_dir : str
        Directory to save serialized models.
    n_clusters : int
        Number of clusters (default 4 for 4-tier risk classification).
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame, dict
        Clustered DataFrame and evaluation summary dictionary.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Feature dataset not found: {input_path}")
        
    df = pd.read_csv(input_path)
    X = df[FEATURE_COLS].values
    
    # 1. Scaling: Standard Scaler for z-score normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Dimensionality Reduction (PCA & t-SNE for 2D/3D visualization)
    pca_2d = PCA(n_components=2, random_state=random_state)
    coords_pca_2d = pca_2d.fit_transform(X_scaled)
    explained_var_2d = pca_2d.explained_variance_ratio_
    
    pca_3d = PCA(n_components=3, random_state=random_state)
    coords_pca_3d = pca_3d.fit_transform(X_scaled)
    explained_var_3d = pca_3d.explained_variance_ratio_
    
    # t-SNE with perplexity suitable for sample size
    tsne = TSNE(n_components=2, perplexity=min(10, len(df)-1), random_state=random_state, max_iter=1000)
    coords_tsne = tsne.fit_transform(X_scaled)
    
    df["PCA1"] = np.round(coords_pca_2d[:, 0], 4)
    df["PCA2"] = np.round(coords_pca_2d[:, 1], 4)
    df["PCA3"] = np.round(coords_pca_3d[:, 2], 4)
    df["tSNE1"] = np.round(coords_tsne[:, 0], 4)
    df["tSNE2"] = np.round(coords_tsne[:, 1], 4)
    
    # 3. Optimal K Evaluation
    k_eval_results = evaluate_optimal_k(X_scaled, k_range=range(2, 8), random_state=random_state)
    
    # 4. Train K-Means
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=30, random_state=random_state)
    raw_kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # 5. Train Agglomerative Hierarchical Clustering (Ward Linkage)
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    # 6. Train Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    
    # 7. Train DBSCAN (Density-based outlier detection)
    dbscan = DBSCAN(eps=1.8, min_samples=3)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    df["Raw_Cluster"] = raw_kmeans_labels
    df["Hierarchical_Cluster"] = hierarchical_labels
    df["GMM_Cluster"] = gmm_labels
    df["DBSCAN_Cluster"] = dbscan_labels
    df["GMM_Max_Probability"] = np.round(np.max(gmm_probs, axis=1), 4)
    
    # 8. Sort and Rank Clusters into Financial Risk Tiers
    # Rank clusters based on mean Annualized Volatility and Drawdown
    cluster_risk_rank = (
        df.groupby("Raw_Cluster")[["Annualized_Volatility_%", "Max_Drawdown_%", "VaR_95_%"]]
        .mean()
        .assign(Composite_Risk=lambda d: d["Annualized_Volatility_%"] * 0.5 + d["Max_Drawdown_%"] * 0.5)
        .sort_values("Composite_Risk")
        .index.tolist()
    )
    
    # Mapping rank (0 to 3) to intuitive human-readable risk categories
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
    
    # Model validation metrics
    sil_score = silhouette_score(X_scaled, df["Risk_Cluster_ID"])
    db_score = davies_bouldin_score(X_scaled, df["Risk_Cluster_ID"])
    ch_score = calinski_harabasz_score(X_scaled, df["Risk_Cluster_ID"])
    
    metrics_summary = {
        "n_samples": len(df),
        "n_clusters": n_clusters,
        "silhouette_score": round(sil_score, 4),
        "davies_bouldin_index": round(db_score, 4),
        "calinski_harabasz_index": round(ch_score, 4),
        "pca_explained_variance_2d": [round(float(v), 4) for v in explained_var_2d],
        "pca_total_variance_2d": round(float(np.sum(explained_var_2d)), 4),
        "pca_explained_variance_3d": [round(float(v), 4) for v in explained_var_3d],
        "pca_total_variance_3d": round(float(np.sum(explained_var_3d)), 4),
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
    print("CLUSTERING PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"Total Cryptocurrencies Clustered: {len(df)}")
    print(f"Optimal Clusters (k): {n_clusters}")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f} (lower is better)")
    print(f"Calinski-Harabasz Index: {ch_score:.2f} (higher is better)")
    print(f"PCA 2D Total Variance Explained: {metrics_summary['pca_total_variance_2d']*100:.2f}%")
    print(f"PCA 3D Total Variance Explained: {metrics_summary['pca_total_variance_3d']*100:.2f}%")
    print("\nCluster Member Counts:")
    for tier, count in metrics_summary["cluster_distribution"].items():
        print(f"  * {tier}: {count} assets")
    print("="*60)
    print(f"[+] Clustered dataset saved to: {output_path}")
    print(f"[+] Serialized models saved to: {models_dir}")
    
    return df, metrics_summary


if __name__ == "__main__":
    run_clustering_pipeline()
