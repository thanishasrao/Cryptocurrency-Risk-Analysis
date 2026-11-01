"""
Unit tests for Cryptocurrency Risk Analysis & Clustering Pipeline
"""

import os
import joblib
import numpy as np
import pandas as pd

def test_feature_matrix_exists_and_valid():
    feat_path = "data/processed/crypto_risk_features.csv"
    assert os.path.exists(feat_path), f"Processed features not found at {feat_path}"
    
    df = pd.read_csv(feat_path)
    assert len(df) >= 25, f"Expected at least 25 assets, found {len(df)}"
    
    required_cols = [
        "Symbol", "Annualized_Volatility_%", "Max_Drawdown_%", "VaR_95_%",
        "CVaR_95_%", "Mean_Daily_Spread_%", "Log_Avg_Volume_USD", "Volume_Volatility_CV"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
        assert not df[col].isnull().any(), f"NaN values detected in {col}"


def test_clustered_results_integrity():
    clustered_path = "data/processed/crypto_clustered_results.csv"
    assert os.path.exists(clustered_path), f"Clustered results not found at {clustered_path}"
    
    df = pd.read_csv(clustered_path)
    assert "Risk_Tier" in df.columns
    assert "Risk_Cluster_ID" in df.columns
    assert "PCA1" in df.columns
    assert "PCA2" in df.columns
    
    # Check 4 unique risk tiers
    unique_tiers = df["Risk_Tier"].unique()
    assert len(unique_tiers) == 4, f"Expected 4 distinct risk tiers, found {len(unique_tiers)}"
    
    # Verify risk tier monotonicity: Mean Volatility of Tier 1 < Tier 2 < Tier 3 < Tier 4
    tier_order = ["Ultra-Low Risk", "Low-Moderate Risk", "Moderate-High Risk", "Extreme Risk"]
    vol_means = [df[df["Risk_Tier"] == t]["Annualized_Volatility_%"].mean() for t in tier_order]
    
    assert vol_means[0] < vol_means[1] < vol_means[2] < vol_means[3], \
        f"Risk tiers violate volatility monotonicity: {vol_means}"


def test_serialized_models_and_prediction():
    scaler_path = "data/models/scaler.joblib"
    kmeans_path = "data/models/kmeans_model.joblib"
    pca_path = "data/models/pca_model.joblib"
    
    assert os.path.exists(scaler_path)
    assert os.path.exists(kmeans_path)
    assert os.path.exists(pca_path)
    
    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    pca = joblib.load(pca_path)
    
    # Test sample prediction with an 8-feature dummy vector
    # [Vol, MDD, VaR95, CVaR95, Spread, LogVol, VolCV, VolToVol]
    sample_vec = np.array([[75.0, 70.0, 6.5, 9.0, 5.0, 9.0, 0.8, 8.3]])
    scaled_vec = scaler.transform(sample_vec)
    
    pred_cluster = kmeans.predict(scaled_vec)
    assert pred_cluster[0] in [0, 1, 2, 3]
    
    pca_transformed = pca.transform(scaled_vec)
    assert pca_transformed.shape == (1, 2)


if __name__ == "__main__":
    test_feature_matrix_exists_and_valid()
    test_clustered_results_integrity()
    test_serialized_models_and_prediction()
    print("[+] All pipeline tests passed successfully!")
