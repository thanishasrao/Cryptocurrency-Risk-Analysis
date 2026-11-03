"""
Cryptocurrency Risk Analysis & Clustering Dashboard
Interactive Streamlit Application for CS2235 Capstone Project
Aligned with Modules 1 to 5 (Data Collection, ETL, K-Means Clustering, PCA, and Streamlit Deployment)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Crypto Risk Profiler | CS2235 Capstone",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    csv_path = "data/processed/crypto_clustered_results.csv"
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}. Please run pipeline scripts first.")
        return None
    return pd.read_csv(csv_path)


@st.cache_resource
def load_models():
    model_dir = "data/models"
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    kmeans_path = os.path.join(model_dir, "kmeans_model.joblib")
    pca_path = os.path.join(model_dir, "pca_model.joblib")
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    kmeans = joblib.load(kmeans_path) if os.path.exists(kmeans_path) else None
    pca = joblib.load(pca_path) if os.path.exists(pca_path) else None
    
    return scaler, kmeans, pca


def main():
    st.markdown('<div class="main-header">🪙 Cryptocurrency Risk Profiler Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header"><b>CS2235 Data Science Capstone Project | RV University</b><br>'
                'Grouping digital assets into empirical risk categories based on price volatility and trading volume.</div>', unsafe_allow_html=True)
                
    df = load_data()
    scaler, kmeans_model, pca_model = load_models()
    
    if df is None:
        return
        
    # Sidebar
    st.sidebar.title("📌 CS2235 Navigation")
    
    tab_selection = st.sidebar.radio(
        "Explore Project Sections:",
        ["📊 1. Volatility vs. Volume Frontier", 
         "🔍 2. Asset Risk Matrix", 
         "🧠 3. K-Means & PCA Diagnostics", 
         "🔮 4. Live Risk Classifier", 
         "💼 5. Portfolio Risk Allocator",
         "📑 6. Syllabus & Course Alignment"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Course:** CS2235 Data Science\n\n"
        "**Semester:** III (First-Year Capstone)\n\n"
        "**Department:** Data Science, RV University\n\n"
        "**Primary Algorithm:** K-Means Clustering\n\n"
        "**Student Lead:** Thanisha S Rao"
    )
    
    tier_short_colors = {
        "Ultra-Low Risk": "#2ca02c",
        "Low-Moderate Risk": "#1f77b4",
        "Moderate-High Risk": "#ff7f0e",
        "Extreme Risk": "#d62728"
    }

    # =========================================================================
    # TAB 1: Volatility vs. Volume Frontier
    # =========================================================================
    if tab_selection == "📊 1. Volatility vs. Volume Frontier":
        st.subheader("Core Problem Statement: Price Volatility vs. Trading Volume")
        st.write("This chart answers the problem statement directly: grouping cryptocurrencies based on **how much their prices change** (y-axis) and **how much they are traded** (x-axis).")
        
        # KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analyzed Assets", f"{len(df)} Cryptos")
        c2.metric("Median Volatility", f"{df['Annualized_Volatility_%'].median():.1f}%")
        c3.metric("Median Daily Volume", f"${df['Avg_Daily_Volume_USD'].median()/1e6:.1f}M")
        c4.metric("K-Means Clusters", "4 Risk Categories")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        for tier, color in tier_short_colors.items():
            subset = df[df["Risk_Tier"] == tier]
            ax.scatter(subset["Log_Avg_Volume_USD"], subset["Annualized_Volatility_%"],
                       color=color, label=tier, s=120, edgecolors="black", linewidths=0.8, alpha=0.9)
                       
        for _, row in df.iterrows():
            ax.annotate(row["Symbol"], (row["Log_Avg_Volume_USD"], row["Annualized_Volatility_%"]),
                        xytext=(5, 3), textcoords="offset points", fontsize=8, fontweight="bold", alpha=0.85)
                        
        ax.set_xlabel("Log10 Average Daily Trading Volume (USD) [Module 4 Feature]", fontweight="bold", fontsize=10)
        ax.set_ylabel("Annualized Price Volatility (%) [Module 4 Feature]", fontweight="bold", fontsize=10)
        ax.set_title("Cryptocurrency Risk Grouping: Volatility vs. Trading Volume Depth", fontweight="bold", fontsize=12)
        ax.legend(title="Assigned Risk Tier", frameon=True, facecolor="white", loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
        
        st.markdown("### Risk Tier Summary")
        tier_summary = df.groupby("Risk_Tier").agg({
            "Symbol": "count",
            "Annualized_Volatility_%": "mean",
            "Max_Drawdown_%": "mean",
            "Avg_Daily_Volume_USD": lambda v: f"${v.mean()/1e6:.1f}M"
        }).rename(columns={"Symbol": "Number of Coins", "Avg_Daily_Volume_USD": "Mean Daily Volume"})
        st.dataframe(tier_summary, use_container_width=True)

    # =========================================================================
    # TAB 2: Asset Risk Matrix
    # =========================================================================
    elif tab_selection == "🔍 2. Asset Risk Matrix":
        st.subheader("Asset Risk & Trading Metrics Table (Module 1 & 4)")
        st.write("Browse and inspect the quantitative features computed during the ETL pipeline.")
        
        selected_tiers = st.multiselect(
            "Filter by Risk Category:",
            options=df["Risk_Tier"].unique(),
            default=df["Risk_Tier"].unique()
        )
        
        filtered_df = df[df["Risk_Tier"].isin(selected_tiers)]
        display_cols = [
            "Symbol", "Name", "Risk_Tier", "Asset_Category",
            "Annualized_Volatility_%", "Max_Drawdown_%", "Mean_Daily_Spread_%", 
            "Annualized_Return_%", "Avg_Daily_Volume_USD"
        ]
        
        st.dataframe(
            filtered_df[display_cols].style.format({
                "Annualized_Volatility_%": "{:.1f}%",
                "Max_Drawdown_%": "{:.1f}%",
                "Mean_Daily_Spread_%": "{:.2f}%",
                "Annualized_Return_%": "{:.1f}%",
                "Avg_Daily_Volume_USD": "${:,.0f}"
            }),
            use_container_width=True,
            height=450
        )

    # =========================================================================
    # TAB 3: Model Diagnostics & PCA
    # =========================================================================
    elif tab_selection == "🧠 3. K-Means & PCA Diagnostics":
        st.subheader("Clustering Evaluation & PCA Basics (Module 4)")
        st.write("Module 4 covers **K-Means Clustering**, **Silhouette Score Evaluation**, and **PCA Basics (Dimensionality Reduction)**.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 1. Elbow Method & Silhouette Evaluation")
            metrics_path = "data/models/metrics_summary.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    summary = json.load(f)
                st.metric("Optimal Clusters (k)", summary.get("n_clusters", 4))
                st.metric("Silhouette Score", f"{summary.get('silhouette_score', 0):.4f}")
                st.metric("PCA 2D Variance Explained", f"{summary.get('pca_total_variance_2d', 0)*100:.2f}%")
                
            fig_k = "docs/figures/04_optimal_k_evaluation.png"
            if os.path.exists(fig_k):
                st.image(fig_k, caption="Elbow Method (WCSS) & Silhouette Score across k")
                
        with c2:
            st.markdown("#### 2. 2D Principal Component Analysis (PCA)")
            fig_pca = "docs/figures/05_pca_clusters_2d.png"
            if os.path.exists(fig_pca):
                st.image(fig_pca, caption="2D PCA Latent Space Separation (96.94% Variance)")

    # =========================================================================
    # TAB 4: Live Risk Classifier
    # =========================================================================
    elif tab_selection == "🔮 4. Live Risk Classifier":
        st.subheader("Predict Risk Category for a New Cryptocurrency (Module 5)")
        st.write("Input expected price volatility and daily volume for any cryptocurrency to predict its risk tier using our trained K-Means model.")
        
        with st.form("risk_form"):
            c1, c2 = st.columns(2)
            with c1:
                token_name = st.text_input("Cryptocurrency Name / Symbol", "MyCustomCoin")
                volatility = st.slider("Expected Annualized Volatility (%)", 0.0, 200.0, 75.0, 1.0)
                max_drawdown = st.slider("Historical Max Drawdown (%)", 0.0, 100.0, 65.0, 1.0)
            with c2:
                daily_volume = st.number_input("Average Daily Volume (USD)", 10000.0, 50000000000.0, 50000000.0, step=1000000.0)
                spread = st.slider("Average Daily High-Low Spread (%)", 0.0, 30.0, 5.0, 0.5)
                
            submitted = st.form_submit_button("⚡ Predict Risk Profile")
            
        if submitted and scaler and kmeans_model:
            log_vol = np.log10(daily_volume)
            vol_to_vol = volatility / (log_vol + 1e-5)
            
            # 5 features: ["Annualized_Volatility_%", "Max_Drawdown_%", "Mean_Daily_Spread_%", "Log_Avg_Volume_USD", "Vol_to_Volume_Ratio"]
            input_vector = np.array([[volatility, max_drawdown, spread, log_vol, vol_to_vol]])
            scaled_vec = scaler.transform(input_vector)
            raw_cluster = kmeans_model.predict(scaled_vec)[0]
            
            tier_match = df[df["Raw_Cluster"] == raw_cluster]["Risk_Tier"].iloc[0]
            tier_desc = df[df["Raw_Cluster"] == raw_cluster]["Risk_Category"].iloc[0]
            
            st.success(f"### Classification Result: **{tier_match}**")
            st.info(f"**Full Profile:** {tier_desc}")

    # =========================================================================
    # TAB 5: Portfolio Risk Allocator
    # =========================================================================
    elif tab_selection == "💼 5. Portfolio Risk Allocator":
        st.subheader("Cluster-Based Portfolio Allocation (Module 5 Application in Finance)")
        st.write("Allocate digital assets across the four discovered risk categories based on investor risk tolerance.")
        
        strategy = st.selectbox(
            "Select Investor Risk Tolerance:",
            ["Conservative Wealth Preservation", 
             "Balanced Core-Satellite Growth", 
             "Aggressive Growth"]
        )
        
        allocations = {
            "Conservative Wealth Preservation": {"Ultra-Low Risk": 60, "Low-Moderate Risk": 30, "Moderate-High Risk": 10, "Extreme Risk": 0},
            "Balanced Core-Satellite Growth": {"Ultra-Low Risk": 20, "Low-Moderate Risk": 50, "Moderate-High Risk": 25, "Extreme Risk": 5},
            "Aggressive Growth": {"Ultra-Low Risk": 5, "Low-Moderate Risk": 35, "Moderate-High Risk": 45, "Extreme Risk": 15}
        }
        
        weights = allocations[strategy]
        
        c1, c2 = st.columns(2)
        with c1:
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            ax_pie.pie(
                [w for w in weights.values() if w > 0],
                labels=[t for t, w in weights.items() if w > 0],
                autopct="%1.0f%%",
                colors=[tier_short_colors[t] for t, w in weights.items() if w > 0],
                startangle=140
            )
            ax_pie.set_title(f"Target Weights: {strategy}", fontweight="bold", fontsize=11)
            st.pyplot(fig_pie)
            
        with c2:
            st.markdown(f"#### Allocation Insights ({strategy})")
            for tier, wt in weights.items():
                if wt > 0:
                    assets = df[df["Risk_Tier"] == tier]["Symbol"].tolist()
                    st.write(f"- **{tier} ({wt}% Weight):** e.g., `{', '.join(assets[:4])}`")

    # =========================================================================
    # TAB 6: Syllabus Alignment
    # =========================================================================
    elif tab_selection == "📑 6. Syllabus & Course Alignment":
        st.subheader("RV University CS2235 Syllabus Mapping")
        st.markdown(r"""
        | Course Module | Syllabus Topic | Project Implementation in this Capstone |
        | :--- | :--- | :--- |
        | **Module 1: Data Science Lifecycle** | Data Collection via API, Pandas/NumPy, Data Types | Ingested 2-year daily OHLCV data using Yahoo Finance API, stored in Pandas DataFrame. Measurement scale: Ratio (price, volume), Nominal (symbol). |
        | **Module 2: Data Ingestion & ETL** | Batch ETL Pipelines, Data Validation & Quality Checks | `data_collection.py` extracts raw market data; `feature_engineering.py` transforms prices into returns & volatility; validated 0 null values. |
        | **Module 3: Machine Learning** | Supervised vs. Unsupervised Learning | Formulated as Unsupervised Learning (Clustering) since there are no predefined risk labels. |
        | **Module 4: Modeling & Features** | Feature Scaling, PCA Basics, K-Means, Silhouette Score | Used `StandardScaler`, 2D PCA (96.9% variance), K-Means++ clustering with optimal $k=4$ confirmed via Elbow & Silhouette score (0.3868). |
        | **Module 5: Deployment & Finance** | Applications in Finance, Streamlit Deployment | Deployed this interactive Streamlit dashboard (`app.py`) with risk classification and portfolio simulation. |
        """)

if __name__ == "__main__":
    main()
