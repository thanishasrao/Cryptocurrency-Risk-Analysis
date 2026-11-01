"""
Cryptocurrency Risk Analysis & Clustering Dashboard
Interactive Streamlit Application for CS2235 Capstone Project
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Page Configuration
st.set_page_config(
    page_title="Crypto Risk Profiler | Fintech Capstone",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
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
    st.markdown('<div class="main-header">🪙 Cryptocurrency Risk Profiling & Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header"><b>CS2235 Data Science Capstone Project | RV University</b><br>'
                'Grouping digital assets into empirical risk categories based on price volatility dynamics and trading volume liquidity.</div>', unsafe_allow_html=True)
                
    df = load_data()
    scaler, kmeans_model, pca_model = load_models()
    
    if df is None:
        return
        
    # Sidebar
    st.sidebar.image("https://img.icons8.com/isometric/100/cryptocurrency.png", width=70)
    st.sidebar.title("Navigation & Parameters")
    
    tab_selection = st.sidebar.radio(
        "Explore Sections:",
        ["📊 Market Risk Frontier", 
         "🔍 Asset Risk Matrix", 
         "🧠 Model Diagnostics & PCA", 
         "🔮 Live Risk Classifier", 
         "💼 Portfolio Risk Allocator",
         "📑 Project Documentation"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Project:** CS2235 Capstone\n\n"
        "**Domain:** Fintech / Algorithmic Finance\n\n"
        "**Technique:** Unsupervised Clustering\n\n"
        "**Data Horizon:** 2-Year Daily OHLCV\n\n"
        "**Lead Researcher:** Thanisha S Rao"
    )
    
    # Tier color palette
    tier_colors = {
        "Tier 1: Ultra-Low Risk (Pegged / Stablecoins)": "#2ca02c",
        "Tier 2: Low-to-Moderate Risk (Blue-Chip / Deep Liquidity)": "#1f77b4",
        "Tier 3: Moderate-to-High Risk (Established Altcoins / DeFi)": "#ff7f0e",
        "Tier 4: Extreme Speculative Risk (High Volatility / Meme Tokens)": "#d62728"
    }
    
    tier_short_colors = {
        "Ultra-Low Risk": "#2ca02c",
        "Low-Moderate Risk": "#1f77b4",
        "Moderate-High Risk": "#ff7f0e",
        "Extreme Risk": "#d62728"
    }

    # =========================================================================
    # TAB 1: Market Risk Frontier
    # =========================================================================
    if tab_selection == "📊 Market Risk Frontier":
        st.subheader("Market Risk Frontier: Volatility vs. Liquidity Depth")
        st.write("This primary visualization directly addresses the core problem statement: grouping cryptocurrencies into similar risk tiers based on **how much their prices change** (y-axis) and **how much they are traded** (x-axis).")
        
        # Top KPI Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analyzed Cryptocurrencies", f"{len(df)} Assets")
        c2.metric("Median Volatility", f"{df['Annualized_Volatility_%'].median():.1f}%")
        c3.metric("Median Daily Volume", f"${df['Avg_Daily_Volume_USD'].median()/1e6:.1f}M")
        c4.metric("Discovered Risk Clusters", "4 Tiers")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
        for tier, color in tier_short_colors.items():
            subset = df[df["Risk_Tier"] == tier]
            ax.scatter(subset["Log_Avg_Volume_USD"], subset["Annualized_Volatility_%"],
                       color=color, label=tier, s=130, edgecolors="black", linewidths=0.9, alpha=0.9)
                       
        for _, row in df.iterrows():
            ax.annotate(row["Symbol"], (row["Log_Avg_Volume_USD"], row["Annualized_Volatility_%"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8, fontweight="bold", alpha=0.85)
                        
        ax.set_xlabel("Log10 Average Daily Trading Volume (USD) [Liquidity Proxy]", fontweight="bold", fontsize=11)
        ax.set_ylabel("Annualized Price Volatility (%) [Risk Proxy]", fontweight="bold", fontsize=11)
        ax.set_title("Cryptocurrency Volatility vs. Trading Volume Liquidity Frontier", fontweight="bold", fontsize=13)
        ax.legend(title="Assigned Risk Tier", frameon=True, facecolor="white", loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
        
        st.markdown("### Risk Tier Characteristics")
        tier_summary = df.groupby("Risk_Tier").agg({
            "Symbol": "count",
            "Annualized_Volatility_%": "mean",
            "Max_Drawdown_%": "mean",
            "VaR_95_%": "mean",
            "Avg_Daily_Volume_USD": lambda v: f"${v.mean()/1e6:.1f}M"
        }).rename(columns={"Symbol": "Asset Count", "Avg_Daily_Volume_USD": "Mean Daily Volume"})
        st.dataframe(tier_summary, use_container_width=True)

    # =========================================================================
    # TAB 2: Asset Risk Matrix
    # =========================================================================
    elif tab_selection == "🔍 Asset Risk Matrix":
        st.subheader("Asset Risk & Liquidity Matrix")
        st.write("Browse, filter, and inspect detailed quantitative risk metrics across all examined digital assets.")
        
        selected_tiers = st.multiselect(
            "Filter by Risk Tier:",
            options=df["Risk_Tier"].unique(),
            default=df["Risk_Tier"].unique()
        )
        
        search_query = st.text_input("Search by Token Name or Symbol:", "").strip().upper()
        
        filtered_df = df[df["Risk_Tier"].isin(selected_tiers)]
        if search_query:
            filtered_df = filtered_df[
                filtered_df["Symbol"].str.contains(search_query) | 
                filtered_df["Name"].str.upper().str.contains(search_query)
            ]
            
        display_cols = [
            "Symbol", "Name", "Risk_Tier", "Asset_Category",
            "Annualized_Volatility_%", "Annualized_Return_%", "Max_Drawdown_%",
            "VaR_95_%", "CVaR_95_%", "Mean_Daily_Spread_%", "Avg_Daily_Volume_USD", "Sortino_Ratio"
        ]
        
        st.dataframe(
            filtered_df[display_cols].style.format({
                "Annualized_Volatility_%": "{:.1f}%",
                "Annualized_Return_%": "{:.1f}%",
                "Max_Drawdown_%": "{:.1f}%",
                "VaR_95_%": "{:.1f}%",
                "CVaR_95_%": "{:.1f}%",
                "Mean_Daily_Spread_%": "{:.2f}%",
                "Avg_Daily_Volume_USD": "${:,.0f}",
                "Sortino_Ratio": "{:.2f}"
            }),
            use_container_width=True,
            height=500
        )
        
        # Download button
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="crypto_risk_profiles_filtered.csv",
            mime="text/csv"
        )

    # =========================================================================
    # TAB 3: Model Diagnostics & PCA
    # =========================================================================
    elif tab_selection == "🧠 Model Diagnostics & PCA":
        st.subheader("Clustering Evaluation & Latent Space Decomposition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1. Optimal Cluster Diagnostics")
            metrics_path = "data/models/metrics_summary.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    summary = json.load(f)
                st.metric("Optimal Clusters (k)", summary.get("n_clusters", 4))
                st.metric("Silhouette Score", f"{summary.get('silhouette_score', 0):.4f}")
                st.metric("Davies-Bouldin Index", f"{summary.get('davies_bouldin_index', 0):.4f}")
                st.metric("PCA 2D Variance Explained", f"{summary.get('pca_total_variance_2d', 0)*100:.2f}%")
                
                # Show figure
                fig_k = "docs/figures/04_optimal_k_evaluation.png"
                if os.path.exists(fig_k):
                    st.image(fig_k, caption="Elbow Method & Silhouette Validation Across k")
            else:
                st.info("Metrics summary not found.")
                
        with col2:
            st.markdown("#### 2. 2D Principal Component Space (PCA)")
            fig_pca = "docs/figures/05_pca_clusters_2d.png"
            if os.path.exists(fig_pca):
                st.image(fig_pca, caption="2D PCA Latent Space Separation")
            else:
                st.info("PCA plot not generated yet.")
                
        st.markdown("---")
        st.markdown("#### 3. Hierarchical Clustering Dendrogram (Ward Linkage)")
        fig_dendro = "docs/figures/06_hierarchical_dendrogram.png"
        if os.path.exists(fig_dendro):
            st.image(fig_dendro, caption="Hierarchical Taxonomy of Digital Assets", use_container_width=True)

    # =========================================================================
    # TAB 4: Live Risk Classifier
    # =========================================================================
    elif tab_selection == "🔮 Live Risk Classifier":
        st.subheader("Predict Risk Tier for a New Cryptocurrency")
        st.write("Input simulated or observed trading characteristics of any cryptocurrency to classify its empirical risk category using our trained K-Means pipeline.")
        
        with st.form("risk_form"):
            c1, c2 = st.columns(2)
            with c1:
                token_name = st.text_input("Cryptocurrency Name / Symbol", "Custom Token (XYZ)")
                volatility = st.slider("Expected Annualized Volatility (%)", 0.0, 250.0, 85.0, 0.5)
                max_drawdown = st.slider("Historical Max Drawdown (%)", 0.0, 100.0, 75.0, 0.5)
                var_95 = st.slider("Daily Value at Risk (95% VaR) (%)", 0.0, 30.0, 8.5, 0.1)
            with c2:
                daily_volume = st.number_input("Average Daily Volume (USD)", 10000.0, 50000000000.0, 50000000.0, step=1000000.0)
                spread = st.slider("Average Intraday High-Low Spread (%)", 0.0, 40.0, 6.0, 0.1)
                vol_cv = st.slider("Volume Volatility (CV)", 0.0, 5.0, 0.8, 0.05)
                
            submitted = st.form_submit_button("⚡ Predict Risk Profile & Classify")
            
        if submitted and scaler and kmeans_model:
            log_vol = np.log10(daily_volume)
            cvar_95 = var_95 * 1.35
            vol_to_vol = volatility / (log_vol + 1e-5)
            
            # Feature vector matching FEATURE_COLS:
            # ["Annualized_Volatility_%", "Max_Drawdown_%", "VaR_95_%", "CVaR_95_%", "Mean_Daily_Spread_%", "Log_Avg_Volume_USD", "Volume_Volatility_CV", "Vol_to_Volume_Ratio"]
            input_vector = np.array([[volatility, max_drawdown, var_95, cvar_95, spread, log_vol, vol_cv, vol_to_vol]])
            scaled_vec = scaler.transform(input_vector)
            
            raw_cluster = kmeans_model.predict(scaled_vec)[0]
            
            # Distance to cluster centers
            distances = np.linalg.norm(kmeans_model.cluster_centers_ - scaled_vec, axis=1)
            
            # Determine nearest cluster from dataset
            # Map raw cluster to risk tier
            tier_match = df[df["Raw_Cluster"] == raw_cluster]["Risk_Tier"].iloc[0]
            tier_desc = df[df["Raw_Cluster"] == raw_cluster]["Risk_Category"].iloc[0]
            
            st.success(f"### Classification Result: **{tier_match}**")
            st.info(f"**Full Profile:** {tier_desc}")
            
            # Actionable financial guidance
            if "Ultra-Low" in tier_match:
                st.write("🛡️ **Investor Recommendation:** Suitable for cash management, collateralization, and risk-off treasury holdings.")
            elif "Low-Moderate" in tier_match:
                st.write("💎 **Investor Recommendation:** Core blue-chip allocation; deep institutional liquidity cushions large capital flows.")
            elif "Moderate-High" in tier_match:
                st.write("⚖️ **Investor Recommendation:** Tactical growth allocation; requires active monitoring of drawdowns and volatility.")
            else:
                st.write("⚠️ **Investor Recommendation:** Highly speculative / aggressive; restrict allocation to small fraction of total portfolio with strict stop-losses.")

    # =========================================================================
    # TAB 5: Portfolio Risk Allocator
    # =========================================================================
    elif tab_selection == "💼 Portfolio Risk Allocator":
        st.subheader("Cluster-Based Portfolio Allocation Engine")
        st.write("Build an optimal digital asset allocation strategy derived from the empirical cluster boundaries.")
        
        strategy = st.selectbox(
            "Select Investor Risk Appetite:",
            ["Conservative Wealth Preservation", 
             "Balanced Core-Satellite Growth", 
             "Aggressive Growth Alpha", 
             "Speculative High-Beta"]
        )
        
        allocations = {
            "Conservative Wealth Preservation": {"Ultra-Low Risk": 60, "Low-Moderate Risk": 30, "Moderate-High Risk": 10, "Extreme Risk": 0},
            "Balanced Core-Satellite Growth": {"Ultra-Low Risk": 20, "Low-Moderate Risk": 50, "Moderate-High Risk": 25, "Extreme Risk": 5},
            "Aggressive Growth Alpha": {"Ultra-Low Risk": 5, "Low-Moderate Risk": 35, "Moderate-High Risk": 45, "Extreme Risk": 15},
            "Speculative High-Beta": {"Ultra-Low Risk": 0, "Low-Moderate Risk": 20, "Moderate-High Risk": 40, "Extreme Risk": 40}
        }
        
        weights = allocations[strategy]
        
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
            ax_pie.pie(
                [w for w in weights.values() if w > 0],
                labels=[t for t, w in weights.items() if w > 0],
                autopct="%1.0f%%",
                colors=[tier_short_colors[t] for t, w in weights.items() if w > 0],
                startangle=140,
                textprops={"fontweight": "bold", "fontsize": 10}
            )
            ax_pie.set_title(f"Target Allocation: {strategy}", fontweight="bold", fontsize=12)
            st.pyplot(fig_pie)
            
        with c2:
            st.markdown(f"#### Recommended Portfolio Archetype: {strategy}")
            for tier, wt in weights.items():
                if wt > 0:
                    assets = df[df["Risk_Tier"] == tier]["Symbol"].tolist()
                    st.write(f"- **{tier} ({wt}% Weight):** Exemplars include `{', '.join(assets[:5])}`")
                    
            st.markdown("""
            **Portfolio Construction Insights:**
            - Allocating a non-zero portion to **Ultra-Low Risk (Stablecoins)** provides dry powder during macro liquidity contractions.
            - Anchoring with **Low-to-Moderate Risk (BTC/ETH/BNB)** establishes institutional liquidity depth.
            - Restricting **Extreme Risk** protects capital from catastrophic >85% drawdowns.
            """)

    # =========================================================================
    # TAB 6: Project Documentation
    # =========================================================================
    elif tab_selection == "📑 Project Documentation":
        st.subheader("CS2235 Capstone Documentation & Academic Framework")
        st.markdown(r"""
        ### RV University - B.Sc (Hons.) Data Science
        - **Course:** CS2235 Data Science Capstone Project
        - **Mentors:** Prof. Muralidhar Billa, Prof. Thirukumaran Subbiramani
        - **Topic:** Fintech: Cryptocurrency Analysis — Grouping cryptocurrencies into similar risk categories based on how much their prices change and how much they are traded.
        
        #### Mathematical Formulations:
        1. **Annualized Volatility ($\sigma_{ann}$):** $\sigma_{ann} = \sigma_{daily} \times \sqrt{365}$
        2. **Maximum Drawdown (MDD):** $\text{MDD} = |\min_{t} (\frac{P_t - \max_{\tau \le t} P_\tau}{\max_{\tau \le t} P_\tau})|$
        3. **Value at Risk (95% VaR):** 5th percentile of daily return distribution.
        4. **Amihud Illiquidity Ratio:** $\text{Median}(\frac{|R_t|}{\text{Volume}_t}) \times 10^9$
        
        #### Unsupervised Clustering Methodology:
        - Evaluated **K-Means**, **Agglomerative Hierarchical (Ward Linkage)**, **GMM**, and **DBSCAN**.
        - Optimal number of clusters validated at $k=4$ using **Inertia Elbow Curves**, **Silhouette Score (0.3464)**, **Davies-Bouldin Index (0.8382)**, and **Calinski-Harabasz Index (33.78)**.
        - Dimensionality reduced via **PCA** (explaining 93.78% of 8-dimensional variance in 2 components).
        """)

if __name__ == "__main__":
    main()
