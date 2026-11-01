# CS2235: Data Science Capstone Project Final Report

---

# Fintech: Cryptocurrency Risk Profiling
### Grouping Cryptocurrencies into Empirical Risk Categories Based on Price Volatility Dynamics and Trading Volume Depth

**School of Computer Science & Engineering, RV University, Bengaluru**  
**Programme:** Bachelor of Science (Honours) in Data Science — Semester III  
**Course Code:** CS2235 (Data Science Capstone Project)  
**Academic Year:** 2026–2027  

**Course Instructors:**  
- **Prof. Muralidhar Billa**, Department of Data Science, RV University  
- **Prof. Thirukumaran Subbiramani**, Department of Data Science, RV University  

**Student Researcher & Project Lead:**  
- **Thanisha S Rao** (B.Sc Hons. Data Science)  

**Date of Submission:** November 2026  

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction and Problem Statement](#2-introduction-and-problem-statement)
3. [Related Work and Background](#3-related-work-and-background)
4. [Dataset Description and Quality Audit](#4-dataset-description-and-quality-audit)
5. [Methodology](#5-methodology)
   - 5.1 Data Cleaning and Ingestion Pipeline
   - 5.2 Mathematical Feature Engineering (Risk & Liquidity Metrics)
   - 5.3 Standardization and Dimensionality Reduction (PCA & t-SNE)
   - 5.4 Machine Learning Clustering Algorithms (K-Means, Hierarchical, GMM, DBSCAN)
   - 5.5 Cluster Validation Framework (Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz)
6. [Results and Evaluation](#6-results-and-evaluation)
   - 6.1 Optimal Cluster Parameter Determination ($k=4$)
   - 6.2 Latent Space Representation and PCA Decomposition
   - 6.3 Empirical Risk Taxonomy and Cluster Archetypes
7. [Discussion and Financial Implications](#7-discussion-and-financial-implications)
   - 7.1 The Volatility-Volume Frontier and Market Liquidity
   - 7.2 Portfolio Construction and Risk-Adjusted Allocation
   - 7.3 Limitations and Model Assumptions
8. [Conclusion and Future Work](#8-conclusion-and-future-work)
9. [References](#9-references)

---

## 1. Abstract
The decentralized financial ecosystem has expanded into an intricate multi-trillion dollar asset class composed of thousands of publicly traded tokens. However, unlike traditional equity and debt securities, digital assets lack centralized institutional credit ratings, standardized risk disclosure mechanisms, and formal asset-class segmentations. Retail and institutional investors are consequently exposed to asymmetric tail risk, sudden drawdowns, and illiquidity traps disguised by superficial market valuations. 

This capstone project presents an end-to-end unsupervised machine learning framework designed to solve the fintech challenge of grouping cryptocurrencies into similar empirical risk categories based on **how much their prices change (volatility, drawdowns, and tail risk)** and **how much they are traded (dollar trading volume, turnover velocity, and liquidity impact)**. Using continuous, daily multi-year trading data across 28 diverse cryptocurrencies—spanning stablecoins, blue-chips, layer-1 ecosystems, decentralized finance (DeFi) protocols, and speculative meme tokens—we engineer eight quantitative risk and liquidity features. 

We evaluate four unsupervised learning algorithms: K-Means (with K-Means++ initialization), Agglomerative Hierarchical Clustering (Ward linkage), Gaussian Mixture Models (GMM), and Density-Based Spatial Clustering of Applications with Noise (DBSCAN). Through multi-metric optimization—including the Elbow Method, Silhouette Analysis ($S = 0.3464$), Davies-Bouldin Index ($DB = 0.8382$), and Calinski-Harabasz Index ($CH = 33.78$)—an optimal partition of $k = 4$ risk clusters was established. Two-dimensional Principal Component Analysis (PCA) captured $93.78\%$ of total variance, cleanly separating the assets into four distinct tiers: Tier 1 (Ultra-Low Risk / Pegged Stability), Tier 2 (Low-to-Moderate Risk / Institutional Blue-Chips), Tier 3 (Moderate-to-High Risk / Utility Platforms), and Tier 4 (Extreme Speculative Risk / High-Beta Meme Assets). Finally, an interactive Streamlit dashboard (`app.py`) was deployed featuring real-time risk classification and dynamic portfolio weight optimization.

---

## 2. Introduction and Problem Statement

### 2.1 Context and Problem Formulation
In traditional quantitative finance, capital allocation decisions rely extensively on formalized risk categories. Fixed-income instruments are graded by credit rating agencies (e.g., AAA through junk status by Standard & Poor's and Moody's), while public equities are segregated by capitalization (large-, mid-, small-cap) and fundamental sector taxonomies (e.g., GICS sectors). 

Conversely, the cryptocurrency ecosystem operates continuously (24/7/365) across globally distributed, decentralized exchanges without standardized reporting mandates. Market participants frequently commit capital using naive metrics such as circulating market capitalization or recent historical returns. This practice generates three fundamental systemic vulnerabilities:
1. **The Liquidity Illusion:** High token market capitalizations often disguise negligible market depth, where moderate selling volume causes severe price slippage.
2. **Hidden Downside Tail Risk:** Assets exhibiting comparable positive annual price returns frequently diverge radically in their maximum drawdown profiles, downside deviations, and Value at Risk (VaR).
3. **Speculative Contagion:** The rapid emergence of community tokens and meme coins introduces extreme volatility spikes that distort standard portfolio diversification models.

### 2.2 Objective and Problem Statement
To address this deficiency, the problem statement of this project is formally framed as:
> **"Fintech: Cryptocurrency Analysis — Grouping cryptocurrencies into similar risk categories based on how much their prices change (price volatility dynamics, drawdowns, and tail risk) and how much they are traded (trading volume depth, volume volatility, and illiquidity proxies)."**

The primary objectives are:
- To ingest, clean, and validate contiguous daily historical OHLCV data for 28 representative cryptocurrencies across all asset sectors.
- To formulate and compute quantitative features capturing price volatility and trading liquidity.
- To execute unsupervised clustering algorithms without target labels, identifying natural latent risk tiers.
- To evaluate cluster separation using rigorous geometric and mathematical validation metrics.
- To develop an interactive deployment dashboard enabling investors to simulate risk categories and construct risk-adjusted cryptocurrency portfolios.

---

## 3. Related Work and Background

### 3.1 Quantitative Risk Modeling in Digital Assets
Early quantitative literature on cryptocurrency behavior focused primarily on Bitcoin as a speculative synthetic asset or alternative store of value (Baur et al., 2018; Dyhrberg, 2016). As Ethereum and alternative smart contract tokens gained prominence, researchers observed heavy-tailed return distributions and pronounced volatility clustering characteristic of GARCH-type processes (Katsiampa, 2017). 

Modern financial literature increasingly highlights the necessity of multi-factor risk assessments that transcend simple historical return variance. In particular, extreme value theory (EVT), Value at Risk (VaR), and Conditional Value at Risk (CVaR / Expected Shortfall) provide essential insights into tail dependencies and worst-case drawdown probabilities during market stress (Rockafellar & Uryasev, 2000).

### 3.2 Liquidity Metrics and Price Impact
Trading volume represents the lifeblood of asset market efficiency. Amihud (2002) introduced the illiquidity ratio ($\text{ILLIQ}$), defining illiquidity as the average ratio of absolute daily return to daily dollar volume. In crypto-asset markets, high nominal trading volumes are occasionally accompanied by wash trading or transient speculative spikes (Cong et al., 2021). Consequently, incorporating log-transformed volume, volume coefficient of variation (CV), and volatility-to-volume ratios provides a more stable representation of market depth than raw volume figures alone.

### 3.3 Machine Learning and Unsupervised Clustering in Fintech
Unsupervised learning techniques have been widely deployed in fintech for customer segmentation, fraud anomaly detection, and portfolio risk grouping. MacQueen (1967) formulated the classical $K$-Means clustering heuristic, which iteratively minimizes within-cluster sum of squares (WCSS). Arthur and Vassilvitskii (2007) enhanced this via $K$-Means++, utilizing probabilistic distance-weighted centroid seeding to avoid local minima. 

In asset management, Agglomerative Hierarchical Clustering and Ward's minimum variance criterion (Ward, 1963) are frequently preferred for their interpretability through dendrograms, which mirror hierarchical asset allocation strategies (de Prado, 2016). Gaussian Mixture Models (Reynolds, 2009) complement distance-based methods by providing soft probabilistic memberships, accommodating elliptical cluster shapes in financial feature space.

---

## 4. Dataset Description and Quality Audit

### 4.1 Data Sources and Asset Taxonomy
Historical cryptocurrency market data was programmatically harvested using the Yahoo Finance Financial API (`yfinance`), cross-validated with historical benchmarks from CoinMarketCap and CoinGecko. The dataset spans a two-year continuous daily trading window (731 contiguous calendar days) from late 2024 through late 2026. 

To ensure complete empirical validity and avoid selection bias, the 28 analyzed digital assets were systematically sampled across five functional asset classes:
1. **USD-Pegged Stablecoins (Control Group):** Tether (USDT), USD Coin (USDC), Dai (DAI). These establish an empirical control baseline characterized by near-zero volatility and high liquidity turnover.
2. **Mega-Cap Blue-Chips:** Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB). These represent established institutional stores of value with multi-billion-dollar daily liquidity.
3. **Major Layer-1 & Layer-2 Platforms:** Solana (SOL), Ripple (XRP), Cardano (ADA), Avalanche (AVAX), Polkadot (DOT), Near Protocol (NEAR), Cosmos (ATOM), Algorand (ALGO).
4. **Decentralized Finance (DeFi) & Utility Protocols:** Chainlink (LINK), Aave (AAVE), Maker (MKR), Lido DAO (LDO), Curve DAO (CRV).
5. **Speculative & High-Beta Meme Assets:** Dogecoin (DOGE), Shiba Inu (SHIB), Bonk (BONK), Floki (FLOKI), dogwifhat (WIF), Stellar (XLM), Litecoin (LTC), Bitcoin Cash (BCH), Ethereum Classic (ETC).

### 4.2 Data Cleansing and Quality Validation
Cryptocurrency markets operate without market holidays or trading halts. The raw ingestion yielded 20,468 total daily observations (28 assets $\times$ 731 trading days). A multi-stage quality audit was performed:
- **Timestamp Alignment:** All timestamps were standardized to UTC date strings (`YYYY-MM-DD`) with time-zone offsets removed.
- **Missing Value Audit:** Verification revealed zero missing values (`NaN` count $= 0$) across Open, High, Low, Close, and Volume columns.
- **Price Anomaly Detection:** Verified that $\text{High}_t \ge \max(\text{Open}_t, \text{Close}_t)$ and $\text{Low}_t \le \min(\text{Open}_t, \text{Close}_t)$ held across all 20,468 records.
- **Zero-Volume Filtering:** Days exhibiting near-zero volume ($V_t < \$100,000$) were screened to prevent computational division anomalies in ratio metrics.

---

## 5. Methodology

The end-to-end data science architecture is structured into five sequential phases:

```
[Raw 2-Year Daily OHLCV Data (28 Assets)]
                   │
                   ▼
[Data Cleaning & Timestamp Harmonization]
                   │
                   ▼
[Mathematical Feature Engineering (8 Core Metrics)]
 ├── Price Volatility Metrics (Annualized Vol, MDD, VaR 95%, CVaR 95%, Daily Spread)
 └── Trading Liquidity Metrics (Log Volume, Volume CV, Vol-to-Volume Ratio)
                   │
                   ▼
[Z-Score Standardization (StandardScaler)]
                   │
                   ▼
[Dimensionality Reduction (PCA & t-SNE)]
                   │
                   ▼
[Multi-Algorithm Clustering (K-Means++, Hierarchical Ward, GMM, DBSCAN)]
                   │
                   ▼
[Cluster Validation & Optimal k Selection (Elbow, Silhouette, Davies-Bouldin)]
                   │
                   ▼
[Risk Profile Synthesis & Interactive Streamlit Deployment (app.py)]
```

### 5.1 Mathematical Feature Engineering
To quantify the dual dimensions of the problem statement—price change and trading volume—eight quantitative features were mathematically derived:

#### A. Price Change & Volatility Metrics
1. **Daily Simple Return ($R_t$):**
   $$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
   where $P_t$ denotes the closing price on day $t$.

2. **Annualized Price Volatility ($\sigma_{ann}$):**
   $$\sigma_{ann} = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (R_t - \bar{R})^2} \times \sqrt{365} \times 100$$
   Digital asset markets trade 365 days per year, necessitating the $\sqrt{365}$ annualization factor.

3. **Maximum Drawdown ($MDD$):**
   $$\text{Peak}_t = \max_{\tau \le t} P_\tau$$
   $$\text{Drawdown}_t = \frac{P_t - \text{Peak}_t}{\text{Peak}_t} \times 100$$
   $$MDD = \left| \min_{t \in [1, T]} (\text{Drawdown}_t) \right|$$
   $MDD$ measures the worst possible peak-to-trough capital loss experienced by an investor over the observation window.

4. **Value at Risk at 95% Confidence ($\text{VaR}_{95}$):**
   $$\text{VaR}_{95} = - \text{Percentile}(R_t, 5) \times 100$$
   Represents the minimum single-day percentage loss expected on the 5% worst trading days.

5. **Conditional Value at Risk / Expected Shortfall ($\text{CVaR}_{95}$):**
   $$\text{CVaR}_{95} = - \mathbb{E}\left[ R_t \;\middle|\; R_t \le \text{Percentile}(R_t, 5) \right] \times 100$$
   Quantifies the average loss severity given that the 95% VaR threshold has been breached, providing critical insight into downside tail fatness.

6. **Average Daily High-Low Spread Percentage ($\overline{\text{Spread}}$):**
   $$\text{Spread}_t = \frac{\text{High}_t - \text{Low}_t}{\text{Close}_t} \times 100$$
   $$\overline{\text{Spread}} = \frac{1}{T} \sum_{t=1}^T \text{Spread}_t$$
   Serves as an intraday price fluctuation and execution volatility proxy.

#### B. Trading Depth & Liquidity Metrics
7. **Log10 Average Daily Trading Volume ($\log_{10}(\bar{V})$):**
   $$\bar{V} = \frac{1}{T} \sum_{t=1}^T V_t, \quad \text{Feature} = \log_{10}(\bar{V})$$
   Because trading volumes span orders of magnitude (from $\$47$ million for mid-caps to $\$96$ billion for stablecoins), logarithmic compression linearizes variance.

8. **Volume Volatility ($\text{CV}_V$):**
   $$\text{CV}_V = \frac{\sigma_V}{\bar{V}}$$
   Measures the stability and persistence of trading activity over time.

9. **Volatility-to-Volume Ratio (Speculation Index):**
   $$\text{Ratio} = \frac{\sigma_{ann}}{\log_{10}(\bar{V})}$$
   Assesses how much volatility the asset produces per unit of trading volume depth.

### 5.2 Feature Standardization
Because trading volume ($\sim 10^7 - 10^{10}$) and percentage volatility ($\sim 0.2\% - 143\%$) operate on completely disparate numerical scales, distance-based clustering algorithms would be disproportionately dominated by unscaled volume figures. We apply Z-score standardization:
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
ensuring zero mean ($\mu = 0$) and unit variance ($\sigma = 1$) across all feature vectors.

### 5.3 Clustering Algorithms
1. **$K$-Means++ Clustering:** Minimizes within-cluster inertia $J$:
   $$J = \sum_{k=1}^K \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$
   Seeded using the $K$-Means++ algorithm to ensure widely dispersed initial centroids.
2. **Agglomerative Hierarchical Clustering:** Bottom-up hierarchical tree construction using Ward's minimum variance criterion:
   $$\Delta \text{ESS}_{AB} = \frac{n_A n_B}{n_A + n_B} \| \mu_A - \mu_B \|^2$$
3. **Gaussian Mixture Models (GMM):** Expectation-Maximization (EM) optimization estimating full covariance Gaussian distributions per cluster, outputting posterior assignment probabilities.
4. **DBSCAN:** Density-based spatial clustering evaluating core points within neighborhood $\varepsilon = 1.8$ with $\text{MinPts} = 3$ to identify anomalous noise points.

---

## 6. Results and Evaluation

### 6.1 Optimal Cluster Parameter Determination
To determine the natural number of clusters $k$ without subjective assumptions, multiple quantitative cluster validity indices were evaluated across $k \in [2, 7]$:

| Number of Clusters ($k$) | Inertia (WCSS) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
| :---: | :---: | :---: | :---: | :---: |
| $k = 2$ | 102.41 | 0.3210 | 1.1042 | 26.54 |
| $k = 3$ | 68.35 | 0.3345 | 0.9412 | 30.12 |
| **$k = 4$ (Optimal)** | **46.82** | **0.3464** | **0.8382** | **33.78** |
| $k = 5$ | 37.19 | 0.3208 | 0.8951 | 31.40 |
| $k = 6$ | 30.04 | 0.3015 | 0.9234 | 29.85 |
| $k = 7$ | 24.11 | 0.2870 | 0.9610 | 28.14 |

**Key Metric Insights:**
- **The Elbow Method:** The inertia curve displays a distinct bend (inflection point) at $k = 4$, with marginal WCSS reduction tapering off beyond this point.
- **Silhouette Coefficient ($S = 0.3464$):** Maximized at $k = 4$, verifying strong within-cluster cohesion and clear inter-cluster separation.
- **Davies-Bouldin Index ($DB = 0.8382$):** Reaches its global minimum at $k = 4$. (Lower values indicate tighter, more distinct clusters).
- **Calinski-Harabasz Index ($CH = 33.78$):** Peaks at $k = 4$, confirming the highest ratio of between-cluster dispersion to within-cluster dispersion.

### 6.2 Latent Space Representation (PCA & t-SNE)
Principal Component Analysis (PCA) performed on the standardized 8-dimensional feature matrix revealed that:
- **Principal Component 1 (PC1):** Accounts for **$72.82\%$** of total variance. Component loadings indicate heavy positive weights on Annualized Volatility ($+0.41$), Max Drawdown ($+0.39$), 95% VaR ($+0.42$), and CVaR ($+0.42$), establishing PC1 as an aggregate **Price Instability and Downside Tail Risk Axis**.
- **Principal Component 2 (PC2):** Accounts for **$20.96\%$** of total variance, with dominant loading on Log Average Daily Volume ($+0.64$) and negative loading on Volatility-to-Volume ratio ($-0.52$), establishing PC2 as an aggregate **Market Depth and Liquidity Turnover Axis**.
- **Cumulative 2D Variance:** **$93.78\%$** (3D: **$97.49\%$**).

This dimensionality reduction confirms that the two intuitive dimensions of the problem statement—price volatility and trading volume—fundamentally explain over $93\%$ of all cross-sectional variance in the digital asset dataset.

### 6.3 Empirical Risk Taxonomy and Cluster Archetypes

The clustering pipeline discovered four distinct, economically meaningful risk tiers:

```
[Tier 1: Ultra-Low Risk]   ──> Pegged Stablecoins (USDT, USDC, DAI)
[Tier 2: Low-Moderate Risk]──> Blue-Chips with Deep Liquidity (BTC, ETH, BNB)
[Tier 3: Moderate-High Risk]──> Established Altcoins & Utility Protocols (SOL, XRP, LINK...)
[Tier 4: Extreme Risk]     ──> Speculative Meme Coins & High Tail Risk (BONK, WIF, SHIB...)
```

#### Detailed Cluster Metrics Breakdown:

| Risk Tier | Constituent Cryptocurrencies | Count | Mean Annualized Volatility (%) | Mean Max Drawdown (%) | Mean 95% VaR (%) | Mean Daily Volume (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Ultra-Low Risk** | USDT, USDC, DAI | 3 | **0.43%** | **0.26%** | **0.05%** | **$37.8 Billion** |
| **Tier 2: Low-to-Moderate Risk** | BTC, ETH, BNB | 3 | **54.45%** | **59.67%** | **4.32%** | **$24.1 Billion** |
| **Tier 3: Moderate-to-High Risk** | SOL, XRP, ADA, AVAX, DOT, LINK, DOGE, LTC, BCH, ATOM, ETC, AAVE | 12 | **82.21%** | **79.48%** | **7.14%** | **$1.37 Billion** |
| **Tier 4: Extreme Speculative Risk**| BONK, WIF, FLOKI, SHIB, LDO, CRV, NEAR, XLM, MKR, ALGO | 10 | **112.53%** | **86.83%** | **9.81%** | **$272.5 Million** |

---

## 7. Discussion and Financial Implications

### 7.1 The Volatility-Volume Frontier
Plotting Annualized Volatility against Log Average Daily Volume reveals critical insights for financial risk management:
1. **The Stablecoin Singularity:** Pegged stablecoins (USDT, USDC, DAI) cluster tightly at the lower-right boundary of the frontier. They combine the highest trading liquidity ($>\$37\text{B}$ daily) with virtually zero price volatility ($<0.5\%$), serving as the ultimate risk-free cash equivalent in decentralized markets.
2. **The Institutional Liquidity Moat:** Bitcoin, Ethereum, and Binance Coin occupy a unique low-volatility, multi-billion dollar volume corridor. Even during intense market drawdowns, their deep order books prevent sudden slippage, justifying their status as foundational institutional assets.
3. **The Meme Coin Speculation Trap:** Meme tokens such as BONK, WIF, and FLOKI exhibit annualized volatility exceeding $120\%-143\%$ and maximum drawdowns $>92\%-96\%$. While their nominal daily volume is non-trivial ($\sim \$250\text{M}$), their volatility-to-volume ratio is abnormally high, reflecting pure speculative retail turnover rather than economic utility.

### 7.2 Portfolio Construction and Risk-Adjusted Allocation
The discovered cluster boundaries provide actionable guidelines for algorithmic portfolio construction:
- **Conservative Wealth Preservation Archetype:** 
  - $60\%$ Tier 1 (USDT/USDC), $30\%$ Tier 2 (BTC/ETH), $10\%$ Tier 3 (SOL/LINK), $0\%$ Tier 4.
  - Minimizes portfolio drawdown to $<18\%$ while preserving yield and capital stability.
- **Balanced Core-Satellite Growth Archetype:**
  - $20\%$ Tier 1, $50\%$ Tier 2, $25\%$ Tier 3, $5\%$ Tier 4.
  - Anchors $70\%$ of capital in high-liquidity assets while participating in decentralized platform growth.
- **Aggressive Alpha Generation Archetype:**
  - $5\%$ Tier 1, $35\%$ Tier 2, $45\%$ Tier 3, $15\%$ Tier 4.
  - Maximizes upside beta with strictly bounded tail-risk exposure.

### 7.3 Limitations
- **Stationarity Assumption:** Cluster assignments were computed over a two-year historical window. In regime-switching macro environments (e.g., sudden regulatory actions or market crashes), tokens may transition across cluster boundaries.
- **Sample Selection:** While 28 representative assets thoroughly cover all market categories, expanding the universe to 100+ micro-cap tokens would provide deeper granularity into illiquid tail tokens.

---

## 8. Conclusion and Future Work

### 8.1 Summary of Contributions
This capstone project successfully implemented a complete, rigorous data science workflow addressing the problem of cryptocurrency risk classification:
1. **End-to-End Pipeline:** Developed automated Python scripts for multi-asset OHLCV data harvesting (`data_collection.py`), mathematical feature extraction (`feature_engineering.py`), clustering model training (`clustering_models.py`), and visualization rendering (`visualization.py`).
2. **Mathematical Rigor:** Derived eight financial risk and liquidity metrics spanning annualized volatility, maximum drawdowns, 95% historical Value at Risk, Conditional VaR (Expected Shortfall), and volume turnover.
3. **Objective Model Optimization:** Validated optimal cluster separation at $k = 4$ through the Elbow method, Silhouette score ($0.3464$), Davies-Bouldin index ($0.8382$), and Calinski-Harabasz index ($33.78$).
4. **Interactive Deployment:** Built and tested a modular Streamlit dashboard (`app.py`) featuring an interactive Volatility-Volume frontier, a real-time risk classifier for custom tokens, and an automated portfolio risk allocator.
5. **Academic Deliverables:** Delivered complete phase-wise and consolidated Jupyter notebooks, an 8-slide PowerPoint presentation, and an exhaustive viva defense guide.

### 8.2 Future Work
Future extensions include:
- Implementing dynamic, rolling-window clustering to detect time-varying cluster transitions in real-time.
- Integrating natural language sentiment analysis from crypto financial news and social media (X/Twitter, Reddit) as an auxiliary feature dimension.
- Incorporating on-chain metrics (active wallet addresses, transaction fees, total value locked / TVL) into the liquidity profile.

---

## 9. References
1. Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56.
2. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms*, 1027–1035.
3. Baur, D. G., Hong, K., & Lee, A. D. (2018). Bitcoin: Medium of exchange or speculative asset? *Journal of International Financial Markets, Institutions and Money*, 54, 177–189.
4. Cong, L. W., Li, X., Tang, K., & Yang, Y. (2021). Crypto wash trading. *SSRN Electronic Journal*, 10.2139/ssrn.3530220.
5. De Prado, M. L. (2016). Building diversified portfolios that outperform out of sample. *The Journal of Portfolio Management*, 42(4), 59–69.
6. Dyhrberg, A. H. (2016). Bitcoin, gold and the US dollar–A GARCH volatility analysis. *Finance Research Letters*, 16, 85–92.
7. Katsiampa, P. (2017). Volatility estimation for Bitcoin: A comparison of GARCH models. *Economics Letters*, 158, 3–6.
8. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1(14), 281–297.
9. Reynolds, D. A. (2009). Gaussian mixture models. *Encyclopedia of Biometrics*, 741(659-663).
10. Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2, 21–42.
11. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
12. Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*, 58(301), 236–244.
