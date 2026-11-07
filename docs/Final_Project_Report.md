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
   - 5.1 Data Collection via API (Module 1)
   - 5.2 Building the ETL Pipeline & Quality Checks (Module 2)
   - 5.3 Feature Engineering & Preprocessing (Module 4)
   - 5.4 Feature Scaling & Dimensionality Reduction via PCA Basics (Module 4)
   - 5.5 K-Means Clustering & Hyperparameter Tuning (Module 4)
   - 5.6 Model Evaluation: Elbow Method & Silhouette Score (Module 4)
6. [Results and Evaluation](#6-results-and-evaluation)
   - 6.1 Optimal Cluster Parameter Determination ($k=4$)
   - 6.2 2D PCA Latent Space Visualization
   - 6.3 Discovered Risk Taxonomy and Categories
7. [Discussion and Financial Applications](#7-discussion-and-financial-applications)
   - 7.1 Volatility vs. Trading Volume Frontier
   - 7.2 Portfolio Construction & Asset Allocation (Module 5)
   - 7.3 Streamlit Deployment (Module 5)
8. [Conclusion and Future Work](#8-conclusion-and-future-work)
9. [References](#9-references)

---

## 1. Abstract
The digital currency market has grown into a major financial asset class consisting of thousands of publicly traded tokens. Unlike traditional stock markets where companies are categorized by sector and credit ratings, cryptocurrencies lack standardized risk classifications. As a result, investors face challenges distinguishing between stable assets, large established utility platforms, and highly volatile speculative tokens.

This capstone project presents an end-to-end data science project designed to solve the problem statement: **"Grouping cryptocurrencies into similar risk categories based on how much their prices change and how much they are traded."** Using continuous 2-year daily historical market data (731 days, 20,468 observations) collected via the Yahoo Finance API across 28 representative cryptocurrencies, we build an ETL pipeline to extract five core financial risk and trading volume features.

We implement **K-Means Clustering** as taught in **CS2235 Module 4**. Using Z-score normalization (`StandardScaler`), we tune the hyperparameter $k$ from 2 to 7. The optimal number of clusters is verified at $k = 4$ using the **Elbow Method (WCSS / Inertia)** and **Silhouette Score ($S = 0.3868$)**. Furthermore, **Principal Component Analysis (PCA basics in 2D)** explains **$96.94\%$** of total variance, providing clear two-dimensional separation. The 28 digital assets are grouped into four clear risk categories: Tier 1 (Ultra-Low Risk / Pegged Stablecoins), Tier 2 (Low-to-Moderate Risk / Blue-Chips), Tier 3 (Moderate-to-High Risk / Utility Ecosystems), and Tier 4 (Extreme Speculative Risk / Meme Coins). Finally, an interactive web dashboard is deployed using **Streamlit (`app.py`)** to provide live risk prediction and portfolio allocation.

---

## 2. Introduction and Problem Statement

### 2.1 Context and Background (Module 1 & 5)
In quantitative finance and data science, grouping assets into clear risk tiers is fundamental for investment planning and risk management. In traditional stock markets, investors categorize stocks into large-cap, mid-cap, and small-cap brackets, alongside well-established sector classifications.

In cryptocurrency markets, digital assets trade continuously 24 hours a day, 7 days a week across global decentralized exchanges. Investors frequently rely only on market capitalization or recent prices, which can be misleading because:
1. **Price Volatility:** Many altcoins experience extreme daily price fluctuations of 10% to 30%, whereas major assets experience more moderate swings.
2. **Trading Volume Depth:** High market prices do not necessarily mean high trading activity. Coins with low daily trading volume can suffer severe price slippage when sold.

### 2.2 Problem Statement & Objectives
The problem statement assigned for this capstone is:
> **"Fintech: Cryptocurrency Analysis — Grouping cryptocurrencies into similar risk categories based on how much their prices change and how much they are traded."**

In alignment with the CS2235 syllabus, the primary objectives are:
- **Module 1:** Collect historical daily price and volume data using the Yahoo Finance API.
- **Module 2:** Build a batch ETL pipeline to clean the data and verify data quality (zero missing values, timestamp alignment).
- **Module 3:** Frame the problem as **Unsupervised Learning** because there are no ground-truth target risk labels.
- **Module 4:** Engineer quantitative features, apply `StandardScaler`, reduce dimensions with 2D PCA, train **K-Means Clustering**, and evaluate clusters using the Elbow Method and Silhouette Score.
- **Module 5:** Build and deploy an interactive **Streamlit dashboard (`app.py`)** for cryptocurrency risk profiling.

---

## 3. Related Work and Background
Cryptocurrency risk research traditionally focuses on measuring volatility and returns (Dyhrberg, 2016). Early studies demonstrated that Bitcoin and Ethereum exhibit distinct return distributions compared to traditional equities. 

In data science, **K-Means clustering** (MacQueen, 1967) is one of the most widely used unsupervised learning algorithms for partitioning numerical data into $k$ distinct groups. Arthur & Vassilvitskii (2007) introduced the **K-Means++** initialization algorithm, which places initial cluster centroids far apart, leading to faster and more reliable convergence. For cluster evaluation, Rousseeuw (1987) introduced the **Silhouette Coefficient**, which measures how similar an observation is to its own cluster compared to neighboring clusters. Dimensionality reduction via **Principal Component Analysis (PCA)** allows multi-dimensional financial features to be visualized in a 2D plane while preserving the vast majority of variance.

---

## 4. Dataset Description and Quality Audit

### 4.1 Data Sources & Measurement Scales (Module 1)
Historical daily market data was acquired using the Python `yfinance` library, covering a 2-year window (731 contiguous daily sessions per asset, from late 2024 to late 2026), producing 20,468 total records.

As taught in **Module 1 (Types of Data by Measurement Scale)**:
- **Ratio Scale Data:** Open Price, High Price, Low Price, Close Price, and Daily Trading Volume. These are continuous numerical features with a true meaningful zero point.
- **Nominal Scale Data:** Asset Symbol (e.g., `BTC`, `ETH`), Asset Name, and Category (e.g., Stablecoin, Layer-1).

The 28 digital assets were selected across representative market sectors:
1. **USD Stablecoins (Control Baseline):** USDT, USDC, DAI.
2. **Mega-Cap Blue-Chips:** Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB), Solana (SOL), Ripple (XRP).
3. **Layer-1 & Layer-2 Platforms:** Cardano (ADA), Avalanche (AVAX), Polkadot (DOT), Near (NEAR), Cosmos (ATOM), Algorand (ALGO).
4. **DeFi & Utility Protocols:** Chainlink (LINK), Aave (AAVE), Maker (MKR), Lido DAO (LDO), Curve (CRV).
5. **Meme & High-Beta Coins:** Dogecoin (DOGE), Shiba Inu (SHIB), Bonk (BONK), Floki (FLOKI), dogwifhat (WIF), Stellar (XLM), Litecoin (LTC), Bitcoin Cash (BCH), Ethereum Classic (ETC).

### 4.2 Data Cleansing & Validation Checks (Module 2)
In accordance with **Module 2 (Data Validation & Quality Checks)**:
- **Missing Value Check:** Verified zero `NaN` values across all 20,468 daily records.
- **Logical Price Consistency:** Verified that $\text{High}_t \ge \text{Close}_t$ and $\text{Low}_t \le \text{Close}_t$ across all records.
- **Timestamp Standardization:** Standardized all timestamps to UTC calendar dates.

---

## 5. Methodology

The end-to-end data science lifecycle followed in this project:

```
[Module 1: Data Collection (Yahoo Finance API)]
                   │
                   ▼
[Module 2: Batch ETL Pipeline & Data Quality Checks]
                   │
                   ▼
[Module 4: Feature Engineering (Price Change & Trading Volume)]
                   │
                   ▼
[Module 4: Feature Scaling (StandardScaler)]
                   │
                   ▼
[Module 4: PCA Basics (2D Dimensionality Reduction)]
                   │
                   ▼
[Module 4: K-Means Clustering & Hyperparameter Tuning]
                   │
                   ▼
[Module 4: Model Evaluation (Elbow Method & Silhouette Score)]
                   │
                   ▼
[Module 5: Streamlit Web Dashboard Deployment (app.py)]
```

### 5.1 Mathematical Feature Engineering (Module 4)
We engineer 5 intuitive features directly addressing the problem statement:

#### A. Price Change Features
1. **Daily Return ($R_t$):**
   $$R_t = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$$
2. **Annualized Volatility ($\sigma_{ann}$):**
   $$\sigma_{ann} = \text{std}(R_t) \times \sqrt{365} \times 100$$
   Digital assets trade every day of the year, so we scale daily return standard deviation by $\sqrt{365}$.
3. **Maximum Drawdown (MDD %):**
   $$\text{Peak}_t = \max_{\tau \le t} \text{Close}_\tau, \quad \text{Drawdown}_t = \frac{\text{Close}_t - \text{Peak}_t}{\text{Peak}_t} \times 100$$
   $$MDD = \left| \min_{t} (\text{Drawdown}_t) \right|$$
   Measures the worst possible peak-to-trough price decline over the 2 years.
4. **Mean Daily Spread (%):**
   $$\text{Spread}_t = \frac{\text{High}_t - \text{Low}_t}{\text{Close}_t} \times 100$$
   Average daily price range within each day.

#### B. Trading Volume Features
5. **Log10 Average Daily Volume:**
   $$\text{Log\_Avg\_Volume} = \log_{10}(\text{mean}(\text{Volume}_t))$$
   Because daily trading volumes range from millions of dollars to over $\$90$ billion, logarithmic transformation brings values into a manageable linear scale.
6. **Volatility-to-Volume Ratio:**
   $$\text{Ratio} = \frac{\sigma_{ann}}{\text{Log\_Avg\_Volume}}$$
   An intuitive index of price swings relative to daily volume depth.

### 5.2 Feature Scaling (StandardScaler, Module 4)
Because trading volume and percentage volatility are on different numerical scales, we apply **Z-score standardization**:
$$z = \frac{x - \mu}{\sigma}$$
This ensures all features have a mean of 0 and standard deviation of 1.

### 5.3 K-Means Clustering & Model Evaluation (Module 4)
- **Algorithm:** K-Means clustering partitions $n$ assets into $k$ clusters by minimizing the within-cluster sum of squares (WCSS / Inertia):
  $$\text{WCSS} = \sum_{k=1}^K \sum_{x \in C_k} \| x - \mu_k \|^2$$
- **Initialization:** K-Means++ initialization is used to select well-separated initial cluster centers.
- **Hyperparameter Tuning:** We evaluate $k \in [2, 7]$ using:
  1. **The Elbow Method (Inertia):** Identifying the value of $k$ where the curve bends.
  2. **Silhouette Score:** Measuring cluster cohesion and separation:
     $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

---

## 6. Results and Evaluation

### 6.1 Optimal Cluster Parameter Determination ($k = 4$)
Evaluating K-Means across $k \in [2, 7]$ produced the following results:

| Number of Clusters ($k$) | Inertia (WCSS) | Silhouette Score |
| :---: | :---: | :---: |
| $k = 2$ | 54.12 | 0.3512 |
| $k = 3$ | 32.84 | 0.3640 |
| **$k = 4$ (Optimal)** | **20.15** | **0.3868 (Peak)** |
| $k = 5$ | 15.62 | 0.3421 |
| $k = 6$ | 12.30 | 0.3150 |
| $k = 7$ | 9.85 | 0.2980 |

- **Elbow Method:** The inertia drops sharply from $k=2$ to $k=4$ and levels off thereafter, showing a clear elbow at $k=4$.
- **Silhouette Score:** The silhouette score peaks at **$0.3868$** at $k = 4$, confirming that four clusters provide the clearest separation.

### 6.2 2D PCA Dimensionality Reduction (Module 4)
Applying PCA basics to project the 5 features onto a 2D plane:
- **Principal Component 1 (PC1):** Explains **$79.84\%$** of variance (loads on Price Volatility and Max Drawdown).
- **Principal Component 2 (PC2):** Explains **$17.10\%$** of variance (loads on Trading Volume Depth).
- **Total Explained Variance:** **$96.94\%$** in 2 dimensions, confirming that price change and volume account for almost all variance.

### 6.3 Discovered Risk Taxonomy (The 4 Categories)

| Risk Category | Constituent Digital Assets | Coin Count | Mean Volatility (%) | Mean Max Drawdown (%) | Mean Daily Volume (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tier 1: Ultra-Low Risk** | USDT, USDC, DAI | 3 | **0.43%** | **0.26%** | **$37.8 Billion** |
| **Tier 2: Low-to-Moderate Risk** | BTC, ETH, BNB, SOL, XRP | 5 | **64.80%** | **65.50%** | **$16.5 Billion** |
| **Tier 3: Moderate-to-High Risk** | ADA, AVAX, DOT, LINK, DOGE, LTC, BCH, ATOM, ETC, AAVE, MKR, CRV | 12 | **86.80%** | **83.20%** | **$785.4 Million** |
| **Tier 4: Extreme Speculative Risk**| BONK, WIF, FLOKI, SHIB, NEAR, XLM, ALGO, LDO | 8 | **114.20%** | **87.70%** | **$248.1 Million** |

---

## 7. Discussion and Financial Applications (Module 5)

### 7.1 Volatility vs. Volume Frontier
Plotting price volatility against trading volume confirms the problem statement:
1. **Stablecoins:** Form a compact group at the bottom right with massive daily volume ($>\$37\text{B}$) and near-zero volatility ($0.43\%$), acting as cash equivalents.
2. **Blue-Chips (BTC, ETH, BNB, SOL, XRP):** High volume ($>\$16\text{B}$) provides liquidity depth that buffers against extreme crashes.
3. **Speculative Meme Tokens (BONK, WIF, FLOKI):** Experience extreme annualized volatility ($>114\%$) and severe drawdowns ($>87\%$).

### 7.2 Portfolio Construction (Module 5 Application in Finance)
The discovered risk tiers help guide portfolio diversification:
- **Conservative Portfolio:** 60% Tier 1 (Stablecoins), 30% Tier 2 (Blue-chips), 10% Tier 3 (Utility).
- **Balanced Portfolio:** 20% Tier 1, 50% Tier 2, 25% Tier 3, 5% Tier 4.
- **Aggressive Growth Portfolio:** 5% Tier 1, 35% Tier 2, 45% Tier 3, 15% Tier 4.

### 7.3 Streamlit Deployment (Module 5)
An interactive web application was deployed via **Streamlit (`app.py`)**, featuring:
- Interactive Volatility vs. Volume frontier.
- Searchable asset table with all engineered features.
- Live Risk Classifier: Users enter any cryptocurrency's expected volatility and volume to predict its risk tier.
- Portfolio Risk Allocator: Custom asset weighting based on investor risk preference.

---

## 8. Conclusion and Future Work
This project completed an end-to-end data science capstone aligned directly with **Modules 1 through 5 of CS2235**:
- Ingested 20,468 records via API (Module 1).
- Built a batch ETL pipeline with quality validation checks (Module 2).
- Framed the task as Unsupervised Clustering (Module 3).
- Engineered price and volume features, applied `StandardScaler`, 2D PCA, and K-Means with optimal $k=4$ confirmed by Elbow and Silhouette score $0.3868$ (Module 4).
- Deployed an interactive Streamlit application for finance (Module 5).

---

## 9. References
1. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms*, 1027–1035.
2. Dyhrberg, A. H. (2016). Bitcoin, gold and the US dollar–A volatility analysis. *Finance Research Letters*, 16, 85–92.
3. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1(14), 281–297.
4. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
