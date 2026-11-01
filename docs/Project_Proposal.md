# CS2235: Data Science Capstone Project Proposal

## 1. Project Identification
- **Institution:** RV University, School of Computer Science & Engineering
- **Programme:** B.Sc (Hons.) Data Science
- **Course Title:** Data Science (Course Code: CS2235)
- **Semester:** III (Third Semester)
- **Course Instructors:** Prof. Muralidhar Billa, Prof. Thirukumaran Subbiramani
- **Project Title:** **Fintech Cryptocurrency Risk Profiling: Unsupervised Clustering of Digital Assets Based on Volatility Dynamics and Liquidity Turnover**
- **Date of Submission:** November 2026

---

## 2. Team Structure & Role Nominations
In accordance with Capstone Project Guidelines (Section 2):

| Role | Team Member | Primary Responsibilities |
| :--- | :--- | :--- |
| **Team Leader** | **Thanisha S Rao** | Overall project coordination, pipeline architecture, instructor communication, milestone delivery. |
| **Documentation Lead** | **Thanisha S Rao** | Comprehensive 8–10 page technical report, LaTeX/Markdown styling, references, and proposal drafting. |
| **Coding & Version Control Lead** | **Thanisha S Rao** | Git repository management, modular Python codebase (`src/`), Jupyter notebook automation, reproducible execution. |
| **Presentation & Deployment Lead** | **Thanisha S Rao** | Interactive Streamlit deployment, 6–8 slide executive presentation, and viva voce defense preparation. |

---

## 3. Problem Statement & Motivation
### Problem Statement
> **"Fintech: Cryptocurrency Analysis — Grouping cryptocurrencies into similar risk categories based on how much their prices change (volatility/drawdown) and how much they are traded (liquidity volume/turnover)."**

### Context & Financial Motivation
The digital asset ecosystem has evolved into a multi-trillion dollar asset class characterized by extreme price dispersion, varying liquidity depth, and asymmetric risk profiles. Unlike traditional equity markets where instruments are categorized into well-defined sectors with institutional rating agencies (e.g., Moody's, S&P), cryptocurrencies lack standardized risk classifications.

Retail and institutional market participants face significant challenges:
1. **Hidden Tail Risk:** Tokens with superficially similar recent price appreciation often exhibit drastically different drawdown behavior, skewness, and downside tail risk.
2. **Liquidity Illusion:** Certain speculative micro-cap or meme tokens display high price surges but negligible genuine market depth, exposing investors to severe liquidity traps and slippage during market sell-offs.
3. **Portfolio Diversification Deficit:** Standard market-cap weighting treats volatile speculative assets similarly to established blue-chip utility tokens and stablecoins.

By applying **unsupervised machine learning (clustering)** to high-frequency historical trading and price dynamics, this project groups cryptocurrencies into distinct, mathematically substantiated risk categories without relying on subjective human labels.

---

## 4. Problem Classification
- **Type of Machine Learning Problem:** **Unsupervised Learning (Clustering)**
- **Output Variable:** No predefined ground-truth label; discovery of natural groupings/clusters representing risk tiers.
- **Algorithms Evaluated:**
  - K-Means Clustering (with K-Means++ initialization)
  - Agglomerative Hierarchical Clustering (Ward's linkage, Euclidean metric)
  - Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
  - Gaussian Mixture Models (GMM - Probabilistic Soft Clustering)

---

## 5. Dataset Identification & Data Sources
- **Data Source:** Verified historical daily market data extracted via Yahoo Finance API (`yfinance`), corroborated with CoinMarketCap/CoinGecko open financial metrics.
- **Scope of Coverage:** 35+ diverse cryptocurrencies spanning:
  - **Blue-Chip Digital Currencies:** Bitcoin (BTC), Ethereum (ETH)
  - **Major Layer-1 & Layer-2 Platforms:** Solana (SOL), Binance Coin (BNB), Cardano (ADA), Ripple (XRP), Avalanche (AVAX), Polygon (MATIC), Polkadot (DOT), Near Protocol (NEAR), Cosmos (ATOM), Aptos (APT), Sui (SUI)
  - **DeFi & Infrastructure Protocols:** Chainlink (LINK), Uniswap (UNI), Aave (AAVE), Maker (MKR), Lido DAO (LDO)
  - **High-Beta / Speculative / Meme Tokens:** Dogecoin (DOGE), Shiba Inu (SHIB), Pepe (PEPE), Bonk (BONK), Floki (FLOKI)
  - **Payment & Legacy Coins:** Litecoin (LTC), Bitcoin Cash (BCH), Monero (XMR), Stellar (XLM)
  - **Pegged Stablecoins (Zero-volatility Benchmark):** Tether (USDT), USD Coin (USDC), Dai (DAI)
- **Time Horizon:** Multi-year daily OHLCV (Open, High, Low, Close, Volume) data providing robust exposure across bullish, bearish, and consolidation cycles.

---

## 6. Mathematical Risk & Liquidity Metrics (Feature Engineering)
To rigorously quantify "how much prices change" and "how much they are traded", we engineer the following quantitative metrics:

### A. Price Volatility & Downside Risk Features
1. **Annualized Volatility ($\sigma_{ann}$):**
   $$\sigma_{ann} = \sigma_{daily} \times \sqrt{365}$$
2. **Maximum Drawdown (MDD):**
   $$MDD = \min_{t \in [0, T]} \left( \frac{P_t - \max_{\tau \le t} P_\tau}{\max_{\tau \le t} P_\tau} \right)$$
3. **Historical Value at Risk (VaR 95%):** The 5th percentile of daily return distribution.
4. **Conditional Value at Risk (CVaR 95% / Expected Shortfall):** Average loss beyond the 95% VaR threshold.
5. **Downside Deviation & Sortino Ratio:** Volatility penalizing only negative returns below risk-free rate.
6. **Return Skewness & Kurtosis:** Measuring asymmetric loss probability and heavy fat tails.
7. **Average Daily High-Low Price Range (%):**
   $$\text{Range}_t = \frac{\text{High}_t - \text{Low}_t}{\text{Close}_t} \times 100$$

### B. Trading Depth & Liquidity Metrics
1. **Log Mean Daily Trading Volume ($\log_{10}(\bar{V})$):** Captures total dollar liquidity across orders of magnitude.
2. **Volume Volatility (Coefficient of Variation of Volume):** Stability of trading activity over time.
3. **Volume-to-Volatility Liquidity Proxy:** Ratio assessing how much volume is required to absorb price swings.
4. **Amihud Illiquidity Ratio Proxy:** Price impact per unit of trading volume:
   $$\text{Illiq}_i = \frac{1}{T} \sum_{t=1}^T \frac{|R_{i,t}|}{\text{Volume}_{i,t}}$$

---

## 7. Expected Outputs & Deliverables
1. **Cleaned & Feature-Engineered Datasets:** Raw time-series CSV and normalized multi-factor risk feature matrix.
2. **Comprehensive Jupyter Notebooks:** Complete, modular, reproducible workflow documenting Phase 2 to Phase 5.
3. **Optimal Cluster Determination:** Elbow curve analysis, Silhouette score optimization, Davies-Bouldin index, and Calinski-Harabasz evaluation.
4. **Risk Profile Classification:** Archetype mapping:
   - *Cluster 0: Pegged / Cash-Equivalent Assets (Ultra-Low Risk)*
   - *Cluster 1: Blue-Chip Institutional Assets (Low-to-Moderate Risk, Deep Liquidity)*
   - *Cluster 2: High-Growth Utility & Mid-Cap Ecosystems (Moderate-to-High Risk)*
   - *Cluster 3: Speculative / High-Beta / Meme Assets (Extreme Risk, Severe Tail Volatility)*
5. **Interactive Streamlit Web Dashboard:** Interactive 2D/3D PCA & t-SNE visualizer, custom crypto risk scorer, and risk-adjusted portfolio simulator.
6. **Academic Capstone Report & Slide Deck:** 8–10 page formal report following university guidelines and 6–8 slide viva presentation.

---

## 8. Milestone Timeline (CS2235 Alignment)

| Phase | Milestone Description | Target Status |
| :--- | :--- | :--- |
| **Phase 1** | Topic Selection, Dataset Identification, Framing Problem Statement | Completed |
| **Phase 2** | Data Collection, OHLCV Ingestion, Quality Checks, Missing Value Handling | Completed |
| **Phase 3** | Quantitative Risk Feature Extraction, Log Transformation, Scaling (Robust/Standard) | Completed |
| **Phase 4** | Model Implementation: K-Means, Hierarchical, DBSCAN, GMM, Metric Evaluation | Completed |
| **Phase 5** | Dimensionality Reduction (PCA, t-SNE), Risk Categorization, Streamlit Web App | Completed |
| **Phase 6** | Academic Report (8–10 pages), Slide Deck (6–8 slides), Viva Voce Defense Guide | Completed |
