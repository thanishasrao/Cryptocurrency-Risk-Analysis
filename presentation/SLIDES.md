# CS2235 Capstone Presentation Slides
## Fintech: Cryptocurrency Risk Profiling — Grouping Digital Assets into Empirical Risk Tiers
**RV University | B.Sc (Hons.) Data Science, Semester III**  
*Instructors:* Prof. Muralidhar Billa, Prof. Thirukumaran Subbiramani  
*Presenter / Researcher:* Thanisha S Rao  

---

### Slide 1: Title & Overview
- **Title:** Fintech: Cryptocurrency Risk Profiling
- **Subtitle:** Grouping Cryptocurrencies into Similar Risk Categories Based on Price Volatility Dynamics and Trading Volume Depth
- **Programme:** B.Sc (Hons.) Data Science (CS2235 Capstone Project)
- **Institution:** RV University, Bangalore
- **Core Concept:** Applying unsupervised machine learning to uncover empirical digital asset risk categories without human labelling bias.

---

### Slide 2: Problem Statement & Motivation
- **Problem Statement:** *"Grouping cryptocurrencies into similar risk categories based on how much their prices change and how much they are traded."*
- **Fintech Dilemma:**
  - Lack of traditional credit rating agencies (Moody's, S&P) in digital assets.
  - Extreme cross-sectional dispersion in volatility and liquidity.
  - The "Liquidity Illusion": High token prices do not imply high market depth.
- **Machine Learning Formulation:** Unsupervised Clustering (K-Means, Agglomerative Hierarchical, GMM, DBSCAN).

---

### Slide 3: Dataset Identification & Scope
- **Data Source:** Verified daily historical OHLCV data from Yahoo Finance API.
- **Time Horizon:** 2 full years (731 contiguous daily trading sessions).
- **Scale:** 28 diverse cryptocurrencies (20,468 total daily records).
- **Taxonomy Covered:**
  - *Stablecoins:* USDT, USDC, DAI
  - *Blue-Chips:* BTC, ETH, BNB
  - *Layer 1/2 Ecosystems:* SOL, XRP, ADA, AVAX, DOT, NEAR, ATOM, ALGO
  - *DeFi Protocols:* LINK, AAVE, MKR, LDO, CRV
  - *Meme & High-Beta:* DOGE, SHIB, BONK, FLOKI, WIF

---

### Slide 4: Quantitative Feature Engineering
- **Price Volatility Metrics ("How much prices change"):**
  - Annualized Volatility: $\sigma_{ann} = \sigma_{daily} \times \sqrt{365}$
  - Maximum Drawdown (MDD): Peak-to-trough worst collapse percentage
  - Value at Risk (95% VaR) & Conditional VaR (95% CVaR / Expected Shortfall)
  - Intraday High-Low Percentage Spread
- **Trading Volume Metrics ("How much they are traded"):**
  - Log10 Average Daily Trading Volume (USD)
  - Volume Volatility (Coefficient of Variation)
  - Amihud Illiquidity Ratio Proxy ($\text{median}(|R_t| / \text{Vol}_t) \times 10^9$)
  - Volatility-to-Volume Speculation Ratio

---

### Slide 5: Clustering Pipeline & Model Optimization
- **Z-Score Normalization:** StandardScaler ensures fair distance weighting across metrics.
- **Optimal K Determination:**
  - Elbow Method: Inflection point observed at $k = 4$.
  - Silhouette Score: Peak value of **0.3464**.
  - Davies-Bouldin Index: **0.8382** (lower is better).
  - Calinski-Harabasz Index: **33.78** (higher is better).
- **Model Comparison:** K-Means vs. Agglomerative Hierarchical (Ward Linkage) vs. GMM vs. DBSCAN.

---

### Slide 6: Discovered Risk Taxonomy (The 4 Tiers)
1. **Tier 1: Ultra-Low Risk (Pegged / Stablecoins)**
   - *Assets:* USDT, USDC, DAI (3 assets)
   - *Metrics:* Mean Volatility: 0.43% | Max Drawdown: 0.26% | High Volume
   - *Role:* Cash management, risk-off treasury buffer.
2. **Tier 2: Low-to-Moderate Risk (Blue-Chips / Institutional)**
   - *Assets:* BTC, ETH, BNB (3 assets)
   - *Metrics:* Mean Volatility: 54.4% | Max Drawdown: 59.7% | >$2B Daily Volume
   - *Role:* Foundational core portfolio anchors.
3. **Tier 3: Moderate-to-High Risk (Established Altcoins / DeFi)**
   - *Assets:* SOL, XRP, ADA, AVAX, DOT, LINK, DOGE, LTC, etc. (12 assets)
   - *Metrics:* Mean Volatility: 82.2% | Max Drawdown: 79.5% | Active Volume
   - *Role:* Tactical growth, smart contract utility.
4. **Tier 4: Extreme Speculative Risk (Meme & High-Beta)**
   - *Assets:* BONK, WIF, FLOKI, SHIB, LDO, CRV, NEAR, XLM, MKR, ALGO (10 assets)
   - *Metrics:* Mean Volatility: 112.5% | Max Drawdown: 86.8% | High Tail Risk
   - *Role:* High-risk speculation requiring strict stop-loss controls.

---

### Slide 7: Dimensionality Reduction & Core Findings
- **Principal Component Analysis (PCA):**
  - PC1: 72.8% Variance (Volatility & Tail Drawdown Loading)
  - PC2: 21.0% Variance (Trading Volume Depth & Liquidity Loading)
  - Cumulative 2D Variance Explained: **93.78%** (3D: **97.49%**).
- **Core Findings:**
  - High volume does not eliminate extreme price swings in hype tokens (e.g., DOGE, SHIB).
  - Stablecoins form an isolated, invariant micro-cluster.
  - Empirical cluster boundaries provide dynamic risk parameters for portfolio management.

---

### Slide 8: Interactive Deployment & Capstone Summary
- **Live Streamlit Deployment (`app.py`):**
  - Interactive Volatility-Volume Frontier visualizer.
  - Multi-factor risk filtering matrix.
  - Real-time Risk Predictor / Simulator for custom tokens.
  - Dynamic Portfolio Allocation Engine by investor risk tolerance.
- **Capstone Deliverables Delivered:**
  - ✅ Project Proposal (`docs/Project_Proposal.md`)
  - ✅ 5 Modular & Consolidated Jupyter Notebooks (`notebooks/`)
  - ✅ Academic Capstone Thesis Report (`docs/Final_Project_Report.md`)
  - ✅ PowerPoint Slide Deck (`presentation/crypto_risk_clustering_presentation.pptx`)
  - ✅ Streamlit Web Application (`app.py`)
  - ✅ Comprehensive Viva Voce Defense Guide (`docs/Viva_Voce_Preparation_Guide.md`)
