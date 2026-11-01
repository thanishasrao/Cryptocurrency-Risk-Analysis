# CS2235: Data Science Capstone Project - Viva Voce Defense Guide

**RV University | School of Computer Science & Engineering**  
**Course:** Data Science Capstone Project (CS2235)  
**Topic:** Fintech: Cryptocurrency Risk Profiling — Grouping Cryptocurrencies into Similar Risk Categories Based on Price Volatility Dynamics and Trading Volume Depth  
**Student Researcher:** Thanisha S Rao  
**Weightage in Final Evaluation:** 10 Marks out of 25 (40% of Total Grade!)  

---

## 1. Core Foundational Questions

### Q1: What is the core problem statement of your project, and why is it important in Fintech?
**Answer:**
> "The problem statement is to group cryptocurrencies into empirical risk categories based on how much their prices change (volatility, drawdowns, and tail risk) and how much they are traded (dollar trading volume, turnover velocity, and liquidity impact).
> 
> Unlike traditional equities or fixed-income instruments—which have institutional credit ratings (S&P, Moody's) and formalized market cap categorizations—cryptocurrencies operate without standardized risk disclosures. Market participants frequently fall into the 'liquidity illusion,' mistaking high token market caps for genuine market depth. Our machine learning framework clusters digital assets into mathematically substantiated risk tiers without subjective human bias."

---

### Q2: Why did you formulate this as an Unsupervised Learning problem rather than Supervised Classification?
**Answer:**
> "In supervised learning, models require ground-truth target labels (e.g., $y \in \{\text{'Low Risk'}, \text{'High Risk'}\}$). However, in digital assets, there are no universally agreed ground-truth risk labels. Any manual labelling would introduce human subjectivity and confirmation bias.
> 
> By framing this as unsupervised clustering, we let the underlying geometry and distribution of the quantitative features reveal natural groupings and latent risk boundaries directly from market data."

---

### Q3: What datasets did you collect, and how did you verify data quality?
**Answer:**
> "We collected continuous daily OHLCV (Open, High, Low, Close, Volume) market data for 28 representative cryptocurrencies over a 2-year observation window (731 contiguous daily sessions per asset, totaling 20,468 observations) using the Yahoo Finance API, cross-referenced with CoinGecko.
> 
> Data quality checks included:
> 1. Verifying contiguous calendar trading days (since crypto markets trade 24/7/365).
> 2. Ensuring zero missing values (`NaN = 0`).
> 3. Enforcing physical price consistency ($\text{High}_t \ge \text{Close}_t \ge \text{Low}_t$).
> 4. Auditing for zero-volume days to prevent division-by-zero artifacts."

---

## 2. Feature Engineering & Mathematical Questions

### Q4: Explain the mathematical features you engineered to capture "how much prices change".
**Answer:**
> "To comprehensively measure price instability and downside risk, we engineered:
> 1. **Annualized Volatility ($\sigma_{ann}$):** Computed as daily return standard deviation scaled by the crypto annualization factor:
>    $$\sigma_{ann} = \sigma_{daily} \times \sqrt{365} \times 100$$
> 2. **Maximum Drawdown (MDD):** The absolute largest peak-to-trough price collapse over the entire 2-year period:
>    $$\text{MDD} = \left| \min_{t} \left( \frac{P_t - \max_{\tau \le t} P_\tau}{\max_{\tau \le t} P_\tau} \right) \right|$$
> 3. **95% Historical Value at Risk ($\text{VaR}_{95}$):** The 5th percentile of daily losses, indicating the loss threshold exceeded on only 5% of trading days.
> 4. **95% Conditional Value at Risk ($\text{CVaR}_{95}$ / Expected Shortfall):** The conditional expectation of loss given that the 95% VaR threshold is breached. This directly quantifies downside tail fatness.
> 5. **Average Daily High-Low Spread Percentage:** Measuring intraday volatility and bid-ask price fluctuation."

---

### Q5: Explain the features you engineered to capture "how much they are traded".
**Answer:**
> "To quantify trading activity and market depth, we engineered:
> 1. **$\log_{10}$ Average Daily Trading Volume:** Because daily dollar volumes span several orders of magnitude (from $\$47$ million for Maker to $\$96$ billion for Tether), applying a logarithmic transformation linearizes the variance and prevents extreme skewness.
> 2. **Volume Volatility (Coefficient of Variation $\text{CV}_V = \sigma_V / \bar{V}$):** Measures whether trading activity is sustained and predictable, or sporadic and prone to flash-hype spikes.
> 3. **Amihud Illiquidity Ratio Proxy:** Evaluates daily price impact per dollar volume:
>    $$\text{Amihud} = \text{median}\left(\frac{|R_t|}{\text{Volume}_t}\right) \times 10^9$$
> 4. **Volatility-to-Volume Ratio:** A speculation index measuring how much price fluctuation occurs relative to trading volume depth."

---

### Q6: Why was feature scaling essential, and which scaler did you choose?
**Answer:**
> "Features such as daily trading volume (measured in hundreds of millions or billions of dollars) and daily return percentages (measured from $0.05$ to $10$) operate on vastly different orders of magnitude. In Euclidean distance-based algorithms like K-Means and Hierarchical Clustering, unscaled volume would completely dominate distance calculations, rendering volatility metrics irrelevant.
> 
> We applied **StandardScaler** ($Z = \frac{X - \mu}{\sigma}$), ensuring every feature has zero mean and unit variance. This guarantees that price volatility and trading liquidity exert balanced, equal weighting in multi-dimensional space."

---

## 3. Machine Learning & Clustering Methodology Questions

### Q7: Which clustering algorithms did you implement, and how do they compare?
**Answer:**
> "We implemented four distinct clustering paradigms:
> 1. **$K$-Means++:** Distance-based centroid partitioning. We used K-Means++ initialization, which seeds initial centroids probabilistically based on squared distance from existing centers, preventing suboptimal convergence.
> 2. **Agglomerative Hierarchical Clustering:** Bottom-up linkage using Ward's minimum variance criterion, which minimizes total within-cluster variance. It provides full interpretability via dendrograms.
> 3. **Gaussian Mixture Models (GMM):** Probabilistic soft clustering using Expectation-Maximization (EM), fitting multivariate Gaussian distributions with full covariance matrices.
> 4. **DBSCAN:** Density-based clustering that groups core points within neighborhood $\varepsilon$ and flags outliers as noise.
> 
> All four algorithms demonstrated consistent grouping boundaries, validating the structural robustness of our clusters."

---

### Q8: How did you mathematically justify choosing $k = 4$ clusters?
**Answer:**
> "We did not arbitrarily pick $k=4$; we validated it using four complementary quantitative validation metrics across $k \in [2, 7]$:
> 1. **The Elbow Method (Inertia / WCSS):** Displays an elbow inflection point at $k = 4$, where adding further clusters yields diminishing returns in within-cluster variance reduction.
> 2. **Silhouette Coefficient:** Reached its global maximum of **$0.3464$** at $k = 4$. (The Silhouette score measures how close each point is to points in its own cluster versus the nearest neighboring cluster; higher is better).
> 3. **Davies-Bouldin Index:** Reached its global minimum of **$0.8382$** at $k = 4$. (Measures similarity between clusters based on scatter and separation; lower is better).
> 4. **Calinski-Harabasz Index:** Peaked at **$33.78$** at $k = 4$ (measures between-cluster dispersion relative to within-cluster dispersion; higher is better)."

---

### Q9: What did Principal Component Analysis (PCA) reveal about your data?
**Answer:**
> "PCA revealed that our 8-dimensional feature space can be effectively condensed into two principal components with **$93.78\%$** total explained variance (and $97.49\%$ in 3D):
> - **PC1 ($72.82\%$ variance):** Loaded heavily on Annualized Volatility ($+0.41$), Max Drawdown ($+0.39$), 95% VaR ($+0.42$), and CVaR ($+0.42$). It represents the **Price Instability & Downside Risk Axis**.
> - **PC2 ($20.96\%$ variance):** Loaded heavily on Log Average Daily Volume ($+0.64$) and negatively on Volatility-to-Volume ratio ($-0.52$). It represents the **Trading Liquidity & Market Depth Axis**.
> 
> This proves that the two dimensions of our problem statement—how much prices change and how much they are traded—capture nearly all the fundamental variance in digital asset markets."

---

## 4. Financial Findings & Deployment Questions

### Q10: What are the four discovered risk tiers, and what assets belong to them?
**Answer:**
> "The four discovered tiers are:
> 1. **Tier 1: Ultra-Low Risk (Pegged Stablecoins - 3 assets):** USDT, USDC, DAI.
>    - *Characteristics:* Mean volatility $0.43\%$, maximum drawdown $0.26\%$, massive daily volume ($>\$37\text{B}$).
> 2. **Tier 2: Low-to-Moderate Risk (Institutional Blue-Chips - 3 assets):** BTC, ETH, BNB.
>    - *Characteristics:* Mean volatility $54.45\%$, maximum drawdown $59.67\%$, deep multi-billion-dollar daily volume ($>\$24\text{B}$).
> 3. **Tier 3: Moderate-to-High Risk (Established Altcoins & Utility Protocols - 12 assets):** SOL, XRP, ADA, AVAX, DOT, LINK, DOGE, LTC, BCH, ATOM, ETC, AAVE.
>    - *Characteristics:* Mean volatility $82.21\%$, maximum drawdown $79.48\%$, solid liquidity ($\sim \$1.37\text{B}$ daily).
> 4. **Tier 4: Extreme Speculative Risk (Meme Tokens & High Tail Beta - 10 assets):** BONK, WIF, FLOKI, SHIB, LDO, CRV, NEAR, XLM, MKR, ALGO.
>    - *Characteristics:* Mean volatility $112.53\%$, maximum drawdown $86.83\%$, severe tail risk and erratic volume spikes."

---

### Q11: How does your Streamlit deployment (`app.py`) translate this model into a practical tool?
**Answer:**
> "The Streamlit dashboard operationalizes the research into four interactive features:
> 1. **Market Risk Frontier:** Visualizes volatility against trading volume with real-time asset annotations.
> 2. **Interactive Risk Matrix:** Allows filtering and sorting assets by risk tier, category, and drawdown metrics.
> 3. **Live Risk Classifier:** An interactive tool where an investor enters expected price volatility (%) and daily volume ($) of an unlisted or new cryptocurrency, and our trained K-Means pipeline predicts its risk tier in real time.
> 4. **Portfolio Risk Allocator:** Recommends optimal portfolio weights across the four tiers based on the user's risk appetite (Conservative, Balanced, Aggressive, Speculative)."

---

### Q12: What are the limitations of your approach, and how would you extend it?
**Answer:**
> "The primary limitation is the stationarity assumption: our clusters represent an aggregate 2-year window. In real markets, assets can experience regime shifts during major macro liquidity events.
> 
> As future work, we propose:
> 1. Implementing rolling-window dynamic clustering to track real-time transitions between risk tiers.
> 2. Incorporating on-chain metrics such as active wallet counts, gas fee consumption, and Total Value Locked (TVL).
> 3. Integrating NLP sentiment signals from financial news and developer activity."
