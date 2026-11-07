# CS2235: Data Science Capstone Project Proposal

## 1. Project Identification
- **Institution:** RV University, School of Computer Science & Engineering
- **Programme:** B.Sc (Hons.) Data Science
- **Course Title:** Data Science (Course Code: CS2235)
- **Semester:** III (Third Semester)
- **Course Instructors:** Prof. Muralidhar Billa, Prof. Thirukumaran Subbiramani
- **Project Title:** **Fintech Cryptocurrency Risk Profiling: Unsupervised K-Means Clustering of Digital Assets Based on Volatility and Trading Volume**
- **Date of Submission:** November 2026

---

## 2. Team Structure & Role Nominations
In accordance with Capstone Project Guidelines (Section 2):

| Role | Team Member | Primary Responsibilities |
| :--- | :--- | :--- |
| **Team Leader** | **Thanisha S Rao** | Overall project coordination, pipeline architecture, instructor communication, milestone delivery. |
| **Documentation Lead** | **Thanisha S Rao** | Comprehensive report, Markdown styling, references, and proposal drafting. |
| **Coding & Version Control Lead** | **Thanisha S Rao** | Git repository management, modular Python codebase (`src/`), Jupyter notebook automation. |
| **Presentation & Deployment Lead** | **Thanisha S Rao** | Interactive Streamlit deployment, 8-slide executive presentation, and viva voce defense preparation. |

---

## 3. Problem Statement & Motivation
### Problem Statement
> **"Fintech: Cryptocurrency Analysis — Grouping cryptocurrencies into similar risk categories based on how much their prices change (volatility/drawdown) and how much they are traded (trading volume)."**

### Context & Motivation (Module 5 Application in Finance)
The cryptocurrency market features thousands of actively traded digital assets. Unlike traditional equity markets where stocks are rated by standardized agencies or grouped into established sectors, cryptocurrencies lack clear risk ratings. 

Investors face significant challenges:
1. **Unclear Risk Levels:** Two cryptocurrencies may look similar on the surface, but one may experience much wider price swings and deeper drops.
2. **Trading Activity Differences:** Some coins trade billions of dollars each day, while others have low trading volume and are difficult to buy or sell without causing large price changes.

By applying **K-Means Clustering (Unsupervised Machine Learning, Module 4)** to historical price and volume data, this project groups cryptocurrencies into clear, data-driven risk categories without human bias.

---

## 4. Problem Classification & Syllabus Alignment
- **Type of Machine Learning Problem (Module 3):** **Unsupervised Learning (Clustering)**
- **Output Variable:** No predefined ground-truth label; discovery of natural groupings representing risk tiers.
- **Algorithm (Module 4):** **K-Means Clustering** (with K-Means++ initialization).
- **Dimensionality Reduction (Module 4):** **Principal Component Analysis (PCA basics in 2D)**.
- **Model Evaluation (Module 4):** **Elbow Method (Inertia/WCSS)** and **Silhouette Score**.

---

## 5. Dataset Identification & Data Sources (Module 1 & 2)
- **Data Source (Module 1):** Historical daily market data collected using the Yahoo Finance API (`yfinance`).
- **Data Measurement Scale (Module 1):**
  - **Ratio Scale:** Closing Price, High Price, Low Price, and Trading Volume (continuous numerical variables with a true zero).
  - **Nominal Scale:** Cryptocurrency Symbol, Name, and Asset Category.
- **Scope:** 28 diverse cryptocurrencies across 2 full years (731 daily records per coin, 20,468 total observations):
  - **USD Stablecoins:** USDT, USDC, DAI (baseline for low price change).
  - **Large-Cap Blue-Chips:** Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB), Solana (SOL), Ripple (XRP).
  - **Layer-1 & Layer-2 Ecosystems:** Cardano (ADA), Avalanche (AVAX), Polkadot (DOT), Near (NEAR), Cosmos (ATOM), Algorand (ALGO).
  - **DeFi & Utility Tokens:** Chainlink (LINK), Aave (AAVE), Maker (MKR), Lido DAO (LDO), Curve (CRV).
  - **Meme / High-Beta Coins:** Dogecoin (DOGE), Shiba Inu (SHIB), Bonk (BONK), Floki (FLOKI), dogwifhat (WIF).

---

## 6. Feature Engineering (Module 4)
To capture "how much prices change" and "how much they are traded", we engineer 5 clean features:

1. **Annualized Volatility (%):** Measures the spread of daily returns scaled over a 365-day trading year:
   $$\sigma_{ann} = \text{std}(R_t) \times \sqrt{365} \times 100$$
2. **Maximum Drawdown (MDD %):** The largest percentage drop from a peak price to the lowest trough.
3. **Mean Daily Spread (%):** Average intraday price variation:
   $$\text{Spread}_t = \frac{\text{High}_t - \text{Low}_t}{\text{Close}_t} \times 100$$
4. **Log10 Average Daily Volume (USD):** Log-transformed daily dollar volume to place billion-dollar coins and smaller coins on a comparable scale.
5. **Volatility-to-Volume Ratio:** Simple ratio of price fluctuation to volume depth.

---

## 7. Expected Outputs & Deliverables
1. **Cleaned Dataset & Features:** Raw historical CSV and scaled feature matrix.
2. **Documented Jupyter Notebooks:** Complete phase-by-phase notebooks (Phase 2 to Phase 5) and a master notebook.
3. **K-Means Model & Optimal k:** Elbow curve and Silhouette score evaluation showing optimal $k=4$.
4. **4 Risk Categories:**
   - *Tier 1: Ultra-Low Risk (Pegged Stablecoins)*
   - *Tier 2: Low-to-Moderate Risk (Blue-Chip Institutional Cryptos)*
   - *Tier 3: Moderate-to-High Risk (Established Ecosystem Tokens)*
   - *Tier 4: Extreme Speculative Risk (Meme & Highly Volatile Coins)*
5. **Interactive Streamlit App (`app.py`):** Visualizer for the Volatility vs. Volume frontier, live custom coin risk classifier, and portfolio risk allocator.
6. **Academic Report & Presentation:** Formal report and 8-slide presentation deck.
