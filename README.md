# 🪙 Cryptocurrency Risk Profiling (CS2235 Capstone)
### Grouping Cryptocurrencies into Risk Categories Using K-Means Clustering

A first-year Data Science capstone project for **CS2235 at RV University**, grouping 28 cryptocurrencies into 4 clear risk categories based on **how much their prices change** and **how much they are traded**.

---

## 📌 Problem Statement
> **"Fintech: Cryptocurrency Analysis — Grouping cryptocurrencies into similar risk categories based on how much their prices change (volatility, drawdowns) and how much they are traded (trading volume)."**

---

## 🎓 Syllabus Alignment (Modules 1 to 5)

This project strictly follows the topics taught in **CS2235**:

| Course Module | Topic in Syllabus | How We Used It |
| :--- | :--- | :--- |
| **Module 1** | Data Collection via API, Pandas, NumPy, Measurement Scales | Collected 2 years of daily data (731 days, 20,468 rows) using Yahoo Finance API into a Pandas DataFrame. Price & Volume are **Ratio scale**, Symbol/Category are **Nominal scale**. |
| **Module 2** | Batch ETL Pipeline & Data Quality Checks | Built an automated script to Extract from API, Transform returns/volatility, and Load into CSV. Verified 0 missing values (`NaN = 0`). |
| **Module 3** | Supervised vs. Unsupervised Learning | Chose **Unsupervised Learning (Clustering)** because cryptocurrencies do not have predefined risk labels. |
| **Module 4** | Feature Engineering, StandardScaler, PCA Basics, K-Means, Silhouette Score | • Engineered 5 simple features (Volatility, Drawdown, Daily Spread, Log Volume, Vol/Volume ratio).<br>• Scaled features with `StandardScaler` ($Z = \frac{X - \mu}{\sigma}$).<br>• Used **2D PCA** ($96.94\%$ variance).<br>• Tuned **K-Means** with optimal $k=4$ confirmed by **Elbow Method** and **Silhouette Score (0.3868)**. |
| **Module 5** | Finance Application & Model Deployment | Built an interactive **Streamlit web dashboard (`app.py`)** with custom coin classification and portfolio allocation. |

---

## 🔬 Discovered Risk Categories (The 4 Tiers)

K-Means clustered the 28 cryptocurrencies into 4 distinct groups:

| Risk Category | Cryptocurrencies | Count | Mean Volatility (%) | Mean Max Drawdown (%) | Mean Daily Volume (USD) | What It Means |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: Ultra-Low Risk** | USDT, USDC, DAI | 3 | **0.43%** | **0.26%** | **$37.8 Billion** | **Stablecoins:** Pegged to the US Dollar, virtually no price swings. |
| **Tier 2: Low-to-Moderate Risk** | BTC, ETH, BNB, SOL, XRP | 5 | **64.80%** | **65.50%** | **$16.5 Billion** | **Blue-Chips:** Top market coins with massive daily volume that buffers drops. |
| **Tier 3: Moderate-to-High Risk** | ADA, AVAX, DOT, LINK, DOGE, LTC, BCH, ATOM, ETC, AAVE, MKR, CRV | 12 | **86.80%** | **83.20%** | **$785.4 Million** | **Utility & Platform Coins:** Active ecosystems with higher price fluctuations. |
| **Tier 4: Extreme Risk** | BONK, WIF, FLOKI, SHIB, NEAR, XLM, ALGO, LDO | 8 | **114.20%** | **87.70%** | **$248.1 Million** | **Meme & Speculative Coins:** Wild price swings (>114% volatility) and heavy drawdowns. |

---

## 📊 Key Visualizations

- **Volatility vs. Volume:** Answers the problem statement directly by plotting price change against trading volume.
- **Elbow Method & Silhouette Curve:** Shows why $k=4$ was selected (WCSS elbow and peak silhouette score of 0.3868).
- **2D PCA Scatter Plot:** Shows the 4 clusters separated across 2 principal components (96.94% variance explained).
- **Cluster Bar Charts:** Compares mean volatility, drawdown, and volume across the 4 tiers.

*(Generated charts are saved in `docs/figures/`)*

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Scripts
```bash
# Extract raw data from API (Module 1 & 2)
python src/data_collection.py

# Transform data into features (Module 2 & 4)
python src/feature_engineering.py

# Train K-Means and evaluate with Silhouette score (Module 4)
python src/clustering_models.py

# Generate visualization plots (Module 4 & 5)
python src/visualization.py
```

### 3. Open the Streamlit App (Module 5)
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
├── app.py                                   # Streamlit interactive dashboard (Module 5)
├── requirements.txt                         # Project libraries
├── README.md                                # Project overview
├── data/
│   ├── raw/crypto_historical_raw.csv        # 2-year daily market data (Module 1 & 2)
│   └── processed/
│       ├── crypto_risk_features.csv         # 5 engineered features (Module 4)
│       └── crypto_clustered_results.csv     # Final K-Means cluster assignments
├── docs/
│   ├── Project_Proposal.md                  # Project proposal document
│   ├── Final_Project_Report.md              # 10-page academic capstone report
│   ├── Final_Project_Report.docx            # Formatted Word report
│   ├── Viva_Voce_Preparation_Guide.md       # 10 viva defense Q&As for Modules 1-5
│   └── figures/                             # Visual charts (PNG)
├── notebooks/
│   ├── 01_data_collection_and_cleaning.ipynb
│   ├── 02_etl_and_feature_engineering.ipynb
│   ├── 03_model_implementation_and_clustering.ipynb
│   ├── 04_evaluation_and_risk_profiling.ipynb
│   └── Cryptocurrency_Risk_Analysis_Capstone.ipynb
├── presentation/
│   ├── crypto_risk_clustering_presentation.pptx    # 8-slide presentation deck
│   └── SLIDES.md                                   # Presentation notes
├── src/
│   ├── data_collection.py                   # API extraction script
│   ├── feature_engineering.py               # Feature calculations
│   ├── clustering_models.py                 # K-Means clustering & PCA
│   ├── visualization.py                     # Plot generation
│   ├── generate_notebooks.py                # Notebook generator
│   └── generate_docx_report.py              # Word report builder
└── tests/
    └── test_pipeline.py                     # Pipeline verification tests
```
