# CS2235: Data Science Capstone Project - Viva Voce Defense Guide

**RV University | School of Computer Science & Engineering**  
**Course:** Data Science Capstone Project (CS2235)  
**Topic:** Fintech: Cryptocurrency Risk Profiling — Grouping Cryptocurrencies into Similar Risk Categories Based on Price Volatility Dynamics and Trading Volume Depth  
**Student Researcher:** Thanisha S Rao  
**Weightage in Final Evaluation:** 10 Marks out of 25 (40% of Total Grade!)  
**Syllabus Focus:** Modules 1 to 5  

---

## 1. Questions on Modules 1 & 2 (Data Science Lifecycle, Data Types & ETL)

### Q1: What is the core problem statement of your capstone project?
**Answer:**
> "The problem statement is: *Grouping cryptocurrencies into similar risk categories based on how much their prices change and how much they are traded.*
> 
> Unlike traditional stock markets which have formal credit ratings and sector standards, cryptocurrencies lack clear risk ratings. By applying unsupervised learning on historical price volatility and trading volume, we group digital assets into data-driven risk categories without human bias."

---

### Q2: What measurement scales are present in your dataset (Module 1)?
**Answer:**
> "In accordance with Module 1 (Types of Data by Measurement Scale):
> - **Ratio Scale:** Open Price, High Price, Low Price, Close Price, and Trading Volume. These are continuous numerical features with a true zero point (e.g., zero volume means no trading occurred).
> - **Nominal Scale:** Cryptocurrency Symbol (e.g., `BTC`, `ETH`), Asset Name, and Category (e.g., Stablecoin, Layer-1). These are categorical labels with no intrinsic ordering."

---

### Q3: How did you implement your ETL Pipeline and perform Data Quality Checks (Module 2)?
**Answer:**
> "In Module 2, we built an automated batch ETL pipeline:
> 1. **Extract:** Downloaded 2 years of daily OHLCV trading data for 28 cryptocurrencies using the Yahoo Finance API (`yfinance`).
> 2. **Transform:** Standardized timestamps to daily calendar dates, calculated daily percentage returns and volatility, and computed logarithmic volume.
> 3. **Data Quality Checks:** Verified zero missing values (`NaN = 0`), verified physical price consistency ($\text{High}_t \ge \text{Close}_t \ge \text{Low}_t$), and confirmed no negative prices.
> 4. **Load:** Exported the cleaned, consolidated dataset to CSV format."

---

## 2. Questions on Module 3 & 4 (Machine Learning, Feature Engineering, PCA & Clustering)

### Q4: Why did you choose Unsupervised Learning instead of Supervised Learning (Module 3)?
**Answer:**
> "In supervised learning (Module 3), a model requires labeled target outputs (such as 'Low Risk' or 'High Risk'). However, in digital assets, there are no predefined, ground-truth risk labels. Any manual labelling would be subjective.
> 
> Therefore, we used **Unsupervised Learning (Clustering)** to let the geometry of the price and volume data reveal the natural groupings directly from market data."

---

### Q5: What features did you engineer, and why (Module 4)?
**Answer:**
> "We engineered 5 features addressing the two parts of the problem statement:
> - **Price Change Features:**
>   1. **Annualized Volatility (%):** $\sigma_{ann} = \text{std}(\text{daily returns}) \times \sqrt{365} \times 100$. This measures how much the price fluctuates over a full year.
>   2. **Maximum Drawdown (%):** The largest drop from a previous peak to the lowest trough.
>   3. **Mean Daily Spread (%):** Average $(\text{High} - \text{Low}) / \text{Close} \times 100$ measuring intraday fluctuations.
> - **Trading Volume Features:**
>   4. **$\log_{10}$ Average Daily Volume (USD):** Daily volume ranges from millions to billions of dollars, so applying $\log_{10}$ brings them onto a linear scale.
>   5. **Volatility-to-Volume Ratio:** An intuitive ratio comparing price fluctuation to daily volume depth."

---

### Q6: Why did you use `StandardScaler` before clustering (Module 4)?
**Answer:**
> "Features such as daily trading volume (millions or billions of dollars) and percentage volatility (values between 0.5% and 150%) operate on vastly different scales. In distance-based algorithms like K-Means, unscaled volume would completely dominate Euclidean distance calculations.
> 
> We applied **StandardScaler** ($Z = \frac{X - \mu}{\sigma}$), which normalizes each feature to have a mean of 0 and a standard deviation of 1, ensuring all features contribute equally."

---

### Q7: Explain K-Means Clustering and how you determined the optimal number of clusters $k$ (Module 4).
**Answer:**
> "K-Means is an unsupervised clustering algorithm that groups $n$ data points into $k$ clusters by minimizing the within-cluster sum of squares (WCSS / Inertia):
> $$\text{WCSS} = \sum_{k=1}^K \sum_{x \in C_k} \| x - \mu_k \|^2$$
> 
> We evaluated $k$ from 2 to 7 using two standard metrics from Module 4:
> 1. **The Elbow Method (Inertia):** The curve shows a distinct bend or 'elbow' at $k = 4$, after which adding more clusters yields diminishing returns.
> 2. **Silhouette Score:** The score measures how well-separated the clusters are. The Silhouette score peaked at **$0.3868$** at $k = 4$.
> 
> Both metrics confirmed that $k = 4$ is the optimal number of risk categories."

---

### Q8: What did PCA (Dimensionality Reduction) reveal about your data (Module 4)?
**Answer:**
> "PCA (Principal Component Analysis) projected our 5 features into 2 Principal Components:
> - **PC1 ($79.84\%$ variance):** Loads heavily on Price Volatility and Max Drawdown (the Price Change axis).
> - **PC2 ($17.10\%$ variance):** Loads heavily on Average Daily Trading Volume (the Trading Activity axis).
> 
> Together, the 2 components capture **$96.94\%$** of the total variance, confirming that price change and volume explain almost all the variance in the data."

---

## 3. Questions on Module 5 (Deployment & Financial Application)

### Q9: What are the four discovered risk categories?
**Answer:**
> "The four categories discovered by K-Means are:
> 1. **Tier 1: Ultra-Low Risk (Stablecoins - 3 coins):** USDT, USDC, DAI. Minimal price change (mean volatility 0.43%, max drawdown 0.26%) with massive daily volume ($>\$37\text{B}$).
> 2. **Tier 2: Low-to-Moderate Risk (Blue-Chips - 5 coins):** BTC, ETH, BNB, SOL, XRP. Established assets with deep multi-billion-dollar daily liquidity that buffers against price drops (mean volatility 64.8%).
> 3. **Tier 3: Moderate-to-High Risk (Utility & Ecosystems - 12 coins):** ADA, AVAX, DOT, LINK, DOGE, etc. Higher growth and higher price swings (mean volatility 86.8%).
> 4. **Tier 4: Extreme Speculative Risk (Meme Tokens - 8 coins):** BONK, WIF, FLOKI, SHIB, etc. Extreme price fluctuations (mean volatility $>114\%$) and severe drawdowns ($>87\%$).

---

### Q10: How does your Streamlit deployment work (Module 5)?
**Answer:**
> "In Module 5, we deployed an interactive web dashboard using **Streamlit (`app.py`)**:
> 1. **Volatility vs. Volume Scatter Plot:** Graphically answers the problem statement.
> 2. **Asset Risk Matrix:** Allows filtering coins by risk category.
> 3. **Live Risk Classifier:** An interactive tool where an investor inputs expected price volatility and daily volume for any cryptocurrency, and our trained K-Means pipeline predicts which risk tier it belongs to.
> 4. **Portfolio Allocator:** Suggests portfolio weights across the 4 risk categories based on user risk appetite (Conservative, Balanced, Aggressive)."
