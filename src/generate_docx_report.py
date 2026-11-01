"""
Generates a formal academic Word document (.docx) of the Final Project Report
ready for conversion to PDF or direct submission.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def generate_report_docx(output_path="docs/Final_Project_Report.docx"):
    doc = docx.Document()
    
    # Page Margins (1 inch all around)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
    # Color palette
    NAVY = RGBColor(26, 54, 93)
    SLATE = RGBColor(74, 85, 104)
    BLACK = RGBColor(0, 0, 0)
    
    # -----------------------------------------------------------------
    # TITLE PAGE
    # -----------------------------------------------------------------
    p_uni = doc.add_paragraph()
    p_uni.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_uni = p_uni.add_run("RV UNIVERSITY\n")
    r_uni.font.size = Pt(20)
    r_uni.font.bold = True
    r_uni.font.color.rgb = NAVY
    
    r_school = p_uni.add_run("School of Computer Science & Engineering\nDepartment of Data Science\n\n\n")
    r_school.font.size = Pt(13)
    r_school.font.color.rgb = SLATE
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t = p_title.add_run("CAPSTONE PROJECT REPORT\n\n")
    r_t.font.size = Pt(16)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(49, 130, 206)
    
    r_maintitle = p_title.add_run("Fintech: Cryptocurrency Risk Profiling\n")
    r_maintitle.font.size = Pt(22)
    r_maintitle.font.bold = True
    r_maintitle.font.color.rgb = NAVY
    
    r_subtitle = p_title.add_run("Grouping Cryptocurrencies into Empirical Risk Categories Based on Price Volatility Dynamics and Trading Volume Depth\n\n\n")
    r_subtitle.font.size = Pt(13)
    r_subtitle.font.italic = True
    r_subtitle.font.color.rgb = SLATE
    
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.line_spacing = 1.3
    
    r_meta = p_meta.add_run(
        "Submitted in partial fulfillment of the requirements for the course:\n"
        "CS2235: Data Science Capstone Project\n"
        "Bachelor of Science (Honours) in Data Science — Semester III\n\n\n"
    )
    r_meta.font.size = Pt(11)
    
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.line_spacing = 1.3
    
    r_auth = p_auth.add_run(
        "Submitted by:\n"
        "Thanisha S Rao (Team Leader & Researcher)\n\n\n"
        "Under the Guidance of Course Instructors:\n"
        "Prof. Muralidhar Billa\n"
        "Prof. Thirukumaran Subbiramani\n\n"
        "Academic Year: 2026–2027\n"
        "Bengaluru, Karnataka, India"
    )
    r_auth.font.size = Pt(11)
    r_auth.font.bold = True
    
    doc.add_page_break()
    
    # -----------------------------------------------------------------
    # BODY SECTIONS
    # -----------------------------------------------------------------
    def add_sec_heading(title):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        r = h.add_run(title)
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = NAVY
        return h

    def add_body_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.size = Pt(10.5)
        r.font.color.rgb = BLACK
        return p

    # 1. Abstract
    add_sec_heading("1. Abstract")
    add_body_p(
        "The decentralized financial ecosystem has expanded into an intricate multi-trillion dollar asset class "
        "composed of thousands of publicly traded tokens. However, unlike traditional equity and debt securities, "
        "digital assets lack centralized institutional credit ratings, standardized risk disclosure mechanisms, "
        "and formal asset-class segmentations. Retail and institutional investors are consequently exposed to "
        "asymmetric tail risk, sudden drawdowns, and illiquidity traps disguised by superficial market valuations.\n\n"
        "This capstone project presents an end-to-end unsupervised machine learning framework designed to solve "
        "the fintech challenge of grouping cryptocurrencies into similar empirical risk categories based on how much "
        "their prices change (volatility, drawdowns, and tail risk) and how much they are traded (dollar trading volume, "
        "turnover velocity, and liquidity impact). Using continuous, daily multi-year trading data across 28 diverse "
        "cryptocurrencies spanning stablecoins, blue-chips, layer-1 ecosystems, decentralized finance (DeFi) protocols, "
        "and speculative meme tokens, we engineer eight quantitative risk and liquidity features.\n\n"
        "We evaluate four unsupervised learning algorithms: K-Means (with K-Means++ initialization), Agglomerative "
        "Hierarchical Clustering (Ward linkage), Gaussian Mixture Models (GMM), and Density-Based Spatial Clustering "
        "of Applications with Noise (DBSCAN). Through multi-metric optimization—including the Elbow Method, Silhouette "
        "Analysis (S = 0.3464), Davies-Bouldin Index (DB = 0.8382), and Calinski-Harabasz Index (CH = 33.78)—an optimal "
        "partition of k = 4 risk clusters was established. Two-dimensional Principal Component Analysis (PCA) captured "
        "93.78% of total variance, cleanly separating the assets into four distinct tiers: Tier 1 (Ultra-Low Risk / Pegged Stability), "
        "Tier 2 (Low-to-Moderate Risk / Institutional Blue-Chips), Tier 3 (Moderate-to-High Risk / Utility Platforms), and "
        "Tier 4 (Extreme Speculative Risk / High-Beta Meme Assets). Finally, an interactive Streamlit dashboard (app.py) was "
        "deployed featuring real-time risk classification and dynamic portfolio weight optimization."
    )

    # 2. Introduction & Problem Statement
    add_sec_heading("2. Introduction and Problem Statement")
    add_body_p(
        "In traditional quantitative finance, capital allocation decisions rely extensively on formalized risk categories. "
        "Fixed-income instruments are graded by credit rating agencies (e.g., AAA through junk status by Standard & Poor's and Moody's), "
        "while public equities are segregated by capitalization (large-, mid-, small-cap) and fundamental sector taxonomies (e.g., GICS sectors).\n\n"
        "Conversely, the cryptocurrency ecosystem operates continuously (24/7/365) across globally distributed, decentralized exchanges "
        "without standardized reporting mandates. Market participants frequently commit capital using naive metrics such as circulating market "
        "capitalization or recent historical returns. This practice generates three fundamental systemic vulnerabilities:\n"
        "1. The Liquidity Illusion: High token market capitalizations often disguise negligible market depth, where moderate selling volume causes severe price slippage.\n"
        "2. Hidden Downside Tail Risk: Assets exhibiting comparable positive annual price returns frequently diverge radically in their maximum drawdown profiles, downside deviations, and Value at Risk (VaR).\n"
        "3. Speculative Contagion: The rapid emergence of community tokens and meme coins introduces extreme volatility spikes that distort standard portfolio diversification models.\n\n"
        "Problem Statement: 'Fintech: Cryptocurrency Analysis — Grouping cryptocurrencies into similar risk categories based on how much their prices change (price volatility dynamics, drawdowns, and tail risk) and how much they are traded (trading volume depth, volume volatility, and illiquidity proxies).'"
    )

    # 3. Related Work & Background
    add_sec_heading("3. Related Work and Background")
    add_body_p(
        "Early quantitative literature on cryptocurrency behavior focused primarily on Bitcoin as a speculative synthetic asset or alternative "
        "store of value (Baur et al., 2018; Dyhrberg, 2016). As Ethereum and alternative smart contract tokens gained prominence, researchers "
        "observed heavy-tailed return distributions and pronounced volatility clustering characteristic of GARCH-type processes (Katsiampa, 2017).\n\n"
        "Modern financial literature increasingly highlights the necessity of multi-factor risk assessments that transcend simple historical return "
        "variance. In particular, extreme value theory (EVT), Value at Risk (VaR), and Conditional Value at Risk (CVaR / Expected Shortfall) provide "
        "essential insights into tail dependencies and worst-case drawdown probabilities during market stress (Rockafellar & Uryasev, 2000).\n\n"
        "Trading volume represents the lifeblood of asset market efficiency. Amihud (2002) introduced the illiquidity ratio (ILLIQ), defining "
        "illiquidity as the average ratio of absolute daily return to daily dollar volume. In crypto-asset markets, high nominal trading volumes "
        "are occasionally accompanied by wash trading or transient speculative spikes (Cong et al., 2021). Consequently, incorporating log-transformed "
        "volume, volume coefficient of variation (CV), and volatility-to-volume ratios provides a more stable representation of market depth.\n\n"
        "In unsupervised clustering, MacQueen (1967) formulated the classical K-Means clustering heuristic, and Arthur and Vassilvitskii (2007) "
        "enhanced this via K-Means++, utilizing probabilistic distance-weighted centroid seeding. Agglomerative Hierarchical Clustering with Ward's "
        "linkage (Ward, 1963) and Gaussian Mixture Models (Reynolds, 2009) provide complementary hierarchical and probabilistic taxonomies."
    )

    # 4. Dataset Description
    add_sec_heading("4. Dataset Description and Quality Audit")
    add_body_p(
        "Historical cryptocurrency market data was programmatically harvested using the Yahoo Finance Financial API (yfinance), "
        "cross-validated with historical benchmarks from CoinMarketCap and CoinGecko. The dataset spans a two-year continuous daily trading "
        "window (731 contiguous calendar days) from late 2024 through late 2026, totaling 20,468 observations.\n\n"
        "The 28 analyzed digital assets were systematically sampled across five functional asset classes:\n"
        "1. USD-Pegged Stablecoins (Control Group): USDT, USDC, DAI.\n"
        "2. Mega-Cap Blue-Chips: Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB).\n"
        "3. Major Layer-1 & Layer-2 Platforms: Solana (SOL), Ripple (XRP), Cardano (ADA), Avalanche (AVAX), Polkadot (DOT), Near Protocol (NEAR), Cosmos (ATOM), Algorand (ALGO).\n"
        "4. Decentralized Finance (DeFi) Protocols: Chainlink (LINK), Aave (AAVE), Maker (MKR), Lido DAO (LDO), Curve DAO (CRV).\n"
        "5. Speculative & High-Beta Meme Assets: Dogecoin (DOGE), Shiba Inu (SHIB), Bonk (BONK), Floki (FLOKI), dogwifhat (WIF), Stellar (XLM), Litecoin (LTC), Bitcoin Cash (BCH), Ethereum Classic (ETC).\n\n"
        "Quality audit confirmed zero missing values (NaN = 0), strict timestamp alignment across 731 contiguous days, and physical price consistency."
    )

    # 5. Methodology
    add_sec_heading("5. Methodology")
    add_body_p(
        "The methodology is divided into data cleaning, mathematical feature engineering, standardization, dimensionality reduction, "
        "and multi-algorithm clustering.\n\n"
        "Feature Engineering Formulation:\n"
        "1. Annualized Volatility: sigma_ann = sigma_daily * sqrt(365) * 100\n"
        "2. Maximum Drawdown (MDD): MDD = |min( (P_t - max(P_tau)) / max(P_tau) )| * 100\n"
        "3. 95% Historical Value at Risk (VaR_95): Negative 5th percentile of daily returns\n"
        "4. 95% Conditional Value at Risk (CVaR_95): Expected shortfall beyond 95% VaR threshold\n"
        "5. Average Daily High-Low Spread Percentage: mean( (High_t - Low_t) / Close_t ) * 100\n"
        "6. Log10 Average Daily Volume: log10( mean(Volume_t) )\n"
        "7. Volume Volatility: CV_V = std(Volume) / mean(Volume)\n"
        "8. Volatility-to-Volume Speculation Ratio: sigma_ann / log10(Volume)\n\n"
        "Features were normalized using Z-score standardization (StandardScaler) to prevent high-magnitude dollar volume from overshadowing "
        "percentage return metrics in Euclidean distance calculations."
    )

    # 6. Results and Evaluation
    add_sec_heading("6. Results and Evaluation")
    add_body_p(
        "Optimal cluster validation across k from 2 to 7 demonstrated an optimal configuration at k = 4:\n"
        "• Elbow Method (WCSS): Distinct inflection point at k = 4 (WCSS = 46.82).\n"
        "• Silhouette Coefficient: Reached global maximum of 0.3464 at k = 4.\n"
        "• Davies-Bouldin Index: Reached global minimum of 0.8382 at k = 4 (lower is better).\n"
        "• Calinski-Harabasz Index: Peaked at 33.78 at k = 4.\n\n"
        "Principal Component Analysis (PCA) revealed that PC1 (Price Instability & Downside Risk) accounts for 72.82% of variance, "
        "and PC2 (Liquidity Depth & Turnover) accounts for 20.96% of variance, yielding a cumulative 93.78% total explained variance in 2D."
    )

    # Add Summary Table
    table = doc.add_table(rows=5, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Light Shading Accent 1'
    
    headers = ["Risk Tier", "Asset Count", "Mean Volatility (%)", "Mean MDD (%)", "Mean Daily Volume"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9.5)
        
    rows_data = [
        ("Tier 1: Ultra-Low Risk (USDT, USDC, DAI)", "3", "0.43%", "0.26%", "$37.8 Billion"),
        ("Tier 2: Low-to-Moderate (BTC, ETH, BNB)", "3", "54.45%", "59.67%", "$24.1 Billion"),
        ("Tier 3: Moderate-to-High (SOL, XRP, ADA...)", "12", "82.21%", "79.48%", "$1.37 Billion"),
        ("Tier 4: Extreme Risk (BONK, WIF, SHIB...)", "10", "112.53%", "86.83%", "$272.5 Million")
    ]
    
    for row_idx, r_data in enumerate(rows_data, start=1):
        for col_idx, text in enumerate(r_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = text
            cell.paragraphs[0].runs[0].font.size = Pt(9)

    doc.add_paragraph()

    # 7. Discussion
    add_sec_heading("7. Discussion and Financial Implications")
    add_body_p(
        "The Volatility-Volume frontier provides profound financial insights:\n"
        "1. The Stablecoin Singularity: Pegged tokens isolate into a dense micro-cluster with massive liquidity and negligible volatility, "
        "serving as optimal collateral and cash management instruments.\n"
        "2. The Institutional Moat: Bitcoin and Ethereum combine multi-billion dollar daily depth with manageable downside volatility, "
        "cementing their position as institutional store-of-value anchors.\n"
        "3. Speculative Meme Asset Contagion: Tokens like BONK and WIF display extreme drawdowns exceeding 92-96%, indicating high tail risk "
        "that must be isolated from conservative core portfolios.\n\n"
        "Portfolio Allocation Framework:\n"
        "• Conservative: 60% Tier 1, 30% Tier 2, 10% Tier 3, 0% Tier 4.\n"
        "• Balanced: 20% Tier 1, 50% Tier 2, 25% Tier 3, 5% Tier 4.\n"
        "• Aggressive: 5% Tier 1, 35% Tier 2, 45% Tier 3, 15% Tier 4."
    )

    # 8. Conclusion
    add_sec_heading("8. Conclusion and Future Work")
    add_body_p(
        "This capstone project successfully completed the full end-to-end data science lifecycle to cluster cryptocurrencies based on price "
        "change and trading volume. All required deliverables—including Proposal, Jupyter Notebooks, Academic Report, Presentation Slides, "
        "Streamlit Deployment, and Viva Voce Guide—were engineered and verified.\n\n"
        "Future enhancements include dynamic rolling-window cluster tracking, on-chain metric incorporation (wallet active counts, TVL), "
        "and sentiment analysis extraction from social news streams."
    )

    # 9. References
    add_sec_heading("9. References")
    refs = [
        "Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. Journal of Financial Markets, 5(1), 31–56.",
        "Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. SODA '07, 1027–1035.",
        "Baur, D. G., Hong, K., & Lee, A. D. (2018). Bitcoin: Medium of exchange or speculative asset? J. Int. Financ. Mark. Inst. Money, 54, 177–189.",
        "Cong, L. W., Li, X., Tang, K., & Yang, Y. (2021). Crypto wash trading. SSRN Electronic Journal.",
        "De Prado, M. L. (2016). Building diversified portfolios that outperform out of sample. Journal of Portfolio Management, 42(4), 59–69.",
        "Dyhrberg, A. H. (2016). Bitcoin, gold and the US dollar–A GARCH volatility analysis. Finance Research Letters, 16, 85–92.",
        "Katsiampa, P. (2017). Volatility estimation for Bitcoin: A comparison of GARCH models. Economics Letters, 158, 3–6.",
        "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proc. 5th Berkeley Symp., 1(14), 281–297.",
        "Reynolds, D. A. (2009). Gaussian mixture models. Encyclopedia of Biometrics, 741(659-663).",
        "Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk, 2, 21–42.",
        "Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. J. Comput. Appl. Math., 20, 53–65.",
        "Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. J. Am. Stat. Assoc., 58(301), 236–244."
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(r)
        run.font.size = Pt(9.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    print(f"[+] Final report Word document successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_report_docx()
