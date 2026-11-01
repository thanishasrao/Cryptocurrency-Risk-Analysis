"""
Generates a professional 8-slide PowerPoint presentation (.pptx)
for the CS2235 Capstone Project Presentation & Viva.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

def create_presentation(output_path="presentation/crypto_risk_clustering_presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    # Color palette
    NAVY = RGBColor(26, 54, 93)       # #1A365D
    SLATE = RGBColor(74, 85, 104)     # #4A5568
    BLUE = RGBColor(49, 130, 206)     # #3182CE
    GREEN = RGBColor(47, 133, 90)     # #2F855A
    WHITE = RGBColor(255, 255, 255)
    LIGHT_BG = RGBColor(247, 250, 252) # #F7FAFC
    
    blank_layout = prs.slide_layouts[6]
    
    # Helper to add standard header
    def add_header(slide, title_text, category_text="CS2235 DATA SCIENCE CAPSTONE PROJECT"):
        # Header background banner
        top_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.2))
        tf = top_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = BLUE
        
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(26)
        p1.font.bold = True
        p1.font.color.rgb = NAVY
        
    # Helper to add card container
    def add_card(slide, left, top, width, height, title, body_bullets, title_color=NAVY):
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.15)
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = title_color
        
        for bullet in body_bullets:
            p_b = tf.add_paragraph()
            p_b.text = f"•  {bullet}"
            p_b.font.size = Pt(13)
            p_b.font.color.rgb = SLATE
            p_b.space_before = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    
    # Title box
    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.8))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "Fintech: Cryptocurrency Risk Profiling"
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = NAVY
    
    p_sub = tf1.add_paragraph()
    p_sub.text = "Grouping Digital Assets into Empirical Risk Tiers via Unsupervised Price Volatility & Trading Volume Clustering"
    p_sub.font.size = Pt(19)
    p_sub.font.color.rgb = BLUE
    p_sub.space_before = Pt(10)
    
    p_div = tf1.add_paragraph()
    p_div.text = "____________________________________________________________________________________"
    p_div.font.size = Pt(12)
    p_div.font.color.rgb = RGBColor(226, 232, 240)
    
    p_meta = tf1.add_paragraph()
    p_meta.text = "RV University | School of Computer Science & Engineering | B.Sc (Hons.) Data Science"
    p_meta.font.size = Pt(13)
    p_meta.font.bold = True
    p_meta.font.color.rgb = SLATE
    p_meta.space_before = Pt(15)
    
    p_team = tf1.add_paragraph()
    p_team.text = "Researcher: Thanisha S Rao  |  Course Code: CS2235  |  Instructors: Prof. Muralidhar Billa, Prof. Thirukumaran Subbiramani"
    p_team.font.size = Pt(12)
    p_team.font.color.rgb = SLATE

    # -------------------------------------------------------------
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement & Financial Motivation")
    
    add_card(s2, 0.8, 1.8, 5.6, 5.0, "The Fintech Dilemma", [
        "Cryptocurrency markets lack traditional institutional rating agencies (like Moody's or S&P).",
        "Tokens exhibit extreme cross-sectional heterogeneity in volatility and liquidity depth.",
        "Retail & institutional investors face severe tail risk and hidden illiquidity traps.",
        "Standard market-cap rankings obscure asymmetric drawdown severity."
    ], title_color=NAVY)
    
    add_card(s2, 6.8, 1.8, 5.7, 5.0, "Our Machine Learning Objective", [
        "Formulation: Unsupervised Machine Learning (Clustering).",
        "Target: Group assets into mathematically validated risk categories based on:",
        "1. How much their prices change (Annualized Volatility, Max Drawdown, VaR 95%, CVaR).",
        "2. How much they are traded (Log Dollar Volume, Volume Stability, Amihud Illiquidity).",
        "Outcome: Objective, algorithmic risk taxonomy without subjective bias."
    ], title_color=BLUE)

    # -------------------------------------------------------------
    # SLIDE 3: DATASET & MULTI-ASSET TAXONOMY
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Dataset Coverage & Data Integrity Pipeline")
    
    add_card(s3, 0.8, 1.8, 5.6, 5.0, "Dataset Architecture", [
        "Source: Yahoo Finance Financial API & CoinGecko benchmarks.",
        "Horizon: 2 Full Years of continuous daily trading data (731 days).",
        "Coverage: 28 selected cryptocurrencies (20,468 total daily records).",
        "Zero Missing Values: Strictly audited for timestamp alignment and contiguous prices."
    ])
    
    add_card(s3, 6.8, 1.8, 5.7, 5.0, "Diversified Asset Spectrum", [
        "Stablecoins (Baseline): USDT, USDC, DAI (<0.7% Volatility).",
        "Mega-Cap Blue-Chips: BTC, ETH, BNB (Institutional Depth).",
        "Layer 1 / Layer 2 Platforms: SOL, XRP, ADA, AVAX, DOT, NEAR, ATOM, ALGO.",
        "DeFi & Infrastructure: LINK, AAVE, MKR, LDO, CRV.",
        "High-Beta & Meme Coins: DOGE, SHIB, BONK, FLOKI, WIF (>100% Volatility)."
    ], title_color=GREEN)

    # -------------------------------------------------------------
    # SLIDE 4: MATHEMATICAL FEATURE ENGINEERING
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Quantitative Feature Engineering Framework")
    
    add_card(s4, 0.8, 1.8, 5.6, 5.0, "Price Change & Volatility Metrics", [
        "Annualized Volatility: σ_ann = σ_daily × √365",
        "Maximum Drawdown (MDD): Peak-to-trough worst collapse percentage.",
        "Value at Risk (95% VaR): 5th percentile of daily loss distribution.",
        "Conditional VaR (CVaR): Expected Shortfall beyond the 95% threshold.",
        "Sortino Ratio & Intraday High-Low Spread %."
    ])
    
    add_card(s4, 6.8, 1.8, 5.7, 5.0, "Trading Volume & Liquidity Metrics", [
        "Log10 Average Daily Volume (USD): Quantifies depth across orders of magnitude.",
        "Volume Volatility (CV): Coefficient of variation of trading volume.",
        "Amihud Illiquidity Ratio Proxy: Price impact per unit of trading volume.",
        "Volatility-to-Volume Ratio: Speculation index measuring volume absorption capacity."
    ], title_color=BLUE)

    # -------------------------------------------------------------
    # SLIDE 5: CLUSTERING METHODOLOGY & OPTIMAL K
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Clustering Algorithms & Model Optimization")
    
    add_card(s5, 0.8, 1.8, 5.6, 5.0, "Algorithms Evaluated", [
        "K-Means++: Distance-based partitioning with optimized seeding.",
        "Agglomerative Hierarchical: Ward's minimum variance Euclidean linkage.",
        "Gaussian Mixture Models (GMM): Soft probabilistic clustering.",
        "DBSCAN: Density-based spatial noise and outlier detection."
    ])
    
    add_card(s5, 6.8, 1.8, 5.7, 5.0, "Optimal K Validation Metrics", [
        "Elbow Method: Pronounced inflection point at k = 4.",
        "Silhouette Score: Peak separation quality at 0.3464.",
        "Davies-Bouldin Index: 0.8382 (low within-cluster scatter vs. separation).",
        "Calinski-Harabasz Index: 33.78 confirming dense, distinct clusters."
    ], title_color=GREEN)

    # -------------------------------------------------------------
    # SLIDE 6: EMPIRICAL FINDINGS & THE 4 RISK TIERS
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Discovered Cryptocurrency Risk Taxonomy")
    
    add_card(s6, 0.8, 1.8, 5.6, 2.4, "Tier 1: Ultra-Low Risk (Pegged)", [
        "Assets: USDT, USDC, DAI (3 assets)",
        "Mean Volatility: 0.43% | Max Drawdown: 0.26%",
        "Role: Risk-off collateral, dollar pegs, cash management."
    ], title_color=GREEN)
    
    add_card(s6, 6.8, 1.8, 5.7, 2.4, "Tier 2: Low-to-Moderate (Blue-Chips)", [
        "Assets: BTC, ETH, BNB (3 assets)",
        "Mean Volatility: 54.4% | Max Drawdown: 59.7%",
        "Role: Core institutional anchors with >$2B daily liquidity."
    ], title_color=BLUE)
    
    add_card(s6, 0.8, 4.4, 5.6, 2.4, "Tier 3: Moderate-to-High (Utility/DeFi)", [
        "Assets: SOL, XRP, ADA, AVAX, DOT, LINK, etc. (12 assets)",
        "Mean Volatility: 82.2% | Max Drawdown: 79.5%",
        "Role: Ecosystem growth, network utility, smart contracts."
    ], title_color=RGBColor(217, 119, 6))
    
    add_card(s6, 6.8, 4.4, 5.7, 2.4, "Tier 4: Extreme Speculative (Meme/Beta)", [
        "Assets: BONK, WIF, FLOKI, SHIB, LDO, CRV (10 assets)",
        "Mean Volatility: 112.5% | Max Drawdown: 86.8%",
        "Role: Asymmetric speculation; requires strict stop-loss caps."
    ], title_color=RGBColor(220, 38, 38))

    # -------------------------------------------------------------
    # SLIDE 7: DIMENSIONALITY REDUCTION & VISUALIZATION
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "PCA Latent Space & Volatility-Volume Frontier")
    
    add_card(s7, 0.8, 1.8, 5.6, 5.0, "Principal Component Analysis (PCA)", [
        "PCA Component 1 explains 72.8% of variance (Volatility & Tail Risk loading).",
        "PCA Component 2 explains 21.0% of variance (Liquidity & Turnover loading).",
        "Cumulative 2D Explained Variance: 93.78% (3D: 97.49%).",
        "Zero-overlap boundary separation across all 4 discovered clusters."
    ], title_color=NAVY)
    
    add_card(s7, 6.8, 1.8, 5.7, 5.0, "Core Problem Statement Findings", [
        "High Trading Volume does not always eliminate volatility (e.g. DOGE, SHIB).",
        "Tokens with identical returns have drastically divergent tail drawdowns.",
        "Stablecoins cluster into an isolated, hyper-dense centroid.",
        "Provides clear empirical evidence for dynamic risk-based capital allocation."
    ], title_color=BLUE)

    # -------------------------------------------------------------
    # SLIDE 8: DEPLOYMENT DEMO & CONCLUSION
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Interactive Deployment & Strategic Takeaways")
    
    add_card(s8, 0.8, 1.8, 5.6, 5.0, "Interactive Streamlit Deployment", [
        "Web Dashboard (`app.py`) for real-time risk inspection.",
        "Interactive 2D Volatility vs. Volume liquidity frontier.",
        "Live Risk Classifier: Predicts risk tier for custom/new tokens.",
        "Dynamic Portfolio Allocator: Custom weights by investor risk profile."
    ], title_color=GREEN)
    
    add_card(s8, 6.8, 1.8, 5.7, 5.0, "Capstone Deliverables Summary", [
        "Project Proposal: Formal problem framing & team roles.",
        "Jupyter Notebooks: Complete 4-phase reproducible pipelines.",
        "Academic Technical Report: 10-page comprehensive thesis.",
        "Presentation & Viva Voce Guide: Confident technical defense.",
        "GitHub Repository: Structured, version-controlled repository."
    ], title_color=NAVY)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"[+] Presentation slide deck successfully saved to: {output_path}")

if __name__ == "__main__":
    create_presentation()
