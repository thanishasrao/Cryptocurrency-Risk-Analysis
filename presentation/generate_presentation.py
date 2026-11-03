"""
Generates an 8-slide PowerPoint presentation (.pptx)
for the CS2235 Capstone Project Presentation & Viva.
Aligned directly with Modules 1 to 5.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

def create_presentation(output_path="presentation/crypto_risk_clustering_presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    # Color palette
    NAVY = RGBColor(26, 54, 93)       # #1A365D
    SLATE = RGBColor(74, 85, 104)     # #4A5568
    BLUE = RGBColor(49, 130, 206)     # #3182CE
    GREEN = RGBColor(47, 133, 90)     # #2F855A
    
    blank_layout = prs.slide_layouts[6]
    
    def add_header(slide, title_text, category_text="CS2235 DATA SCIENCE CAPSTONE PROJECT"):
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
    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.8))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "Fintech: Cryptocurrency Risk Profiling"
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = NAVY
    
    p_sub = tf1.add_paragraph()
    p_sub.text = "Grouping Cryptocurrencies into Similar Risk Categories via K-Means Clustering"
    p_sub.font.size = Pt(19)
    p_sub.font.color.rgb = BLUE
    p_sub.space_before = Pt(10)
    
    p_meta = tf1.add_paragraph()
    p_meta.text = "RV University | School of Computer Science & Engineering | B.Sc (Hons.) Data Science"
    p_meta.font.size = Pt(13)
    p_meta.font.bold = True
    p_meta.font.color.rgb = SLATE
    p_meta.space_before = Pt(20)
    
    p_team = tf1.add_paragraph()
    p_team.text = "Researcher: Thanisha S Rao  |  Course: CS2235 Data Science  |  Instructors: Prof. Muralidhar Billa, Prof. Thirukumaran Subbiramani"
    p_team.font.size = Pt(12)
    p_team.font.color.rgb = SLATE

    # -------------------------------------------------------------
    # SLIDE 2: PROBLEM STATEMENT & SYLLABUS MAPPING
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement & Course Alignment")
    
    add_card(s2, 0.8, 1.8, 5.6, 5.0, "Problem Statement", [
        "Topic: Fintech: Cryptocurrency Analysis.",
        "Problem: Grouping cryptocurrencies into similar risk categories based on how much their prices change and how much they are traded.",
        "Unlike stock markets, crypto lacks standardized risk ratings.",
        "Investors need objective data-driven risk categories."
    ], title_color=NAVY)
    
    add_card(s2, 6.8, 1.8, 5.7, 5.0, "Alignment with CS2235 Syllabus", [
        "Module 1: Data collection via API (yfinance), Pandas & NumPy.",
        "Module 2: Building ETL pipeline & data quality checks.",
        "Module 3: Unsupervised Machine Learning (no target labels).",
        "Module 4: Feature Engineering, StandardScaler, PCA, K-Means Clustering & Silhouette Score evaluation.",
        "Module 5: Application in Finance & Streamlit deployment."
    ], title_color=BLUE)

    # -------------------------------------------------------------
    # SLIDE 3: DATA INGESTION & QUALITY CHECKS (MODULES 1 & 2)
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Data Ingestion & Quality Checks (Modules 1 & 2)")
    
    add_card(s3, 0.8, 1.8, 5.6, 5.0, "Dataset Architecture", [
        "Source: Yahoo Finance API (2 years of daily data).",
        "Scale: 28 cryptocurrencies across 731 days (20,468 rows).",
        "Measurement Scale (Module 1): Ratio scale for Price & Volume (continuous numerical with true zero).",
        "Nominal scale for Symbol and Asset Category."
    ])
    
    add_card(s3, 6.8, 1.8, 5.7, 5.0, "Data Pipeline & Quality Checks (Module 2)", [
        "Extract: Automated API data harvester in Python.",
        "Transform: Date parsing, timestamp alignment, 0 missing values.",
        "Quality Checks: Verified High >= Low, Close > 0, Volume >= 0.",
        "Load: Cleaned consolidated dataset saved to CSV."
    ], title_color=GREEN)

    # -------------------------------------------------------------
    # SLIDE 4: FEATURE ENGINEERING (MODULE 4)
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Feature Engineering (Module 4)")
    
    add_card(s4, 0.8, 1.8, 5.6, 5.0, "Price Change Features", [
        "Annualized Volatility (%): Daily return std dev × √365 × 100. Measures price fluctuations over a year.",
        "Maximum Drawdown (%): Largest peak-to-trough price drop.",
        "Mean Daily Spread (%): Average daily (High - Low) / Close."
    ])
    
    add_card(s4, 6.8, 1.8, 5.7, 5.0, "Trading Volume Features", [
        "Average Daily Volume (USD): Mean dollar volume traded daily.",
        "Log10 Average Volume: Brings billion-dollar volumes down to a linear scale (Module 4 scaling).",
        "Volatility-to-Volume Ratio: Price fluctuation divided by volume depth."
    ], title_color=BLUE)

    # -------------------------------------------------------------
    # SLIDE 5: K-MEANS CLUSTERING & MODEL EVALUATION (MODULE 4)
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "K-Means Clustering & Hyperparameter Tuning (Module 4)")
    
    add_card(s5, 0.8, 1.8, 5.6, 5.0, "Feature Scaling & K-Means", [
        "StandardScaler (Z-Score): Standardizes features so high volume doesn't dominate percentage volatility.",
        "Algorithm: K-Means++ (probabilistic centroid initialization).",
        "Hyperparameter Tuning: Tested k from 2 to 7."
    ])
    
    add_card(s5, 6.8, 1.8, 5.7, 5.0, "Evaluation: Elbow & Silhouette Score", [
        "Elbow Method (WCSS): Distinct elbow observed at k = 4.",
        "Silhouette Score: Peak score of 0.3868 achieved at k = 4.",
        "Conclusion: Exactly 4 natural risk categories exist in the data."
    ], title_color=GREEN)

    # -------------------------------------------------------------
    # SLIDE 6: EMPIRICAL RISK CATEGORIES (THE 4 TIERS)
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Discovered Cryptocurrency Risk Categories")
    
    add_card(s6, 0.8, 1.8, 5.6, 2.4, "Tier 1: Ultra-Low Risk (Stablecoins)", [
        "USDT, USDC, DAI (3 coins)",
        "Mean Volatility: 0.4% | Max Drawdown: 0.3%",
        "Pegged cash equivalent, minimal price change."
    ], title_color=GREEN)
    
    add_card(s6, 6.8, 1.8, 5.7, 2.4, "Tier 2: Low-Moderate Risk (Blue-Chips)", [
        "BTC, ETH, BNB, SOL, XRP (5 coins)",
        "Mean Volatility: 64.8% | Max Drawdown: 65.5%",
        "Established large-cap coins with deep liquidity."
    ], title_color=BLUE)
    
    add_card(s6, 0.8, 4.4, 5.6, 2.4, "Tier 3: Moderate-High Risk (Utility/DeFi)", [
        "ADA, AVAX, DOT, LINK, DOGE, LTC, etc. (12 coins)",
        "Mean Volatility: 86.8% | Max Drawdown: 83.2%",
        "Active ecosystem growth tokens with higher swings."
    ], title_color=RGBColor(217, 119, 6))
    
    add_card(s6, 6.8, 4.4, 5.7, 2.4, "Tier 4: Extreme Risk (Meme Tokens)", [
        "BONK, WIF, FLOKI, SHIB, NEAR, XLM, etc. (8 coins)",
        "Mean Volatility: 114.2% | Max Drawdown: 87.7%",
        "High speculation, severe drops, unpredictable swings."
    ], title_color=RGBColor(220, 38, 38))

    # -------------------------------------------------------------
    # SLIDE 7: DIMENSIONALITY REDUCTION: PCA BASICS (MODULE 4)
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "PCA Basics & Volatility vs. Volume Frontier (Module 4)")
    
    add_card(s7, 0.8, 1.8, 5.6, 5.0, "2D PCA Visualization", [
        "PCA reduces 5 features into 2 Principal Components.",
        "Component 1: Captures Price Volatility & Drawdowns (79.8%).",
        "Component 2: Captures Trading Volume Depth (17.1%).",
        "Total 2D Explained Variance: 96.94% (clear visual separation)."
    ], title_color=NAVY)
    
    add_card(s7, 6.8, 1.8, 5.7, 5.0, "Key Financial Findings", [
        "Stablecoins group together with massive volume and near-zero volatility.",
        "Bitcoin and Ethereum have deep liquidity that cushions drawdowns.",
        "Meme coins (BONK, WIF) show extreme price changes despite moderate volume."
    ], title_color=BLUE)

    # -------------------------------------------------------------
    # SLIDE 8: STREAMLIT DEPLOYMENT & SUMMARY (MODULE 5)
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Streamlit Deployment & Project Summary (Module 5)")
    
    add_card(s8, 0.8, 1.8, 5.6, 5.0, "Interactive Streamlit App (app.py)", [
        "Interactive Volatility vs. Volume scatter plot.",
        "Searchable Risk Matrix table for all 28 assets.",
        "Live Risk Classifier: Predicts risk tier for custom coins.",
        "Portfolio Allocator: Suggests weights based on risk tolerance."
    ], title_color=GREEN)
    
    add_card(s8, 6.8, 1.8, 5.7, 5.0, "Capstone Checklist Completed", [
        "Phase 1 Proposal: Project title, dataset, objectives.",
        "Phase 2 & 3: Cleaned data and ETL feature notebook.",
        "Phase 4 & 5: K-Means clustering, PCA, and Streamlit demo.",
        "Phase 6: 10-page final report, 8 slides PPTX, Viva guide.",
        "All code version-controlled on GitHub."
    ], title_color=NAVY)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"[+] Presentation slide deck successfully saved to: {output_path}")

if __name__ == "__main__":
    create_presentation()
