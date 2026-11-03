"""
Feature Engineering Module for Cryptocurrency Risk Analysis
CS2235 - Module 4: Feature Engineering & Preprocessing

Extracts intuitive, beginner-friendly financial risk and trading metrics:
1. Annualized Volatility (%) - How much price fluctuates daily
2. Maximum Drawdown (%) - Highest drop from peak price to lowest trough
3. Mean Daily Spread (%) - Average daily (High - Low) price fluctuation
4. Average Daily Volume (USD) - How much dollar volume is traded daily
5. Log10 Daily Volume - Normalized trading volume across orders of magnitude
6. Annualized Return (%) - Average annual percentage growth
"""

import os
import pandas as pd
import numpy as np

def compute_crypto_risk_features(input_path="data/raw/crypto_historical_raw.csv", 
                                 output_path="data/processed/crypto_risk_features.csv"):
    """
    Extracts cross-sectional risk and liquidity metrics for each cryptocurrency.
    Aligned with CS2235 Module 4 (Feature Engineering).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    
    records = []
    
    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").reset_index(drop=True)
        name = group["Name"].iloc[0]
        category = group["Asset_Category"].iloc[0]
        ticker = group["Ticker"].iloc[0]
        
        closes = group["Close"].values
        highs = group["High"].values
        lows = group["Low"].values
        volumes = group["Volume"].values
        
        # Calculate daily percentage returns: (Close_today - Close_yesterday) / Close_yesterday
        returns = np.diff(closes) / closes[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) < 30:
            continue
            
        # 1. Price Change: Annualized Return & Volatility (Module 4)
        daily_mean_ret = np.mean(returns)
        annualized_return_pct = daily_mean_ret * 365 * 100
        daily_vol = np.std(returns, ddof=1)
        annualized_volatility_pct = daily_vol * np.sqrt(365) * 100
        
        # 2. Intraday Price Spread: (High - Low) / Close
        hl_spread = (highs - lows) / closes * 100
        mean_daily_spread_pct = np.mean(hl_spread)
        
        # 3. Maximum Drawdown (MDD): Peak-to-trough worst drop percentage
        cum_max = np.maximum.accumulate(closes)
        drawdowns = (closes - cum_max) / cum_max * 100
        max_drawdown_pct = np.abs(np.min(drawdowns))
        
        # 4. Trading Volume Metrics: Average Volume and Log10 Volume (Module 4)
        valid_volumes = volumes[volumes > 0]
        avg_daily_volume_usd = np.mean(valid_volumes) if len(valid_volumes) > 0 else 1.0
        log_avg_volume_usd = np.log10(avg_daily_volume_usd) if avg_daily_volume_usd > 0 else 0.0
        
        # 5. Volatility-to-Volume Ratio: Intuitive ratio of price fluctuation to volume depth
        vol_to_volume_ratio = annualized_volatility_pct / (log_avg_volume_usd + 1e-5)
        
        records.append({
            "Symbol": symbol,
            "Name": name,
            "Ticker": ticker,
            "Asset_Category": category,
            "Annualized_Volatility_%": round(annualized_volatility_pct, 2),
            "Max_Drawdown_%": round(max_drawdown_pct, 2),
            "Mean_Daily_Spread_%": round(mean_daily_spread_pct, 2),
            "Annualized_Return_%": round(annualized_return_pct, 2),
            "Avg_Daily_Volume_USD": float(avg_daily_volume_usd),
            "Log_Avg_Volume_USD": round(log_avg_volume_usd, 3),
            "Vol_to_Volume_Ratio": round(vol_to_volume_ratio, 3)
        })
        
    features_df = pd.DataFrame(records)
    features_df = features_df.sort_values("Annualized_Volatility_%", ascending=False).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)
    
    print(f"[+] Successfully extracted {len(features_df)} cryptocurrency risk profiles.")
    print(f"[+] Output saved to: {output_path}")
    return features_df

if __name__ == "__main__":
    compute_crypto_risk_features()
