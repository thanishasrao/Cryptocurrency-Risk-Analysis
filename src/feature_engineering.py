"""
Feature Engineering Module for Cryptocurrency Risk Analysis
Calculates quantitative risk (price volatility, drawdowns, tail risk) and
trading liquidity (trading volume, volume stability, Amihud illiquidity) metrics.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

def compute_crypto_risk_features(input_path="data/raw/crypto_historical_raw.csv", 
                                 output_path="data/processed/crypto_risk_features.csv"):
    """
    Extracts cross-sectional risk and liquidity metrics for each cryptocurrency.
    
    Parameters:
    -----------
    input_path : str
        Path to raw historical multi-asset CSV.
    output_path : str
        Path to save extracted feature matrix.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame of extracted features per cryptocurrency.
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
        
        # Prices and returns
        closes = group["Close"].values
        highs = group["High"].values
        lows = group["Low"].values
        volumes = group["Volume"].values
        
        # Calculate daily simple returns: R_t = (P_t - P_{t-1}) / P_{t-1}
        returns = np.diff(closes) / closes[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) < 30:
            continue
            
        # 1. Price Volatility & Return Metrics
        daily_mean_ret = np.mean(returns)
        annualized_return_pct = daily_mean_ret * 365 * 100
        daily_vol = np.std(returns, ddof=1)
        annualized_volatility_pct = daily_vol * np.sqrt(365) * 100
        
        # Intraday High-Low spread percentage: (High - Low) / Close
        hl_spread = (highs - lows) / closes * 100
        mean_daily_spread_pct = np.mean(hl_spread)
        max_daily_spread_pct = np.max(hl_spread)
        
        # 2. Drawdown Metrics
        # Peak-to-trough decline
        cum_max = np.maximum.accumulate(closes)
        drawdowns = (closes - cum_max) / cum_max * 100
        max_drawdown_pct = np.abs(np.min(drawdowns))
        avg_drawdown_pct = np.abs(np.mean(drawdowns))
        
        # 3. Value at Risk (VaR) & Tail Risk
        # Historical VaR at 95% and 99% confidence (expressed as positive loss percentage)
        var_95_pct = -np.percentile(returns, 5) * 100
        var_99_pct = -np.percentile(returns, 1) * 100
        
        # Conditional Value at Risk (CVaR / Expected Shortfall at 95%)
        tail_returns = returns[returns <= np.percentile(returns, 5)]
        cvar_95_pct = -np.mean(tail_returns) * 100 if len(tail_returns) > 0 else var_95_pct
        
        # Downside Deviation (annualized volatility of negative returns)
        neg_returns = np.minimum(returns, 0.0)
        downside_deviation_pct = np.sqrt(np.mean(neg_returns**2)) * np.sqrt(365) * 100
        
        # Sortino Ratio (Return / Downside Risk, assuming 0 risk-free rate for standard crypto benchmark)
        sortino_ratio = (annualized_return_pct / downside_deviation_pct) if downside_deviation_pct > 1e-4 else 0.0
        
        # Return distribution shape
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))  # Fisher's excess kurtosis
        
        # 4. Trading & Liquidity Metrics
        valid_volumes = volumes[volumes > 0]
        avg_daily_volume_usd = np.mean(valid_volumes) if len(valid_volumes) > 0 else 1.0
        log_avg_volume_usd = np.log10(avg_daily_volume_usd) if avg_daily_volume_usd > 0 else 0.0
        std_volume = np.std(valid_volumes) if len(valid_volumes) > 0 else 0.0
        volume_volatility = (std_volume / avg_daily_volume_usd) if avg_daily_volume_usd > 0 else 0.0
        
        # Amihud Illiquidity Ratio Proxy: average(|Return_t| / Volume_t) * 1e9
        # Filter for days with meaningful trading activity (> $100k) to prevent zero-volume division spikes
        robust_mask = (volumes[:-1] > 100000)
        if np.sum(robust_mask) > 10:
            amihud_values = np.abs(returns[robust_mask]) / volumes[:-1][robust_mask] * 1e9
            amihud_illiquidity = float(np.median(amihud_values))
        else:
            amihud_illiquidity = 0.0
            
        log_amihud = float(np.log10(amihud_illiquidity + 1e-4))
            
        # Volatility to Volume Ratio (Speculation Index)
        vol_to_volume_ratio = annualized_volatility_pct / (log_avg_volume_usd + 1e-5)
        
        records.append({
            "Symbol": symbol,
            "Name": name,
            "Ticker": ticker,
            "Asset_Category": category,
            "Annualized_Volatility_%": round(annualized_volatility_pct, 2),
            "Annualized_Return_%": round(annualized_return_pct, 2),
            "Max_Drawdown_%": round(max_drawdown_pct, 2),
            "Avg_Drawdown_%": round(avg_drawdown_pct, 2),
            "VaR_95_%": round(var_95_pct, 2),
            "VaR_99_%": round(var_99_pct, 2),
            "CVaR_95_%": round(cvar_95_pct, 2),
            "Downside_Deviation_%": round(downside_deviation_pct, 2),
            "Sortino_Ratio": round(sortino_ratio, 2),
            "Mean_Daily_Spread_%": round(mean_daily_spread_pct, 2),
            "Return_Skewness": round(skewness, 3),
            "Return_Kurtosis": round(kurtosis, 3),
            "Avg_Daily_Volume_USD": float(avg_daily_volume_usd),
            "Log_Avg_Volume_USD": round(log_avg_volume_usd, 3),
            "Volume_Volatility_CV": round(volume_volatility, 3),
            "Amihud_Illiquidity": round(amihud_illiquidity, 4),
            "Log_Amihud_Illiquidity": round(log_amihud, 4),
            "Vol_to_Volume_Ratio": round(vol_to_volume_ratio, 3)
        })
        
    features_df = pd.DataFrame(records)
    
    # Sort by Annualized Volatility descending
    features_df = features_df.sort_values("Annualized_Volatility_%", ascending=False).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)
    
    print(f"[+] Successfully extracted {len(features_df)} cryptocurrency risk profiles with {len(features_df.columns)} metrics.")
    print(f"[+] Output saved to: {output_path}")
    return features_df


if __name__ == "__main__":
    compute_crypto_risk_features()
