"""
Data Collection Module for Cryptocurrency Analysis
Fetches historical daily OHLCV market data for top cryptocurrencies across multiple categories.
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Curated list of cryptocurrencies spanning major asset classes
# (Large Cap Blue-Chips, Layer 1/2 Ecosystems, DeFi, Speculative/Meme, Stablecoins)
CRYPTO_TICKERS = {
    # Blue-Chips
    "BTC-USD": {"name": "Bitcoin", "symbol": "BTC", "category": "Blue-Chip Store of Value"},
    "ETH-USD": {"name": "Ethereum", "symbol": "ETH", "category": "Smart Contract Platform"},
    "BNB-USD": {"name": "Binance Coin", "symbol": "BNB", "category": "Exchange / Ecosystem"},
    
    # Layer 1 / Layer 2 Platforms
    "SOL-USD": {"name": "Solana", "symbol": "SOL", "category": "High-Throughput L1"},
    "XRP-USD": {"name": "Ripple", "symbol": "XRP", "category": "Payment / Settlement"},
    "ADA-USD": {"name": "Cardano", "symbol": "ADA", "category": "Proof-of-Stake L1"},
    "AVAX-USD": {"name": "Avalanche", "symbol": "AVAX", "category": "Modular L1"},
    "DOT-USD": {"name": "Polkadot", "symbol": "DOT", "category": "Interoperability Protocol"},
    "NEAR-USD": {"name": "Near Protocol", "symbol": "NEAR", "category": "Sharded L1"},
    "MATIC-USD": {"name": "Polygon", "symbol": "MATIC", "category": "Ethereum L2 Scaling"},
    "APT-USD": {"name": "Aptos", "symbol": "APT", "category": "Move-based L1"},
    "SUI-USD": {"name": "Sui", "symbol": "SUI", "category": "Move-based L1"},
    "ATOM-USD": {"name": "Cosmos", "symbol": "ATOM", "category": "Interoperability Hub"},
    "FTM-USD": {"name": "Fantom", "symbol": "FTM", "category": "EVM L1 / Sonic"},
    "ALGO-USD": {"name": "Algorand", "symbol": "ALGO", "category": "Pure PoS L1"},
    
    # DeFi & Oracles
    "LINK-USD": {"name": "Chainlink", "symbol": "LINK", "category": "Decentralized Oracle"},
    "UNI-USD": {"name": "Uniswap", "symbol": "UNI", "category": "Decentralized AMM Exchange"},
    "AAVE-USD": {"name": "Aave", "symbol": "AAVE", "category": "Lending Protocol"},
    "MKR-USD": {"name": "Maker", "symbol": "MKR", "category": "DeFi Governance / CDP"},
    "LDO-USD": {"name": "Lido DAO", "symbol": "LDO", "category": "Liquid Staking"},
    "CRV-USD": {"name": "Curve DAO", "symbol": "CRV", "category": "Stablecoin AMM"},
    
    # Speculative / High-Beta / Meme Tokens
    "DOGE-USD": {"name": "Dogecoin", "symbol": "DOGE", "category": "Meme Currency"},
    "SHIB-USD": {"name": "Shiba Inu", "symbol": "SHIB", "category": "Ecosystem Meme Token"},
    "PEPE-USD": {"name": "Pepe", "symbol": "PEPE", "category": "Community Meme Token"},
    "BONK-USD": {"name": "Bonk", "symbol": "BONK", "category": "Solana Meme Token"},
    "FLOKI-USD": {"name": "Floki", "symbol": "FLOKI", "category": "Utility Meme Token"},
    "WIF-USD": {"name": "dogwifhat", "symbol": "WIF", "category": "Speculative Meme Coin"},
    
    # Legacy / Payment
    "LTC-USD": {"name": "Litecoin", "symbol": "LTC", "category": "Payment Network"},
    "BCH-USD": {"name": "Bitcoin Cash", "symbol": "BCH", "category": "Payment Network"},
    "XLM-USD": {"name": "Stellar Lumens", "symbol": "XLM", "category": "Cross-Border Payment"},
    "ETC-USD": {"name": "Ethereum Classic", "symbol": "ETC", "category": "Legacy Smart Contract"},
    
    # Pegged Stablecoins (Control / Zero-Volatility Baseline)
    "USDT-USD": {"name": "Tether", "symbol": "USDT", "category": "USD Pegged Stablecoin"},
    "USDC-USD": {"name": "USD Coin", "symbol": "USDC", "category": "USD Pegged Stablecoin"},
    "DAI-USD": {"name": "Dai", "symbol": "DAI", "category": "Decentralized Stablecoin"}
}


def download_crypto_historical(tickers=None, period="2y", interval="1d", output_path="data/raw/crypto_historical_raw.csv"):
    """
    Downloads historical OHLCV data for specified cryptocurrency tickers.
    
    Parameters:
    -----------
    tickers : dict, optional
        Dictionary mapping ticker symbols to metadata.
    period : str
        Historical period to download (default '2y' for 2 years).
    interval : str
        Data interval (default '1d' for daily frequency).
    output_path : str
        File path to save the consolidated raw CSV dataset.
        
    Returns:
    --------
    pd.DataFrame
        Consolidated multi-ticker historical DataFrame.
    """
    if tickers is None:
        tickers = CRYPTO_TICKERS
        
    print(f"[*] Starting download for {len(tickers)} cryptocurrencies over period='{period}'...")
    
    all_data = []
    failed_tickers = []
    
    for ticker_symbol, meta in tickers.items():
        try:
            print(f"    Fetching {meta['name']} ({ticker_symbol})...", end=" ")
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty or len(df) < 30:
                print(f"FAILED (insufficient data: {len(df)} rows)")
                failed_tickers.append(ticker_symbol)
                continue
                
            # Reset index to get Date column
            df = df.reset_index()
            # Standardize timezone-aware dates to naive date
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            elif "Datetime" in df.columns:
                df["Date"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
                
            df["Ticker"] = ticker_symbol
            df["Symbol"] = meta["symbol"]
            df["Name"] = meta["name"]
            df["Asset_Category"] = meta["category"]
            
            # Keep core columns
            cols_to_keep = ["Date", "Ticker", "Symbol", "Name", "Asset_Category", 
                            "Open", "High", "Low", "Close", "Volume"]
            existing_cols = [c for c in cols_to_keep if c in df.columns]
            df = df[existing_cols]
            
            all_data.append(df)
            print(f"OK ({len(df)} records)")
        except Exception as e:
            print(f"ERROR ({str(e)})")
            failed_tickers.append(ticker_symbol)
            
    if not all_data:
        raise RuntimeError("Failed to download any cryptocurrency data!")
        
    consolidated_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    consolidated_df.to_csv(output_path, index=False)
    
    print(f"\n[+] Successfully saved {len(consolidated_df)} total records across {consolidated_df['Symbol'].nunique()} cryptocurrencies.")
    print(f"[+] Output file: {output_path}")
    if failed_tickers:
        print(f"[-] Note: Skipped tickers due to issues: {failed_tickers}")
        
    return consolidated_df


if __name__ == "__main__":
    download_crypto_historical()
