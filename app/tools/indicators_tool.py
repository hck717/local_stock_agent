"""Technical indicators computation tool."""
from typing import List, Optional
import pandas as pd
import numpy as np


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to match expected format."""
    result_df = df.copy()
    col_mapping = {}
    for col in result_df.columns:
        lower = str(col).lower()
        if lower == 'open' and 'Open' not in result_df.columns:
            col_mapping[col] = 'Open'
        elif lower == 'high' and 'High' not in result_df.columns:
            col_mapping[col] = 'High'
        elif lower == 'low' and 'Low' not in result_df.columns:
            col_mapping[col] = 'Low'
        elif lower == 'close' and 'Close' not in result_df.columns:
            col_mapping[col] = 'Close'
        elif lower == 'volume' and 'Volume' not in result_df.columns:
            col_mapping[col] = 'Volume'
    if col_mapping:
        result_df = result_df.rename(columns=col_mapping)
    return result_df


def compute_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """
    Compute technical indicators on price data.
    
    Args:
        df: DataFrame with OHLCV columns
        indicators: List of indicator names to compute
    
    Returns:
        DataFrame with added indicator columns
    """
    result_df = _normalize_columns(df)
    
    if 'Close' not in result_df.columns:
        return result_df
    
    close = result_df['Close']
    high = result_df.get('High', close)
    low = result_df.get('Low', close)
    volume = result_df.get('Volume', pd.Series(0, index=close.index))
    
    for indicator in indicators:
        indicator = indicator.lower().strip()
        
        if indicator == 'sma_20' or indicator == 'sma20':
            result_df['SMA_20'] = close.rolling(window=20).mean()
        elif indicator == 'sma_50' or indicator == 'sma50':
            result_df['SMA_50'] = close.rolling(window=50).mean()
        elif indicator == 'sma_200' or indicator == 'sma200':
            result_df['SMA_200'] = close.rolling(window=200).mean()
        elif indicator == 'ema_12' or indicator == 'ema12':
            result_df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
        elif indicator == 'ema_26' or indicator == 'ema26':
            result_df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
        elif indicator == 'rsi_14' or indicator == 'rsi14' or indicator == 'rsi':
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result_df['RSI_14'] = 100 - (100 / (1 + rs))
        elif indicator == 'macd':
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            result_df['MACD'] = ema12 - ema26
            result_df['MACD_Signal'] = result_df['MACD'].ewm(span=9, adjust=False).mean()
            result_df['MACD_Histogram'] = result_df['MACD'] - result_df['MACD_Signal']
        elif indicator == 'bollinger_bands' or indicator == 'bb':
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            result_df['BB_Upper'] = sma20 + (std20 * 2)
            result_df['BB_Middle'] = sma20
            result_df['BB_Lower'] = sma20 - (std20 * 2)
        elif indicator == 'atr' or indicator == 'atr_14':
            high_low = high - low
            high_close = (high - close.shift()).abs()
            low_close = (low - close.shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            result_df['ATR_14'] = true_range.rolling(window=14).mean()
        elif indicator == 'adx' or indicator == 'adx_14':
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where(plus_dm > 0, 0)
            minus_dm = minus_dm.where(minus_dm > 0, 0)
            
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
            result_df['ADX_14'] = dx.rolling(window=14).mean()
        elif indicator == 'stochastic' or indicator == 'stoch':
            low14 = low.rolling(window=14).min()
            high14 = high.rolling(window=14).max()
            result_df['Stoch_K'] = 100 * ((close - low14) / (high14 - low14))
            result_df['Stoch_D'] = result_df['Stoch_K'].rolling(window=3).mean()
        elif indicator == 'obv':
            result_df['OBV'] = (np.sign(close.diff()) * volume.fillna(0)).cumsum()
        elif indicator == 'vwap':
            cumulative_volume = volume.cumsum()
            cumulative_price_volume = (close * volume.fillna(0)).cumsum()
            result_df['VWAP'] = cumulative_price_volume / cumulative_volume
        elif indicator == 'returns':
            result_df['Returns'] = close.pct_change()
        elif indicator == 'log_returns':
            result_df['Log_Returns'] = np.log(close / close.shift(1))
    
    return result_df


def compute_summary_stats(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Compute summary statistics for a ticker.
    
    Args:
        df: DataFrame with price data
        ticker: Ticker symbol for labeling
    
    Returns:
        dict with summary statistics
    """
    result_df = _normalize_columns(df)
    
    if 'Close' not in result_df.columns or result_df['Close'].empty:
        return {"ticker": ticker, "error": "No close price data available"}
    
    close = result_df['Close']
    returns = close.pct_change().dropna()
    
    cumulative_return = 0.0
    if len(close) > 1 and close.iloc[0] != 0:
        cumulative_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    
    volatility_daily = returns.std() * 100 if len(returns) > 0 else 0
    volatility_annual = volatility_daily * np.sqrt(252)
    
    peak = close.cummax()
    drawdown = (close - peak) / peak * 100
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    
    stats = {
        "ticker": ticker,
        "current_price": float(close.iloc[-1]) if len(close) > 0 else None,
        "start_price": float(close.iloc[0]) if len(close) > 0 else None,
        "cumulative_return_pct": round(cumulative_return, 2),
        "daily_volatility_pct": round(volatility_daily, 4),
        "annualized_volatility_pct": round(volatility_annual, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "avg_daily_return_pct": round(returns.mean() * 100, 4) if len(returns) > 0 else 0,
        "positive_days_pct": round((returns > 0).sum() / len(returns) * 100, 2) if len(returns) > 0 else 0,
        "data_points": len(close)
    }
    
    return stats


def compute_correlation_matrix(data_dict: dict) -> dict:
    """
    Compute correlation matrix for multiple tickers.
    
    Args:
        data_dict: dict of {ticker: DataFrame or dict with records}
    
    Returns:
        dict with correlation matrix
    """
    closes = {}
    for ticker, data in data_dict.items():
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                closes[ticker] = data['Close']
        elif isinstance(data, dict) and 'records' in data:
            df = pd.DataFrame(data['records'])
            if 'Close' in df.columns:
                closes[ticker] = df['Close']
    
    if not closes:
        return {"status": "ERROR", "message": "No valid close prices found"}
    
    combined = pd.DataFrame(closes).dropna()
    
    if combined.empty or len(combined.columns) < 2:
        return {"status": "ERROR", "message": "Not enough data for correlation"}
    
    corr_matrix = combined.corr()
    
    return {
        "status": "success",
        "correlation_matrix": corr_matrix.to_dict(),
        "tickers": list(corr_matrix.columns),
        "data_points": len(combined)
    }
