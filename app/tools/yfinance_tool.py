"""yfinance data retrieval tool with automatic retry and HTTP fallback."""
from typing import Optional
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


MAX_RETRIES = 3
INITIAL_DELAY = 2


def _is_rate_limited(error_msg: str) -> bool:
    """Check if error is due to rate limiting."""
    error_lower = error_msg.lower()
    rate_limit_indicators = [
        "delisted", "not found", "no data found", 
        "rate limit", "429", "connection", "timeout",
        "network", "ssl", "expecting value", "too many requests"
    ]
    return any(indicator in error_lower for indicator in rate_limit_indicators)


def _period_to_days(period: str) -> int:
    """Convert yfinance period to approximate days."""
    period_map = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
        "6mo": 180, "1y": 365, "2y": 730, "3y": 1095,
        "5y": 1825, "10y": 3650, "ytd": 180, "max": 36500
    }
    return period_map.get(period, 180)


def _fetch_direct_http(ticker: str, period: str, interval: str) -> Optional[dict]:
    """Fallback: fetch data directly via HTTP request."""
    try:
        days = _period_to_days(period)
        end_date = int(datetime.now().timestamp())
        start_date = int((datetime.now() - timedelta(days=days)).timestamp())
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
        params = {
            'period1': start_date,
            'period2': end_date,
            'interval': interval if interval in ['1d', '1wk', '1m', '5m', '15m', '30m', '60m'] else '1d',
            'events': 'div,split'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 429:
            return None
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        if 'chart' not in data or 'result' not in data['chart']:
            return None
            
        result = data['chart']['result']
        if not result or result is None:
            return None
            
        result = result[0]
        timestamps = result.get('timestamp', [])
        quote = result.get('indicators', {}).get('quote', [{}])[0]
        
        if not timestamps or not quote:
            return None
        
        df = pd.DataFrame({
            'Datetime': pd.to_datetime(timestamps, unit='s'),
            'Open': quote.get('open', []),
            'High': quote.get('high', []),
            'Low': quote.get('low', []),
            'Close': quote.get('close', []),
            'Volume': quote.get('volume', [])
        })
        
        df = df.dropna(subset=['Close'])
        
        if df.empty:
            return None
            
        return df.to_dict(orient="records")
        
    except Exception:
        return None


def get_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None
) -> dict:
    """
    Fetch price history for a single ticker with automatic retry and HTTP fallback.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', '0700.HK')
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
    
    Returns:
        dict with data, metadata, and status
    """
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            stock = yf.Ticker(ticker)
            
            if start and end:
                df = stock.history(start=start, end=end, interval=interval)
            else:
                df = stock.history(period=period, interval=interval)
            
            if df is not None and not df.empty:
                df = df.reset_index()
                if 'Datetime' in df.columns:
                    df['Datetime'] = df['Datetime'].astype(str)
                elif 'Date' in df.columns:
                    df['Date'] = df['Date'].astype(str)
                
                return {
                    "status": "success",
                    "ticker": ticker,
                    "data": df.to_dict(orient="records"),
                    "columns": list(df.columns),
                    "fetched_at": datetime.now().isoformat(),
                    "period": period,
                    "interval": interval
                }
            
        except Exception as e:
            last_error = str(e)
        
        if attempt < MAX_RETRIES - 1:
            delay = INITIAL_DELAY * (2 ** attempt)
            time.sleep(delay)
    
    # Fallback: try direct HTTP request
    http_data = _fetch_direct_http(ticker, period, interval)
    if http_data:
        return {
            "status": "success",
            "ticker": ticker,
            "data": http_data,
            "columns": ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
            "fetched_at": datetime.now().isoformat(),
            "period": period,
            "interval": interval,
            "source": "http_fallback"
        }
    
    if last_error and _is_rate_limited(last_error):
        return {
            "status": "RATE_LIMIT_OR_SOURCE_ERROR",
            "ticker": ticker,
            "message": "Yahoo Finance rate limit exceeded. Please try again in a few minutes.",
            "data": None
        }
    
    return {
        "status": "NO_DATA",
        "ticker": ticker,
        "message": f"No data found for {ticker}",
        "data": None
    }


def get_multi_ticker_history(
    tickers: list[str],
    period: str = "6mo",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None
) -> dict:
    """
    Fetch price history for multiple tickers with automatic retry.
    
    Args:
        tickers: List of stock symbols
        period: Data period
        interval: Data interval
        start: Start date
        end: End date
    
    Returns:
        dict with combined data for all tickers
    """
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            if start and end:
                df = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', progress=False)
            else:
                df = yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=False)
            
            if df is None or df.empty:
                return {
                    "status": "NO_DATA",
                    "tickers": tickers,
                    "message": "No data found for provided tickers",
                    "data": None
                }
            
            result_data = {}
            missing_tickers = []
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        if df is not None and not df.empty:
                            ticker_df = df.reset_index()
                        else:
                            result_data[ticker] = {"error": "No data available"}
                            continue
                    else:
                        if isinstance(df.columns, pd.MultiIndex):
                            if ticker in df.columns.get_level_values(0):
                                ticker_df = df[ticker].copy().reset_index()
                            elif ticker in df.columns.get_level_values(-1):
                                ticker_df = df.xs(ticker, axis=1, level=-1).copy().reset_index()
                            else:
                                result_data[ticker] = {"error": "No data available for this ticker"}
                                missing_tickers.append(ticker)
                                continue
                        else:
                            if df is not None and not df.empty and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                                ticker_df = df.copy().reset_index()
                            else:
                                result_data[ticker] = {"error": "No data available for this ticker"}
                                missing_tickers.append(ticker)
                                continue
                    
                    result_data[ticker] = {
                        "records": ticker_df.to_dict(orient="records"),
                        "columns": list(ticker_df.columns)
                    }
                except Exception as e:
                    result_data[ticker] = {"error": str(e)}
                    missing_tickers.append(ticker)

            # Fallback: fetch missing tickers individually for better resilience
            for ticker in missing_tickers:
                single = get_price_history(ticker, period=period, interval=interval, start=start, end=end)
                if single.get("status") == "success" and single.get("data"):
                    single_df = pd.DataFrame(single["data"])
                    result_data[ticker] = {
                        "records": single_df.to_dict(orient="records"),
                        "columns": list(single_df.columns),
                        "source": single.get("source", "single_fallback")
                    }
                else:
                    result_data[ticker] = {
                        "error": single.get("message", "No data available for this ticker")
                    }

            successful = [t for t, v in result_data.items() if isinstance(v, dict) and v.get("records")]
            failed = [t for t, v in result_data.items() if not (isinstance(v, dict) and v.get("records"))]

            if not successful:
                return {
                    "status": "NO_DATA",
                    "tickers": tickers,
                    "message": "No data found for provided tickers",
                    "data": result_data,
                    "failed_tickers": failed
                }
            
            return {
                "status": "success",
                "tickers": tickers,
                "data": result_data,
                "fetched_at": datetime.now().isoformat(),
                "period": period,
                "interval": interval,
                "successful_tickers": successful,
                "failed_tickers": failed
            }
            
        except Exception as e:
            last_error = str(e)
            error_lower = last_error.lower()
            
            if _is_rate_limited(last_error) and attempt < MAX_RETRIES - 1:
                delay = INITIAL_DELAY * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                if "delisted" in error_lower or "not found" in error_lower or "no data" in error_lower:
                    status = "NO_DATA"
                elif "rate limit" in error_lower or "429" in error_lower or "timeout" in error_lower:
                    status = "RATE_LIMIT_OR_SOURCE_ERROR"
                else:
                    status = "ERROR"
                return {
                    "status": status,
                    "tickers": tickers,
                    "message": last_error,
                    "data": None
                }
    
    return {
        "status": "RATE_LIMIT_OR_SOURCE_ERROR",
        "tickers": tickers,
        "message": last_error or "Failed after multiple retries",
        "data": None
    }


def get_company_info(ticker: str) -> dict:
    """
    Fetch company metadata and info with automatic retry.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        dict with company information
    """
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or len(info) < 5:
                return {
                    "status": "NO_DATA",
                    "ticker": ticker,
                    "message": f"No company info found for {ticker}",
                    "data": None
                }
            
            relevant_fields = [
                "shortName", "longName", "sector", "industry",
                "marketCap", "currentPrice", "previousClose",
                "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
                "trailingPE", "forwardPE", "dividendYield",
                "volume", "averageVolume", "beta", "epsTrailingTwMonths",
                "enterpriseValue", "priceToBook", "priceToSalesTrailing12Months",
                "returnOnEquity", "profitMargins", "operatingMargins",
                "debtToEquity", "currentRatio", "quickRatio"
            ]
            
            filtered_info = {k: info.get(k) for k in relevant_fields if info.get(k) is not None}

            def _extract_from_statement(statement_df, possible_rows):
                try:
                    if statement_df is None or statement_df.empty:
                        return None
                    for row in possible_rows:
                        if row in statement_df.index:
                            row_vals = statement_df.loc[row].dropna()
                            if not row_vals.empty:
                                return float(row_vals.iloc[0])
                    return None
                except Exception:
                    return None

            financials = getattr(stock, "financials", None)
            quarterly_financials = getattr(stock, "quarterly_financials", None)
            balance_sheet = getattr(stock, "balance_sheet", None)
            cashflow = getattr(stock, "cashflow", None)

            statement_metrics = {
                "totalRevenue": _extract_from_statement(
                    quarterly_financials if quarterly_financials is not None and not quarterly_financials.empty else financials,
                    ["Total Revenue", "Revenue"]
                ),
                "netIncome": _extract_from_statement(
                    quarterly_financials if quarterly_financials is not None and not quarterly_financials.empty else financials,
                    ["Net Income", "Net Income Common Stockholders"]
                ),
                "operatingIncome": _extract_from_statement(
                    quarterly_financials if quarterly_financials is not None and not quarterly_financials.empty else financials,
                    ["Operating Income", "EBIT"]
                ),
                "totalAssets": _extract_from_statement(
                    balance_sheet,
                    ["Total Assets"]
                ),
                "totalDebt": _extract_from_statement(
                    balance_sheet,
                    ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
                ),
                "operatingCashFlow": _extract_from_statement(
                    cashflow,
                    ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
                ),
                "freeCashFlow": _extract_from_statement(
                    cashflow,
                    ["Free Cash Flow"]
                ),
            }
            for k, v in statement_metrics.items():
                if v is not None:
                    filtered_info[k] = v
            
            return {
                "status": "success",
                "ticker": ticker,
                "data": filtered_info,
                "fetched_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            last_error = str(e)
            error_lower = last_error.lower()
            
            if _is_rate_limited(last_error) and attempt < MAX_RETRIES - 1:
                delay = INITIAL_DELAY * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                if "delisted" in error_lower or "not found" in error_lower:
                    status = "NO_DATA"
                else:
                    status = "ERROR"
                return {
                    "status": status,
                    "ticker": ticker,
                    "message": last_error,
                    "data": None
                }
    
    return {
        "status": "RATE_LIMIT_OR_SOURCE_ERROR",
        "ticker": ticker,
        "message": last_error or "Failed after multiple retries",
        "data": None
    }


def get_live_quote(ticker: str) -> dict:
    """
    Get current/live quote for a ticker.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        dict with current price data
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        
        return {
            "status": "success",
            "ticker": ticker,
            "data": {
                "price": info.get("last_price"),
                "previous_close": info.get("previous_close"),
                "open": info.get("open"),
                "day_high": info.get("day_high"),
                "day_low": info.get("day_low"),
                "volume": info.get("last_volume"),
                "market_cap": info.get("market_cap"),
                "currency": info.get("currency")
            },
            "fetched_at": datetime.now().isoformat(),
            "warning": "Yahoo Finance data may be delayed."
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "ticker": ticker,
            "message": str(e),
            "data": None
        }
