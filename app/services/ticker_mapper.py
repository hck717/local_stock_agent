"""Ticker mapping and normalization service."""
import re
from typing import Optional, List, Tuple


COMPANY_ALIASES = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "googl": "GOOGL",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "meta": "META",
    "facebook": "META",
    "fb": "META",
    "tesla": "TSLA",
    "tsla": "TSLA",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "netflix": "NFLX",
    "nflx": "NFLX",
    "twitter": "TWTR",
    "twtr": "TWTR",
    "x": "TWTR",
    "jpmorgan": "JPM",
    "jpm": "JPM",
    "visa": "V",
    "walmart": "WMT",
    "wmt": "WMT",
    "disney": "DIS",
    "dis": "DIS",
    "nike": "NKE",
    "nke": "NKE",
    "tencent": "0700.HK",
    "0700": "0700.HK",
    "alibaba": "9988.HK",
    "9988": "9988.HK",
    "baba": "9988.HK",
    "hkex": "0388.HK",
    "0388": "0388.HK",
    "hsbc": "0005.HK",
    "0005": "0005.HK",
    "baidu": "9888.HK",
    "9888": "9888.HK",
    "xiaomi": "1810.HK",
    "1810": "1810.HK",
    "byd": "1211.HK",
    "1211": "1211.HK",
    "meituan": "3690.HK",
    "3690": "3690.HK",
    "kuaishou": "1024.HK",
    "1024": "1024.HK",
    "ctrip": "9961.HK",
    "9961": "9961.HK",
    "mtr": "0066.HK",
    "hk electric": "2638.HK",
    "hke": "2638.HK",
}


MARKET_SUFFIXES = {
    ".HK": "HK",
    ".SS": "China (Shanghai)",
    ".SZ": "China (Shenzhen)",
    ".T": "Japan",
    ".L": "UK",
    ".PA": "France",
    ".DE": "Germany",
    ".AS": "Netherlands",
    ".TO": "Canada",
    ".AX": "Australia",
    ".NZ": "New Zealand",
    ".KS": "South Korea",
    ".BK": "Thailand",
    ".SI": "Singapore",
}


US_SUFFIXES = {
    "": "US",
    "-USD": "Crypto",
    "-USDT": "Crypto",
}


def normalize_ticker(ticker: str) -> str:
    """
    Normalize a ticker symbol.
    
    Args:
        ticker: Raw ticker string
    
    Returns:
        Normalized ticker symbol
    """
    ticker = ticker.strip().upper()
    
    if ticker.lower() in COMPANY_ALIASES:
        return COMPANY_ALIASES[ticker.lower()]
    
    for suffix, _ in MARKET_SUFFIXES.items():
        if ticker.endswith(suffix):
            return ticker
    
    for suffix in [".HK", ".SS", ".SZ", ".T", ".L", ".PA", ".DE", ".AS", ".TO", ".AX", ".NZ", ".KS", ".BK", ".SI"]:
        if suffix in ticker:
            return ticker
    
    if re.match(r'^\d{4}\.HK$', ticker):
        return ticker
    
    return ticker


COMMON_WORDS = {
    "ME", "MY", "A", "AN", "THE", "IS", "ARE", "WAS", "WERE",
    "BE", "BEEN", "BEING", "HAVE", "HAS", "HAD", "DO", "DOES", "DID",
    "WILL", "WOULD", "COULD", "SHOULD", "MAY", "MIGHT", "CAN",
    "THIS", "THAT", "THESE", "THOSE", "HERE", "THERE", "WHEN", "WHERE",
    "WHY", "HOW", "WHAT", "WHO", "WHICH", "SHOW", "ME", "OVER", "UNDER",
    "THROUGH", "THAN", "THEN", "AFTER", "BEFORE", "WITH", "WITHIN",
    "WITHOUT", "FOR", "FROM", "INTO", "TO", "OF", "AND", "OR", "BUT",
    "IF", "SO", "AS", "IT", "ITS", "AT", "ON", "BY", "UP", "OUT",
    "ALL", "ANY", "SOME", "NO", "NOT", "ONLY", "JUST", "TOO", "VERY",
    "PRICE", "PRICES", "STOCK", "STOCKS", "SHARE", "SHARES",
    "YEAR", "YEARS", "MONTH", "MONTHS", "WEEK", "WEEKS", "DAY", "DAYS",
    "RETURN", "RETURNS", "COMPARE", "COMPARISON", "CHART", "PLOT",
    "COMPARE", "COMPARISON", "VS", "VERSUS",
    "YTD", "TODAY", "YESTERDAY", "TOMORROW",
    "MAX", "SINCE", "PAST", "LAST", "NEXT",
    "PERFORMANCE", "ANALYSIS", "STATS", "STATISTICS",
    "GAIN", "GAINS", "LOSS", "LOSSES",
    "HIGH", "LOW", "HIGHS", "LOWS",
    "RSI", "MACD", "SMA", "EMA", "BB", "BOLLINGER",
    "ATR", "ADX", "STOCH", "OBV", "VWAP", "CCI", "ROC",
    "HK", "US", "CN", "JP", "UK", "EU",
}


def extract_tickers(text: str) -> List[str]:
    """
    Extract ticker symbols from text.
    
    Args:
        text: Text containing tickers
    
    Returns:
        List of normalized ticker symbols
    """
    text_upper = text.upper()
    text_lower = text.lower()

    pattern_specs = [
        (r'\b([A-Z]{1,5})\b', 0),
        (r'\b(\d{4}\.[A-Z]{2,3})\b', re.IGNORECASE),
        (r'\b(\d{4}\.(?:SS|SZ))\b', re.IGNORECASE),
        (r'\b([A-Z]{1,5}\.[A-Z]{1,3})\b', 0),
        (r'\b([A-Z]{2,6}-[A-Z]{2,5})\b', re.IGNORECASE),
        (r'\$([A-Z]{1,10})\b', re.IGNORECASE),
    ]

    tickers = set()

    for pattern, flags in pattern_specs:
        matches = re.findall(pattern, text, flags=flags)
        tickers.update(matches)
    
    for alias, ticker in COMPANY_ALIASES.items():
        alias_pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(alias_pattern, text_lower) or re.search(alias_pattern, text_upper):
            tickers.add(ticker)
    
    normalized = []
    seen = set()
    for t in tickers:
        if t.upper() in COMMON_WORDS:
            continue
        norm = normalize_ticker(t)
        if norm and norm not in seen and len(norm) >= 1:
            seen.add(norm)
            normalized.append(norm)
    
    return normalized


def parse_time_period(text: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse time period from text.
    
    Args:
        text: User input text
    
    Returns:
        tuple of (period, interval, start_date)
    """
    text = text.lower()
    
    period_map = {
        "ytd": "ytd",
        "year to date": "ytd",
        "this year": "ytd",
        "today": "1d",
        "this week": "5d",
        "past week": "5d",
        "1w": "1wk",
        "week": "1wk",
        "2w": "2wk",
        "month": "1mo",
        "3mo": "3mo",
        "3 months": "3mo",
        "6mo": "6mo",
        "6 months": "6mo",
        "1y": "1y",
        "1 year": "1y",
        "past year": "1y",
        "2y": "2y",
        "2 years": "2y",
        "3y": "3y",
        "3 years": "3y",
        "5y": "5y",
        "5 years": "5y",
        "10y": "10y",
        "10 years": "10y",
        "max": "max",
        "maximum": "max",
    }
    
    sorted_phrases = sorted(period_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    for phrase, period in sorted_phrases:
        if phrase in text:
            if period in ["1d", "5d"]:
                interval = "5m" if period == "1d" else "15m"
            elif period in ["1wk", "2wk"]:
                interval = "30m"
            elif period in ["1mo", "3mo"]:
                interval = "1h"
            elif period in ["6mo", "1y", "ytd"]:
                interval = "1d"
            elif period in ["2y", "3y", "5y", "10y"]:
                interval = "1wk"
            elif period in ["max"]:
                interval = "1wk"
            else:
                interval = "1d"
            return period, interval, None

    relative_match = re.search(r'(?:last|past)\s+(\d+)\s*(day|days|week|weeks|month|months|year|years)', text)
    if relative_match:
        qty = int(relative_match.group(1))
        unit = relative_match.group(2)
        if unit.startswith("day"):
            return (f"{qty}d", "5m" if qty <= 5 else "1h", None)
        if unit.startswith("week"):
            return (f"{qty}wk", "30m" if qty <= 2 else "1d", None)
        if unit.startswith("month"):
            return (f"{qty}mo", "1h" if qty <= 3 else "1d", None)
        if unit.startswith("year"):
            if qty in [1, 2, 3, 5, 10]:
                period = f"{qty}y"
            else:
                period = "max" if qty > 10 else f"{qty}y"
            interval = "1d" if qty <= 1 else "1wk"
            return period, interval, None
    
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    dates = re.findall(date_pattern, text)
    if len(dates) >= 2:
        return "custom", "1d", dates[0]
    elif len(dates) == 1:
        return "custom", "1d", dates[0]
    
    return "6mo", "1d", None


def detect_market(ticker: str) -> str:
    """
    Detect the market for a ticker.
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        Market name
    """
    ticker = ticker.upper()
    
    if ticker.endswith(".HK"):
        return "HK"
    elif ticker.endswith(".SS") or ticker.endswith(".SZ"):
        return "China"
    elif ticker.endswith((".T", ".PA", ".DE", ".AS", ".L")):
        return "International"
    elif ticker.endswith(".TO"):
        return "Canada"
    elif ticker.endswith(".AX"):
        return "Australia"
    elif ticker.endswith(".NZ"):
        return "New Zealand"
    elif ticker.endswith(".KS"):
        return "South Korea"
    elif "-" in ticker:
        return "Crypto/Alt"
    else:
        return "US"


def validate_ticker(ticker: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a ticker symbol format.
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        tuple of (is_valid, error_message)
    """
    if not ticker or len(ticker.strip()) == 0:
        return False, "Ticker cannot be empty"
    
    ticker = ticker.strip()
    
    if re.match(r'^\d{4}\.[A-Z]{2,3}$', ticker):
        return True, None
    
    if re.match(r'^[A-Z]{1,10}(?:-[A-Z]{3,5})?$', ticker):
        return True, None
    
    if re.match(r'^[A-Z]{2,10}\.[A-Z]{2,3}$', ticker):
        return True, None
    
    if re.match(r'^\^[A-Z]{1,5}$', ticker):
        return True, None
    
    return True, None


def suggest_ticker_format(ticker: str) -> List[str]:
    """
    Suggest possible ticker formats.
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        List of suggested formats
    """
    ticker = ticker.upper().strip()
    suggestions = [ticker]
    
    if not "." in ticker:
        suggestions.append(f"{ticker}.HK")
    
    if not ticker.startswith("0") and len(ticker) == 4:
        suggestions.append(f"{ticker}.HK")
    
    if ticker in COMPANY_ALIASES:
        suggestions.append(COMPANY_ALIASES[ticker.lower()])
    
    return suggestions
