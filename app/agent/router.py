"""Intent router for classifying user queries."""
import re
from typing import Tuple, List, Optional

from app.agent.schemas import Intent, ChartType
from app.services.ticker_mapper import extract_tickers, parse_time_period


CHART_INDICATORS = {
    "candlestick": ["candle", "candlestick", "ohlc", "ohlcv"],
    "rsi": ["rsi", "relative strength"],
    "macd": ["macd", "moving average convergence"],
    "bollinger": ["bollinger", "bb", "band"],
    "sma": ["sma", "simple moving average", "ma"],
    "ema": ["ema", "exponential moving average"],
    "volume": ["volume", "vol"],
    "volatility": ["volatility", "vol", "standard deviation"],
    "drawdown": ["drawdown", "peak to trough", "max loss"],
    "correlation": ["correlation", "correlate", "heatmap"],
    "histogram": ["histogram", "distribution", "returns distribution"],
}


INTENT_PATTERNS = {
    Intent.QUOTE_LOOKUP: [
        r"price", r"quote", r"current.*price", r"trading.*at",
        r"last.*price", r"what.*worth", r"trading.*price"
    ],
    Intent.HISTORICAL_PERFORMANCE: [
        r"performance", r"return", r"ytd", r"year.*to.*date",
        r"gained", r"lost", r"changed", r"movement", r"price.*movement",
        r"over.*(past|last|month|year)",
        r"(1m|3m|6m|1y|2y|ytd).*(return|performance)"
    ],
    Intent.TECHNICAL_ANALYSIS: [
        r"technical", r"indicator", r"analysis", r"pattern",
        r"support", r"resistance", r"trend", r"signal", r"ma\s*\d+",
        r"moving\s+average", r"\b\d+ma\b"
    ],
    Intent.CROSS_TICKER_COMPARISON: [
        r"compare", r"comparison", r"versus", r"vs\.?", r"against",
        r"which.*(better|worse|more|less)"
    ],
    Intent.COMPANY_FUNDAMENTALS: [
        r"fundamental", r"pe.*ratio", r"market.*cap", r"earnings",
        r"revenue", r"profit", r"dividend", r"balance.*sheet",
        r"financial", r"valuation"
    ],
    Intent.NEWS_CONTEXT: [
        r"news", r"happened", r"recent", r"announcement",
        r"earnings.*call", r"quarterly", r"report"
    ],
    Intent.VISUALIZATION_REQUEST: [
        r"plot", r"chart", r"graph", r"visualize", r"show.*(me|a)",
        r"display", r"render"
    ],
    Intent.CODING_REQUEST: [
        r"write.*code", r"code", r"python", r"backtest",
        r"algorithm", r"calculate.*(sharpe|sortino|beta)"
    ],
}


CHART_TYPE_PATTERNS = {
    ChartType.PRICE_LINE: [r"line.*chart", r"price.*chart", r"close.*price"],
    ChartType.CANDLESTICK: [r"candlestick", r"candle", r"ohlc", r"bar.*chart"],
    ChartType.NORMALIZED_COMPARE: [r"normalized", r"base.*100", r"relative.*performance"],
    ChartType.ROLLING_VOLATILITY: [r"volatility", r"rolling.*vol", r"annualized.*vol"],
    ChartType.DRAWDOWN: [r"drawdown", r"peak.*trough", r"max.*drawdown"],
    ChartType.RSI_PANEL: [r"rsi", r"relative.*strength"],
    ChartType.MACD_PANEL: [r"macd"],
    ChartType.CORRELATION_HEATMAP: [r"correlation", r"heatmap", r"correlogram"],
    ChartType.RETURNS_HISTOGRAM: [r"histogram", r"distribution", r"returns.*dist"],
}


def classify_intent(text: str) -> Tuple[Intent, float]:
    """
    Classify the intent of a user query.
    
    Args:
        text: User input text
    
    Returns:
        tuple of (intent, confidence_score)
    """
    text_lower = text.lower()
    
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                score += 1
        scores[intent] = score
    
    if not scores or max(scores.values()) == 0:
        return Intent.GENERAL_QA, 0.5
    
    comparison_keywords = ["compare", "comparison", "versus", "vs", "against"]
    has_comparison = any(kw in text_lower for kw in comparison_keywords)
    
    max_score = max(scores.values())
    
    if has_comparison and Intent.CROSS_TICKER_COMPARISON in scores:
        comparison_score = scores[Intent.CROSS_TICKER_COMPARISON]
        if comparison_score >= max_score * 0.5:
            return Intent.CROSS_TICKER_COMPARISON, min(comparison_score / 2.0, 1.0)
    
    best_intent = max(scores, key=scores.get)
    confidence = min(max_score / 2.0, 1.0)
    
    return best_intent, confidence


def detect_chart_type(text: str) -> Optional[ChartType]:
    """
    Detect requested chart type from text.
    
    Args:
        text: User input text
    
    Returns:
        ChartType or None if no chart requested
    """
    text = text.lower()
    
    for chart_type, patterns in CHART_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return chart_type
    
    if "plot" in text or "chart" in text or "show" in text:
        if "compare" in text or "comparison" in text:
            return ChartType.NORMALIZED_COMPARE
        elif "candle" in text:
            return ChartType.CANDLESTICK
        else:
            return ChartType.PRICE_LINE
    
    return None


def detect_indicators(text: str) -> List[str]:
    """
    Detect requested technical indicators from text.
    
    Args:
        text: User input text
    
    Returns:
        List of indicator names
    """
    text = text.lower()
    indicators = []
    
    indicator_map = {
        "sma": ["sma", "simple moving average", "moving average"],
        "ema": ["ema", "exponential moving average"],
        "rsi": ["rsi", "relative strength index"],
        "macd": ["macd"],
        "bollinger": ["bollinger", "bb", "bollinger band"],
        "atr": ["atr", "average true range"],
        "adx": ["adx", "average directional index"],
        "stochastic": ["stochastic", "stoch"],
        "obv": ["obv", "on balance volume", "on-balance volume"],
        "vwap": ["vwap", "volume weighted average price"],
    }
    
    for indicator, keywords in indicator_map.items():
        if any(kw in text for kw in keywords):
            if indicator == "sma":
                if "20" in text:
                    indicators.append("sma_20")
                if "50" in text:
                    indicators.append("sma_50")
                if "200" in text:
                    indicators.append("sma_200")
                if not indicators:
                    indicators.append("sma_20")
            elif indicator == "ema":
                if "12" in text:
                    indicators.append("ema_12")
                if "26" in text:
                    indicators.append("ema_26")
                if not indicators:
                    indicators.append("ema_12")
            else:
                indicators.append(indicator)
    
    return list(set(indicators))


def route_request(text: str) -> dict:
    """
    Route a user request and extract all relevant components.
    
    Args:
        text: User input text
    
    Returns:
        dict with intent, tickers, period, chart_type, indicators
    """
    tickers = extract_tickers(text)
    period, interval, start_date = parse_time_period(text)
    
    intent, confidence = classify_intent(text)
    chart_type = detect_chart_type(text)
    indicators = detect_indicators(text)
    
    if chart_type and intent not in [Intent.VISUALIZATION_REQUEST, Intent.TECHNICAL_ANALYSIS]:
        intent = Intent.VISUALIZATION_REQUEST
    
    needs_chart = (
        intent == Intent.VISUALIZATION_REQUEST or
        chart_type is not None or
        "show" in text.lower() or
        "plot" in text.lower() or
        "chart" in text.lower()
    )
    
    return {
        "intent": intent,
        "confidence": confidence,
        "tickers": tickers,
        "period": period,
        "interval": interval,
        "start_date": start_date,
        "end_date": None,
        "chart_type": chart_type,
        "indicators": indicators,
        "needs_chart": needs_chart
    }


def needs_clarification(route_result: dict) -> Tuple[bool, Optional[str]]:
    """
    Check if a request needs clarification.
    
    Args:
        route_result: Result from route_request
    
    Returns:
        tuple of (needs_clarification, question)
    """
    tickers = route_result.get("tickers", [])
    intent = route_result.get("intent")
    
    if not tickers and intent not in [Intent.GENERAL_QA]:
        return True, "Which stock or ticker are you asking about?"
    
    if intent == Intent.CROSS_TICKER_COMPARISON and len(tickers) < 2:
        return True, "Which stocks would you like to compare? Please provide at least 2 tickers."
    
    if intent == Intent.CODING_REQUEST and not tickers:
        return True, "Which stock would you like to analyze with code?"
    
    return False, None
