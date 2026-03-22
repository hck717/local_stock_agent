"""Tests for router."""
import pytest
from app.agent.router import (
    classify_intent, detect_chart_type, detect_indicators,
    route_request, needs_clarification
)
from app.agent.schemas import Intent, ChartType


def test_classify_intent():
    """Test intent classification."""
    intent, confidence = classify_intent("Show me AAPL price")
    assert intent == Intent.QUOTE_LOOKUP
    
    intent, confidence = classify_intent("Compare AAPL vs MSFT")
    assert intent == Intent.CROSS_TICKER_COMPARISON
    
    intent, confidence = classify_intent("Plot NVDA candlestick")
    assert intent == Intent.VISUALIZATION_REQUEST

    intent, confidence = classify_intent("Show me MSFT fundamentals for last 6 months")
    assert intent == Intent.COMPANY_FUNDAMENTALS

    intent, confidence = classify_intent("Chart MSFT fundamentals")
    assert intent == Intent.COMPANY_FUNDAMENTALS


def test_detect_chart_type():
    """Test chart type detection."""
    assert detect_chart_type("Show candlestick") == ChartType.CANDLESTICK
    assert detect_chart_type("Plot line chart") == ChartType.PRICE_LINE
    assert detect_chart_type("Show volatility") == ChartType.ROLLING_VOLATILITY
    assert detect_chart_type("Plot drawdown") == ChartType.DRAWDOWN


def test_detect_indicators():
    """Test indicator detection."""
    indicators = detect_indicators("Show me RSI for AAPL")
    assert "rsi" in indicators
    
    indicators = detect_indicators("Plot SMA 20 and SMA 50")
    assert "sma_20" in indicators
    assert "sma_50" in indicators
    
    indicators = detect_indicators("Show MACD")
    assert "macd" in indicators


def test_route_request():
    """Test full request routing."""
    result = route_request("Show me AAPL price over 6 months")
    
    assert "AAPL" in result["tickers"]
    assert result["period"] == "6mo"
    
    result = route_request("Compare NVDA vs TSLA YTD with candlestick")
    assert result["chart_type"] == ChartType.CANDLESTICK
    assert len(result["tickers"]) >= 2

    result = route_request("Show me MSFT fundamentals for last 6 months")
    assert result["intent"] == Intent.COMPANY_FUNDAMENTALS


def test_needs_clarification():
    """Test clarification detection."""
    result = {"tickers": [], "intent": Intent.QUOTE_LOOKUP}
    needs_clar, question = needs_clarification(result)
    assert needs_clar is True
    
    result = {"tickers": ["AAPL"], "intent": Intent.CROSS_TICKER_COMPARISON}
    needs_clar, question = needs_clarification(result)
    assert needs_clar is True
    
    result = {"tickers": ["AAPL", "MSFT", "GOOGL"], "intent": Intent.CROSS_TICKER_COMPARISON}
    needs_clar, question = needs_clarification(result)
    assert needs_clar is False
