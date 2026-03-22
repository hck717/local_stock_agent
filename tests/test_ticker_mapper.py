"""Tests for ticker mapper."""
import pytest
from app.services.ticker_mapper import (
    normalize_ticker, extract_tickers, parse_time_period,
    detect_market, validate_ticker, suggest_ticker_format
)


def test_normalize_ticker():
    """Test ticker normalization."""
    assert normalize_ticker("AAPL") == "AAPL"
    assert normalize_ticker("aapl") == "AAPL"
    assert normalize_ticker("0700.HK") == "0700.HK"
    assert normalize_ticker("tencent") == "0700.HK"
    assert normalize_ticker("msft") == "MSFT"


def test_extract_tickers():
    """Test ticker extraction from text."""
    assert "AAPL" in extract_tickers("Show me AAPL")
    assert "MSFT" in extract_tickers("Compare AAPL and MSFT")
    assert "0700.HK" in extract_tickers("0700.HK stock")
    assert "AAPL" in extract_tickers("Apple stock")
    assert len(extract_tickers("Show me AAPL vs MSFT")) >= 2
    assert "2638.HK" in extract_tickers("HK electric (2638.hk)")
    assert "2638.HK" in extract_tickers("Analyze HK Electric for 3 years")


def test_parse_time_period():
    """Test time period parsing."""
    period, interval, start = parse_time_period("Show me this year")
    assert period == "ytd"
    
    period, interval, start = parse_time_period("past week")
    assert period == "5d"
    
    period, interval, start = parse_time_period("1 year")
    assert period == "1y"

    period, interval, start = parse_time_period("last 3 years")
    assert period == "3y"


def test_detect_market():
    """Test market detection."""
    assert detect_market("AAPL") == "US"
    assert detect_market("0700.HK") == "HK"
    assert detect_market("9988.HK") == "HK"
    assert detect_market("BTC-USD") == "Crypto/Alt"


def test_validate_ticker():
    """Test ticker validation."""
    is_valid, error = validate_ticker("AAPL")
    assert is_valid is True
    
    is_valid, error = validate_ticker("0700.HK")
    assert is_valid is True
    
    is_valid, error = validate_ticker("")
    assert is_valid is False
