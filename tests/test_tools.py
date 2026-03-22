"""Tests for tools."""
import pytest
import pandas as pd
from app.tools.indicators_tool import compute_indicators, compute_summary_stats
from app.tools.chart_tool import build_chart, get_chart_catalog
from app.tools.code_runner import validate_code, generate_analysis_code


def test_compute_indicators():
    """Test indicator computation."""
    df = pd.DataFrame({
        'Close': [100, 102, 101, 105, 103, 107, 110, 108, 112, 115] * 5,
        'Open': [99, 101, 100, 104, 102, 106, 109, 107, 111, 114] * 5,
        'High': [101, 103, 102, 106, 104, 108, 111, 109, 113, 116] * 5,
        'Low': [98, 100, 99, 103, 101, 105, 108, 106, 110, 113] * 5,
        'Volume': [1000000] * 50
    })
    
    result = compute_indicators(df, ['sma_20', 'rsi_14'])
    
    assert 'SMA_20' in result.columns
    assert 'RSI_14' in result.columns
    assert not result['SMA_20'].isna().all()
    assert not result['RSI_14'].isna().all()


def test_compute_summary_stats():
    """Test summary statistics computation."""
    df = pd.DataFrame({
        'Close': [100, 102, 101, 105, 103, 107, 110, 108, 112, 115]
    })
    
    stats = compute_summary_stats(df, 'TEST')
    
    assert stats['ticker'] == 'TEST'
    assert 'current_price' in stats
    assert 'cumulative_return_pct' in stats
    assert 'annualized_volatility_pct' in stats
    assert 'max_drawdown_pct' in stats


def test_build_chart():
    """Test chart building."""
    df = pd.DataFrame({
        'Close': [100, 102, 101, 105, 103, 107, 110, 108, 112, 115],
        'Open': [99, 101, 100, 104, 102, 106, 109, 107, 111, 114],
        'High': [101, 103, 102, 106, 104, 108, 111, 109, 113, 116],
        'Low': [98, 100, 99, 103, 101, 105, 108, 106, 110, 113],
        'Volume': [1000000] * 10
    })
    
    result = build_chart(df, 'price_line', ['TEST'], save=False)
    
    assert result['status'] == 'success'
    assert result['chart_type'] == 'price_line'
    assert 'figure_json' in result


def test_chart_catalog():
    """Test chart catalog."""
    catalog = get_chart_catalog()
    
    assert 'price_line' in catalog
    assert 'candlestick_volume' in catalog
    assert 'rsi_panel' in catalog
    assert 'macd_panel' in catalog


def test_validate_code_safe():
    """Test code validation for safe code."""
    is_valid, error = validate_code("import pandas as pd\nprint('hello')")
    assert is_valid is True
    assert error is None


def test_validate_code_dangerous():
    """Test code validation for dangerous code."""
    is_valid, error = validate_code("import os\nos.system('ls')")
    assert is_valid is False
    assert error is not None


def test_generate_analysis_code():
    """Test code generation."""
    code = generate_analysis_code("Calculate Bollinger Bands", "AAPL")
    
    assert "AAPL" in code
    assert "yfinance" in code or "plotly" in code.lower() or "pandas" in code.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
