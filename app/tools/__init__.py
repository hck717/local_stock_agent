"""Tools package."""
from app.tools.yfinance_tool import get_price_history, get_multi_ticker_history, get_company_info, get_live_quote
from app.tools.indicators_tool import compute_indicators, compute_summary_stats, compute_correlation_matrix
from app.tools.chart_tool import build_chart, get_chart_catalog
from app.tools.code_runner import run_safe_analysis, generate_analysis_code, validate_code
from app.tools.cache_tool import (
    get_cached, set_cached, invalidate_cache, clear_all_cache, get_cache_stats,
    cache_price_data, get_cached_price_data,
    cache_company_info, get_cached_company_info,
    cache_chart, get_cached_chart
)

__all__ = [
    "get_price_history",
    "get_multi_ticker_history",
    "get_company_info",
    "get_live_quote",
    "compute_indicators",
    "compute_summary_stats",
    "compute_correlation_matrix",
    "build_chart",
    "get_chart_catalog",
    "run_safe_analysis",
    "generate_analysis_code",
    "validate_code",
    "get_cached",
    "set_cached",
    "invalidate_cache",
    "clear_all_cache",
    "get_cache_stats",
    "cache_price_data",
    "get_cached_price_data",
    "cache_company_info",
    "get_cached_company_info",
    "cache_chart",
    "get_cached_chart",
]
