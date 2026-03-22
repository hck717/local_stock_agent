"""Services package."""
from app.services.ollama_client import OllamaClient, ollama_client
from app.services.ticker_mapper import (
    normalize_ticker, extract_tickers, parse_time_period,
    detect_market, validate_ticker, suggest_ticker_format
)
from app.services.response_builder import (
    build_text_response, build_chart_response, build_error_response,
    build_table_response, format_summary_stats, format_company_info,
    format_comparison, add_disclaimer, add_investment_warning
)

__all__ = [
    "OllamaClient",
    "ollama_client",
    "normalize_ticker",
    "extract_tickers",
    "parse_time_period",
    "detect_market",
    "validate_ticker",
    "suggest_ticker_format",
    "build_text_response",
    "build_chart_response",
    "build_error_response",
    "build_table_response",
    "format_summary_stats",
    "format_company_info",
    "format_comparison",
    "add_disclaimer",
    "add_investment_warning",
]
