"""Response builder service."""
import json
from typing import Optional, List, Dict, Any
from datetime import datetime


def build_text_response(
    answer: str,
    tickers: Optional[List[str]] = None,
    charts: Optional[List[Dict]] = None,
    tables: Optional[List[Dict]] = None,
    warnings: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a structured text response.
    
    Args:
        answer: Natural language answer
        tickers: List of tickers mentioned
        charts: List of chart objects
        tables: List of data tables
        warnings: List of warning messages
        metadata: Additional metadata
        summary: Brief summary of the analysis
    
    Returns:
        Structured response dict
    """
    return {
        "answer": answer,
        "summary": summary,
        "tickers": tickers or [],
        "charts": charts or [],
        "tables": tables or [],
        "warnings": warnings or [],
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }


def build_chart_response(
    chart_type: str,
    html_path: str,
    figure_json: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a chart response.
    
    Args:
        chart_type: Type of chart
        html_path: Path to saved HTML file
        figure_json: Optional Plotly JSON
        tickers: List of tickers
        title: Chart title
    
    Returns:
        Chart response dict
    """
    return {
        "type": chart_type,
        "html_path": html_path,
        "figure_json": figure_json,
        "tickers": tickers or [],
        "title": title,
        "created_at": datetime.now().isoformat()
    }


def build_error_response(
    error_code: str,
    message: str,
    suggestion: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build an error response.
    
    Args:
        error_code: Error code (INVALID_TICKER, NO_DATA, etc.)
        message: Error message
        suggestion: Optional suggestion for user
    
    Returns:
        Error response dict
    """
    return {
        "status": "error",
        "error_code": error_code,
        "message": message,
        "suggestion": suggestion,
        "timestamp": datetime.now().isoformat()
    }


def build_table_response(
    name: str,
    headers: List[str],
    rows: List[List[Any]],
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a table response.
    
    Args:
        name: Table name
        headers: Column headers
        rows: Data rows
        description: Optional description
    
    Returns:
        Table response dict
    """
    return {
        "name": name,
        "headers": headers,
        "rows": rows,
        "description": description,
        "row_count": len(rows)
    }


def format_summary_stats(stats: Dict[str, Any], ticker: str) -> str:
    """
    Format summary statistics into readable text.
    
    Args:
        stats: Statistics dict
        ticker: Ticker symbol
    
    Returns:
        Formatted text
    """
    if "error" in stats:
        return f"Unable to compute statistics for {ticker}: {stats['error']}"
    
    lines = [f"**{ticker} Summary Statistics**"]
    
    if stats.get("current_price"):
        lines.append(f"- **Current Price**: ${stats['current_price']:.2f}")
    
    if stats.get("cumulative_return_pct") is not None:
        ret = stats['cumulative_return_pct']
        sign = "+" if ret >= 0 else ""
        lines.append(f"- **Period Return**: {sign}{ret:.2f}%")
    
    if stats.get("annualized_volatility_pct") is not None:
        lines.append(f"- **Annualized Volatility**: {stats['annualized_volatility_pct']:.2f}%")
    
    if stats.get("max_drawdown_pct") is not None:
        lines.append(f"- **Max Drawdown**: {stats['max_drawdown_pct']:.2f}%")
    
    if stats.get("positive_days_pct") is not None:
        lines.append(f"- **Positive Days**: {stats['positive_days_pct']:.2f}%")
    
    return "\n".join(lines)


def format_company_info(info: Dict[str, Any], ticker: str) -> str:
    """
    Format company information into readable text.
    
    Args:
        info: Company info dict
        ticker: Ticker symbol
    
    Returns:
        Formatted text
    """
    if not info or info.get("status") == "NO_DATA":
        return f"No company information available for {ticker}"
    
    data = info.get("data", info)
    lines = [f"**{data.get('shortName', ticker)}**"]
    
    if data.get("sector"):
        lines.append(f"- **Sector**: {data['sector']}")
    if data.get("industry"):
        lines.append(f"- **Industry**: {data['industry']}")
    
    if data.get("marketCap"):
        market_cap = data['marketCap']
        if market_cap >= 1e12:
            lines.append(f"- **Market Cap**: ${market_cap/1e12:.2f}T")
        elif market_cap >= 1e9:
            lines.append(f"- **Market Cap**: ${market_cap/1e9:.2f}B")
    
    if data.get("currentPrice"):
        lines.append(f"- **Current Price**: ${data['currentPrice']:.2f}")
    
    if data.get("trailingPE"):
        lines.append(f"- **P/E Ratio (TTM)**: {data['trailingPE']:.2f}")
    
    if data.get("dividendYield"):
        lines.append(f"- **Dividend Yield**: {data['dividendYield']*100:.2f}%")
    
    if data.get("beta"):
        lines.append(f"- **Beta**: {data['beta']:.2f}")
    
    return "\n".join(lines)


def format_comparison(comparisons: List[Dict[str, Any]], metric: str) -> str:
    """
    Format a comparison table.
    
    Args:
        comparisons: List of comparison dicts
        metric: Metric being compared
    
    Returns:
        Formatted text
    """
    lines = [f"**{metric} Comparison**"]
    lines.append("")
    
    header = "| Ticker | Value |"
    separator = "|------|-------|"
    lines.append(header)
    lines.append(separator)
    
    for item in comparisons:
        ticker = item.get("ticker", "N/A")
        value = item.get("value", "N/A")
        if isinstance(value, float):
            lines.append(f"| {ticker} | {value:.2f} |")
        else:
            lines.append(f"| {ticker} | {value} |")
    
    return "\n".join(lines)


def add_disclaimer(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add disclaimer to response.
    
    Args:
        response: Response dict
    
    Returns:
        Response with disclaimer
    """
    disclaimer = "Yahoo Finance data may be delayed and is not suitable for real-time trading decisions."
    
    if "warnings" not in response:
        response["warnings"] = []
    
    if disclaimer not in response["warnings"]:
        response["warnings"].append(disclaimer)
    
    response["disclaimer"] = disclaimer
    
    return response


def add_investment_warning(response: Dict[str, Any], context: str) -> Dict[str, Any]:
    """
    Add investment advice warning.
    
    Args:
        response: Response dict
        context: Context of the question
    
    Returns:
        Response with warning
    """
    warning = "This is not personalized investment advice. Please consult a financial advisor before making investment decisions."
    
    if "warnings" not in response:
        response["warnings"] = []
    
    if warning not in response["warnings"]:
        response["warnings"].append(warning)
    
    return response


def generate_summary(
    intent: str,
    tickers: List[str],
    period: str,
    stats: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a brief summary of the analysis.
    
    Args:
        intent: Type of analysis performed
        tickers: List of tickers analyzed
        period: Time period analyzed
        stats: Optional statistics
    
    Returns:
        Brief summary string
    """
    ticker_str = ", ".join(tickers) if tickers else "unknown"
    
    intent_summaries = {
        "quote_lookup": f"Retrieved current price for {ticker_str}",
        "historical_performance": f"Historical performance analysis for {ticker_str} over {period}",
        "technical_analysis": f"Technical analysis for {ticker_str} over {period}",
        "cross_ticker_comparison": f"Comparison of {ticker_str} over {period}",
        "company_fundamentals": f"Company fundamentals for {ticker_str}",
        "visualization_request": f"Visualization generated for {ticker_str}",
        "general_qa": f"Analysis for {ticker_str}"
    }
    
    summary = intent_summaries.get(intent, f"Analysis for {ticker_str}")
    
    if stats:
        if stats.get("cumulative_return_pct") is not None:
            ret = stats["cumulative_return_pct"]
            direction = "up" if ret >= 0 else "down"
            summary += f" ({ret:+.1f}% {direction})"
        
        if stats.get("current_price") is not None:
            summary += f" at ${stats['current_price']:.2f}"
    
    return summary
