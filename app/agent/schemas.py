"""Agent data schemas and models."""
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class Intent(str, Enum):
    QUOTE_LOOKUP = "quote_lookup"
    HISTORICAL_PERFORMANCE = "historical_performance"
    TECHNICAL_ANALYSIS = "technical_analysis"
    CROSS_TICKER_COMPARISON = "cross_ticker_comparison"
    COMPANY_FUNDAMENTALS = "company_fundamentals"
    NEWS_CONTEXT = "news_context"
    VISUALIZATION_REQUEST = "visualization_request"
    CODING_REQUEST = "coding_request"
    GENERAL_QA = "general_qa"


class ChartType(str, Enum):
    PRICE_LINE = "price_line"
    CANDLESTICK = "candlestick"
    NORMALIZED_COMPARE = "normalized_compare"
    ROLLING_VOLATILITY = "rolling_volatility"
    DRAWDOWN = "drawdown"
    RSI_PANEL = "rsi_panel"
    MACD_PANEL = "macd_panel"
    CORRELATION_HEATMAP = "correlation_heatmap"
    RETURNS_HISTOGRAM = "returns_histogram"


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    intent: Intent
    tickers: List[str] = Field(default_factory=list)
    period: str = "6mo"
    interval: str = "1d"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    tools: List[ToolCall] = Field(default_factory=list)
    response_style: str = "concise"
    chart_type: Optional[ChartType] = None
    indicators: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    ticker_override: Optional[str] = None
    market: Optional[str] = "US"
    chart_preferences: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    provider_config: Optional[Dict[str, Any]] = None
    require_approval: bool = False
    approved: bool = False


class ChartResponse(BaseModel):
    chart_id: str
    chart_type: str
    html_path: Optional[str] = None
    figure_json: Optional[str] = None
    tickers: List[str] = Field(default_factory=list)


class TableData(BaseModel):
    name: str
    headers: List[str]
    rows: List[List[Any]]
    description: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    summary: Optional[str] = None
    approval_required: bool = False
    execution_plan: Optional[Dict[str, Any]] = None
    tickers: List[str] = Field(default_factory=list)
    charts: List[ChartResponse] = Field(default_factory=list)
    tables: List[TableData] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CodeRequest(BaseModel):
    message: str
    execute: bool = True


class CodeResponse(BaseModel):
    generated_code: str
    execution_status: str
    artifacts: List[str] = Field(default_factory=list)
    summary: str


class HealthResponse(BaseModel):
    api: str
    ollama: str
    yfinance: str


class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    message: str
    suggestion: Optional[str] = None
