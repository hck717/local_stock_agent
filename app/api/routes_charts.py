"""Charts API routes."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional

from app.tools import build_chart, get_price_history, compute_indicators
from app.tools.chart_tool import get_chart_catalog
from pydantic import BaseModel
import pandas as pd

router = APIRouter(prefix="/charts", tags=["Charts"])


class ChartRequest(BaseModel):
    ticker: str
    chart_type: str
    period: str = "6mo"
    interval: str = "1d"
    indicators: Optional[List[str]] = None


class ChartResponse(BaseModel):
    chart_id: str
    chart_type: str
    html_path: str


@router.post("/generate", response_model=ChartResponse)
async def generate_chart(request: ChartRequest):
    """
    Generate a chart for a ticker.
    
    Available chart types:
    - price_line: Simple line chart
    - candlestick: Candlestick with volume
    - normalized_compare: Base-100 comparison
    - rolling_volatility: 20-day volatility
    - drawdown: Peak-to-trough
    - rsi_panel: RSI indicator
    - macd_panel: MACD with histogram
    - correlation_heatmap: Multi-ticker correlation
    - returns_histogram: Returns distribution
    """
    try:
        result = get_price_history(
            ticker=request.ticker,
            period=request.period,
            interval=request.interval
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to fetch data"))
        
        df = pd.DataFrame(result["data"])
        
        if request.indicators:
            df = compute_indicators(df, request.indicators)
        
        chart_result = build_chart(
            df=df,
            chart_type=request.chart_type,
            tickers=[request.ticker]
        )
        
        if chart_result.get("status") != "success":
            raise HTTPException(status_code=400, detail=chart_result.get("message", "Failed to build chart"))
        
        return ChartResponse(
            chart_id=chart_result["chart_id"],
            chart_type=chart_result["chart_type"],
            html_path=chart_result["html_path"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{chart_id}")
async def get_chart(chart_id: str):
    """
    Retrieve a generated chart by ID.
    """
    from app.config import config
    
    chart_path = config.OUTPUTS_DIR / f"chart_{chart_id}.html"
    
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    
    return FileResponse(
        path=chart_path,
        media_type="text/html",
        filename=f"chart_{chart_id}.html"
    )


@router.get("/catalog")
async def list_chart_types():
    """
    Get the catalog of available chart types.
    """
    return get_chart_catalog()
