"""Chart generation tool using Plotly."""
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config import config


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names."""
    result_df = df.copy()
    col_mapping = {}
    for col in result_df.columns:
        lower = str(col).lower()
        if lower == 'open' and 'Open' not in result_df.columns:
            col_mapping[col] = 'Open'
        elif lower == 'high' and 'High' not in result_df.columns:
            col_mapping[col] = 'High'
        elif lower == 'low' and 'Low' not in result_df.columns:
            col_mapping[col] = 'Low'
        elif lower == 'close' and 'Close' not in result_df.columns:
            col_mapping[col] = 'Close'
        elif lower == 'volume' and 'Volume' not in result_df.columns:
            col_mapping[col] = 'Volume'
        elif lower == 'date' and 'Date' not in result_df.columns:
            col_mapping[col] = 'Date'
        elif lower == 'datetime' and 'Datetime' not in result_df.columns:
            col_mapping[col] = 'Datetime'
    if col_mapping:
        result_df = result_df.rename(columns=col_mapping)
    return result_df


def _get_date_index(df: pd.DataFrame) -> pd.Series:
    """Get date index from DataFrame."""
    if 'Datetime' in df.columns:
        return df['Datetime']
    elif 'Date' in df.columns:
        return df['Date']
    elif isinstance(df.index, pd.DatetimeIndex):
        return df.index
    else:
        return pd.Series(range(len(df)), name='index')


def build_chart(
    df: Union[pd.DataFrame, dict],
    chart_type: str,
    tickers: Optional[List[str]] = None,
    title: Optional[str] = None,
    save: bool = True
) -> dict:
    """
    Build a chart based on chart type.
    
    Args:
        df: DataFrame with price data or dict from yfinance tool
        chart_type: Type of chart to build
        tickers: List of ticker symbols
        title: Custom chart title
        save: Whether to save the chart to file
    
    Returns:
        dict with chart data, HTML path, and metadata
    """
    multi_series_map = None
    if isinstance(df, dict):
        if chart_type.lower().strip() in ["normalized_compare", "normalized"] and tickers:
            candidate = {}
            for t in tickers:
                ticker_data = df.get(t)
                if isinstance(ticker_data, pd.DataFrame):
                    local_df = _normalize_columns(ticker_data)
                    if 'Close' in local_df.columns and not local_df.empty:
                        candidate[t] = local_df
                elif isinstance(ticker_data, dict) and 'records' in ticker_data:
                    local_df = _normalize_columns(pd.DataFrame(ticker_data['records']))
                    if 'Close' in local_df.columns and not local_df.empty:
                        candidate[t] = local_df
            if len(candidate) >= 2:
                multi_series_map = candidate
                first = next(iter(candidate.keys()))
                df = candidate[first]
            elif 'data' in df and isinstance(df['data'], list):
                df = pd.DataFrame(df['data'])
            elif 'records' in df:
                df = pd.DataFrame(df['records'])
            else:
                return {"status": "ERROR", "message": "Invalid data format"}
        elif 'data' in df and isinstance(df['data'], list):
            df = pd.DataFrame(df['data'])
        elif 'records' in df:
            df = pd.DataFrame(df['records'])
        else:
            return {"status": "ERROR", "message": "Invalid data format"}
    
    df = _normalize_columns(df)
    
    if 'Close' not in df.columns:
        return {"status": "ERROR", "message": "No Close price column found"}
    
    chart_type = chart_type.lower().strip()
    tickers = tickers or ["Unknown"]
    
    try:
        if chart_type == "price_line" or chart_type == "line":
            fig = _build_price_line(df, tickers, title)
        elif chart_type == "candlestick" or chart_type == "candlestick_volume":
            fig = _build_candlestick(df, tickers, title)
        elif chart_type == "normalized_compare" or chart_type == "normalized":
            fig = _build_normalized(df, tickers, title, multi_series_map)
        elif chart_type == "rolling_volatility" or chart_type == "volatility":
            fig = _build_volatility(df, tickers, title)
        elif chart_type == "drawdown":
            fig = _build_drawdown(df, tickers, title)
        elif chart_type == "rsi_panel" or chart_type == "rsi":
            fig = _build_rsi(df, tickers, title)
        elif chart_type == "macd_panel" or chart_type == "macd":
            fig = _build_macd(df, tickers, title)
        elif chart_type == "correlation_heatmap" or chart_type == "heatmap":
            fig = _build_heatmap(df, tickers, title)
        elif chart_type == "returns_histogram" or chart_type == "histogram":
            fig = _build_returns_histogram(df, tickers, title)
        else:
            fig = _build_price_line(df, tickers, title)
        
        chart_id = hashlib.md5(f"{chart_type}_{tickers}_{pd.Timestamp.now()}".encode()).hexdigest()[:8]
        
        result = {
            "status": "success",
            "chart_id": chart_id,
            "chart_type": chart_type,
            "tickers": tickers,
            "figure_json": fig.to_json()
        }
        
        if save:
            output_path = config.OUTPUTS_DIR / f"chart_{chart_id}.html"
            fig.write_html(str(output_path))
            result["html_path"] = str(output_path)
        
        return result
        
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


def _build_price_line(df: pd.DataFrame, tickers: List[str], title: Optional[str]) -> go.Figure:
    """Build a simple price line chart."""
    dates = _get_date_index(df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=df['Close'],
        mode='lines',
        name=f"{tickers[0]} Close" if tickers else 'Close',
        line=dict(width=2)
    ))
    
    if title:
        fig.update_layout(title=title)
    else:
        fig.update_layout(
            title=f"{tickers[0] if tickers else 'Stock'} Price",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            hovermode='x unified'
        )
    
    return fig


def _build_candlestick(df: pd.DataFrame, tickers: List[str], title: Optional[str]) -> go.Figure:
    """Build a candlestick chart with volume."""
    dates = _get_date_index(df)
    
    has_ohlc = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close'])
    has_volume = 'Volume' in df.columns
    
    if has_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('', 'Volume')
        )
        
        if has_ohlc:
            fig.add_trace(
                go.Candlestick(
                    x=dates,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(x=dates, y=df['Close'], mode='lines', name='Close'),
                row=1, col=1
            )
        
        fig.add_trace(
            go.Bar(x=dates, y=df['Volume'], name='Volume', marker_color='rgba(100,100,100,0.5)'),
            row=2, col=1
        )
    else:
        fig = go.Figure()
        if has_ohlc:
            fig.add_trace(go.Candlestick(
                x=dates,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ))
        else:
            fig.add_trace(go.Scatter(x=dates, y=df['Close'], mode='lines', name='Close'))
    
    fig.update_layout(
        title=title or f"{tickers[0] if tickers else 'Stock'} Candlestick",
        template="plotly_dark",
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig


def _build_normalized(
    df: pd.DataFrame,
    tickers: List[str],
    title: Optional[str],
    multi_series_map: Optional[dict] = None
) -> go.Figure:
    """Build a normalized return comparison chart."""
    fig = go.Figure()

    if multi_series_map and len(multi_series_map) >= 2:
        for ticker, local_df in multi_series_map.items():
            dates = _get_date_index(local_df)
            close = local_df['Close']
            if len(close) == 0:
                continue
            normalized = (close / close.iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=dates,
                y=normalized,
                mode='lines',
                name=f"{ticker} (Base 100)",
                line=dict(width=2)
            ))
    else:
        dates = _get_date_index(df)
        close = df['Close']
        normalized = (close / close.iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=dates,
            y=normalized,
            mode='lines',
            name=f"{tickers[0] if tickers else 'Stock'} (Base 100)",
            fill='tozeroy',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title or f"{tickers[0] if tickers else 'Stock'} Normalized Performance",
        xaxis_title="Date",
        yaxis_title="Normalized Price (Base 100)",
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig


def _build_volatility(df: pd.DataFrame, tickers: List[str], title: Optional[str]) -> go.Figure:
    """Build a rolling volatility chart."""
    dates = _get_date_index(df)
    returns = df['Close'].pct_change()
    volatility = returns.rolling(window=20).std() * np.sqrt(252) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=volatility,
        mode='lines',
        name='Annualized Volatility',
        fill='tozeroy',
        line=dict(width=2, color='orange')
    ))
    
    fig.update_layout(
        title=title or f"{tickers[0] if tickers else 'Stock'} Rolling Volatility (20-day)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig


def _build_drawdown(df: pd.DataFrame, tickers: List[str], title: Optional[str]) -> go.Figure:
    """Build a drawdown chart."""
    dates = _get_date_index(df)
    close = df['Close']
    peak = close.cummax()
    drawdown = ((close - peak) / peak) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(width=2, color='red')
    ))
    
    fig.update_layout(
        title=title or f"{tickers[0] if tickers else 'Stock'} Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig


def _build_rsi(df: pd.DataFrame, tickers: List[str], title: Optional[str]) -> go.Figure:
    """Build RSI panel."""
    dates = _get_date_index(df)
    close = df['Close']
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=rsi,
        mode='lines',
        name='RSI(14)',
        line=dict(width=2, color='purple')
    ))
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig.update_layout(
        title=title or f"{tickers[0] if tickers else 'Stock'} RSI(14)",
        xaxis_title="Date",
        yaxis_title="RSI",
        template="plotly_dark",
        hovermode='x unified',
        yaxis_range=[0, 100]
    )
    
    return fig


def _build_macd(df: pd.DataFrame, tickers: List[str], title: Optional[str]) -> go.Figure:
    """Build MACD panel with histogram."""
    dates = _get_date_index(df)
    close = df['Close']
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.6, 0.4], subplot_titles=('MACD', 'Histogram'))
    
    fig.add_trace(go.Scatter(x=dates, y=macd, mode='lines', name='MACD', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=signal, mode='lines', name='Signal', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=histogram, name='Histogram', marker_color='gray'), row=2, col=1)
    
    fig.update_layout(
        title=title or f"{tickers[0] if tickers else 'Stock'} MACD",
        template="plotly_dark",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def _build_heatmap(data_dict: dict, tickers: List[str], title: Optional[str]) -> go.Figure:
    """Build correlation heatmap."""
    if not isinstance(data_dict, dict):
        return go.Figure()
    
    closes = {}
    for ticker, data in data_dict.items():
        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            closes[ticker] = data['Close']
        elif isinstance(data, dict) and 'records' in data:
            df = pd.DataFrame(data['records'])
            if 'Close' in df.columns:
                closes[ticker] = df['Close']
    
    if len(closes) < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough tickers for correlation")
        return fig
    
    combined = pd.DataFrame(closes).dropna()
    corr_matrix = combined.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.columns),
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title or "Correlation Heatmap",
        template="plotly_dark"
    )
    
    return fig


def _build_returns_histogram(df: pd.DataFrame, tickers: List[str], title: Optional[str]) -> go.Figure:
    """Build returns histogram."""
    dates = _get_date_index(df)
    returns = df['Close'].pct_change().dropna() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        name='Daily Returns',
        nbinsx=50,
        marker_color='steelblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=title or f"{tickers[0] if tickers else 'Stock'} Daily Returns Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        barmode='overlay'
    )
    
    return fig


def get_chart_catalog() -> dict:
    """Return available chart types and their descriptions."""
    return {
        "price_line": {
            "description": "Simple line chart of closing prices",
            "inputs": ["ticker", "date_range"],
            "example": "Show me AAPL price"
        },
        "candlestick_volume": {
            "description": "Candlestick chart with volume bars",
            "inputs": ["ticker", "date_range"],
            "example": "Show candlestick for TSLA"
        },
        "normalized_compare": {
            "description": "Base-100 normalized performance comparison",
            "inputs": ["tickers", "date_range"],
            "example": "Compare AAPL vs MSFT YTD"
        },
        "rolling_volatility": {
            "description": "20-day rolling annualized volatility",
            "inputs": ["ticker", "date_range"],
            "example": "Show NVDA volatility"
        },
        "drawdown": {
            "description": "Peak-to-trough drawdown chart",
            "inputs": ["ticker", "date_range"],
            "example": "Plot Tesla drawdown"
        },
        "rsi_panel": {
            "description": "RSI(14) indicator panel",
            "inputs": ["ticker", "date_range"],
            "example": "Show RSI for AAPL"
        },
        "macd_panel": {
            "description": "MACD with signal line and histogram",
            "inputs": ["ticker", "date_range"],
            "example": "Plot MACD for NVDA"
        },
        "correlation_heatmap": {
            "description": "Correlation heatmap for multiple tickers",
            "inputs": ["tickers"],
            "example": "Show correlation between tech stocks"
        },
        "returns_histogram": {
            "description": "Daily returns distribution histogram",
            "inputs": ["ticker", "date_range"],
            "example": "Show AAPL return distribution"
        }
    }
