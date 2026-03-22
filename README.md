# Stock Analysis Agent

Local-first stock/ETF/crypto analysis agent with multi-provider LLM support, dynamic tool planning, human-in-the-loop approval, and interactive Plotly visualizations.

## What It Does

- Natural-language finance Q&A (quotes, historical performance, technical analysis, comparisons)
- Dynamic planning with LLM-first tool plan and safe fallback routing
- Human-in-the-loop approval gate before execution (optional)
- Local execution of data retrieval + analytics + chart generation
- AI-generated textual summary shown above charts
- Execution trace (planner source, routed request, tools executed)

## Key Features

- **LLM Providers**: Ollama (local), OpenAI, Anthropic, Gemini, DeepSeek
- **Runtime Provider Config**: Enter API keys directly in Streamlit UI (no `.env` required for interactive use)
- **Human-in-the-Loop**: Review plan, tools, and risks, then approve execution
- **Execution State Banner**: `Planning -> Planned -> Approved -> Running -> Completed`
- **Data Resilience**:
  - retry with exponential backoff
  - HTTP fallback for yfinance rate-limit cases
  - per-ticker fallback for multi-ticker comparisons
- **Charting**:
  - line, candlestick, normalized compare, volatility, drawdown, RSI, MACD, heatmap, histogram
  - normalized comparison now supports multi-line plots for multi-ticker comparisons
- **Market Coverage**: US, HK (`xxxx.HK`), ETFs (e.g., SPY/GLD/TLT/IEF/SHY), crypto pairs (e.g., BTC-USD)

## Architecture

```text
Streamlit UI -> FastAPI -> StockAgent Orchestrator -> Tools (data/indicators/charts)
                               |
                               -> LLM Gateway (Ollama/OpenAI/Anthropic/Gemini/DeepSeek)
```

## Prerequisites

1. Python 3.11+
2. Optional local model runtime (Ollama)

### Optional: Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-r1:7b
ollama serve
```

## Installation

```bash
cd local_stock_agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

### 1) Start API

```bash
uvicorn app.main:app --reload --port 8000
```

### 2) Start UI

```bash
streamlit run app/ui/streamlit_app.py --server.port 8501
```

Open: http://localhost:8501

## Human-in-the-Loop Flow

When `Human-in-the-loop approval` is enabled in sidebar:

1. Agent plans request (intent/tickers/period/chart/tools)
2. UI shows plan details + risk notes
3. User clicks `Approve and Execute`
4. Agent executes local tools/scripts and returns results

## API Overview

### `POST /api/chat`

Request fields (core):

```json
{
  "message": "Compare AAPL vs MSFT over 1 year",
  "session_id": "optional",
  "provider": "ollama",
  "model": "deepseek-r1:7b",
  "provider_config": {
    "provider": "ollama",
    "model": "deepseek-r1:7b",
    "base_url": "http://localhost:11434",
    "api_key": "optional-for-cloud"
  },
  "require_approval": true,
  "approved": false
}
```

Response includes:

- `summary`: AI textual summary
- `charts`: chart payloads (`figure_json`, `html_path`)
- `tables`: structured tables
- `metadata.execution_trace`:
  - planner source (`llm` or `rule_based`)
  - routed request
  - tools executed

### Provider endpoints

- `GET /api/providers/`
- `GET /api/providers/{provider}/models`
- `GET /api/providers/{provider}/health`
- `POST /api/providers/set`
- `GET /api/providers/{provider}/models/{model}/capabilities`

### Health endpoints

- `GET /api/health`
- `GET /api/health/providers`

## Example Queries

- `Show me AAPL price`
- `Compare AAPL vs MSFT over 1 year`
- `Please show me the price movement of HK electric (2638.hk) for last 3 years with 50MA, 200MA`
- `Analyze 0066.hk price movement for last 5 years`
- `Compare bond ETFs TLT, IEF, SHY over 5 years`
- `Analyze GLD vs SPY max`
- `Plot NVDA candlestick with RSI`

## Supported Instruments & Symbols

- **Stocks/ETFs**: standard US symbols (AAPL, MSFT, SPY, GLD, TLT)
- **HKEX**: `xxxx.HK` (e.g., `0700.HK`, `2638.HK`, `0066.HK`)
- **Crypto pairs**: e.g., `BTC-USD`
- Alias mapping for common company names is included and extensible

## Development

### Run tests

```bash
pytest tests/ -v
```

### Project structure

```text
app/
  api/          FastAPI routes
  agent/        orchestrator, router, prompts, schemas, guardrails
  tools/        yfinance, indicators, charts, code runner, cache
  services/     provider gateway/adapters, ticker mapper, response builder
  ui/           streamlit app
tests/
outputs/
```

## Notes & Limitations

- Yahoo Finance availability/rate limits can affect responses
- Data may be delayed
- Historical comparison uses overlapping available ranges
- This is an analysis assistant, not an execution/trading system

## Security

- Guardrails for unsafe content and blocked requests
- Safe code-generation path with restricted execution policy
- No brokerage/trade execution actions

## Disclaimer

This tool is for informational purposes only. It is not investment advice.

## License

MIT
