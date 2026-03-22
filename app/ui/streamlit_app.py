"""Streamlit UI for Stock Analysis Agent."""
import streamlit as st
import requests
import json
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from pydantic import BaseModel

API_BASE = "http://localhost:8000/api"

PROVIDER_MODELS = {
    "ollama": {
        "name": "Ollama (Local)",
        "models": [
            "deepseek-r1:7b",
            "deepseek-r1:14b", 
            "llama3.1:8b",
            "llama3.2:3b",
            "qwen2.5:7b",
            "qwen2.5:14b",
            "mistral:7b",
            "codellama:7b",
        ],
        "requires_url": True,
        "url_placeholder": "http://localhost:11434"
    },
    "openai": {
        "name": "OpenAI",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
        ],
        "requires_key": True,
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        "requires_key": True,
    },
    "gemini": {
        "name": "Google Gemini",
        "models": [
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
        ],
        "requires_key": True,
    },
    "deepseek": {
        "name": "DeepSeek",
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "requires_key": True,
    },
}

st.set_page_config(
    page_title="Stock Analysis Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Analysis Agent")
st.markdown("*Multi-Provider AI-Powered Stock Analysis*")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "provider" not in st.session_state:
    st.session_state.provider = "ollama"
if "model" not in st.session_state:
    st.session_state.model = "deepseek-r1:7b"
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}
if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"
if "require_approval" not in st.session_state:
    st.session_state.require_approval = True
if "pending_payload" not in st.session_state:
    st.session_state.pending_payload = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "execution_state" not in st.session_state:
    st.session_state.execution_state = "Idle"
if "clear_query_input" not in st.session_state:
    st.session_state.clear_query_input = False

if st.session_state.clear_query_input:
    st.session_state.query_input = ""
    st.session_state.clear_query_input = False

if st.session_state.execution_state != "Idle":
    st.info(f"Execution State: {st.session_state.execution_state}")


def call_api(endpoint: str, payload: dict):
    """Call the API and return the response."""
    try:
        response = requests.post(f"{API_BASE}/{endpoint}", json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}", "detail": response.text}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Please ensure the server is running."}
    except Exception as e:
        return {"error": str(e)}


def check_health():
    """Check API health."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def render_analysis_result(result: dict):
    """Render analysis result payload safely."""
    st.markdown("---")
    st.markdown("### Analysis Results")

    if result.get("summary"):
        st.markdown("### AI Understanding Summary")
        st.success(result["summary"])

    if result.get("answer"):
        st.markdown(result["answer"])

    charts = result.get("charts", [])
    if isinstance(charts, list):
        for chart in charts:
            if isinstance(chart, BaseModel):
                chart = chart.model_dump()
            if not isinstance(chart, dict):
                continue
            if chart.get("figure_json"):
                try:
                    fig = go.Figure(json.loads(chart["figure_json"]))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    if chart.get("html_path"):
                        st.info(f"Chart saved: {chart['html_path']}")
            elif chart.get("html_path"):
                st.info(f"Chart available: {chart['html_path']}")

    tables = result.get("tables", [])
    if isinstance(tables, list):
        for table in tables:
            if isinstance(table, BaseModel):
                table = table.model_dump()
            if not isinstance(table, dict):
                continue
            st.markdown(f"**{table.get('name', 'Data')}**")
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            if isinstance(headers, list) and isinstance(rows, list):
                try:
                    if rows and all(isinstance(r, list) for r in rows):
                        use_columns = headers if (headers and len(headers) == len(rows[0])) else None
                        if use_columns:
                            df_table = pd.DataFrame(rows)
                            df_table.columns = [str(h) for h in use_columns]
                        else:
                            df_table = pd.DataFrame(rows)
                        st.dataframe(df_table, use_container_width=True)
                    else:
                        st.table(table)
                except Exception:
                    st.table(table)
            else:
                st.table(table)

    warnings = result.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            st.warning(warning)

    metadata = result.get("metadata", {}) if isinstance(result.get("metadata", {}), dict) else {}
    trace = metadata.get("execution_trace") if isinstance(metadata.get("execution_trace"), dict) else None
    if trace:
        with st.expander("Execution Trace", expanded=False):
            st.markdown(f"- Planner source: `{trace.get('planner_source', 'unknown')}`")
            st.markdown(f"- LLM plan used: `{trace.get('llm_plan_used', False)}`")
            routed = trace.get("routed_request", {}) if isinstance(trace.get("routed_request", {}), dict) else {}
            if routed:
                st.markdown("- Routed request:")
                st.json(routed)
            tools = trace.get("tools_executed", [])
            if isinstance(tools, list) and tools:
                st.markdown("- Tools executed:")
                st.code("\n".join([str(t) for t in tools]), language="text")

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"⏱️ Runtime: {result.get('runtime_ms', 'N/A')}ms")
    with col2:
        st.caption(f"📅 {result.get('timestamp', '')}")


def render_approval_panel(result: dict):
    """Render approval plan and approval controls."""
    plan = result.get("execution_plan")
    if not (result.get("approval_required") and isinstance(plan, dict)):
        return

    st.session_state.execution_state = "Planned"
    st.markdown("---")
    st.markdown("### Human Approval Required")
    st.markdown("#### Planned Work")
    st.write(plan.get("explanation", ""))
    st.markdown(f"- Intent: `{plan.get('intent', 'unknown')}`")
    st.markdown(f"- Tickers: `{', '.join(plan.get('tickers', []))}`")
    st.markdown(f"- Period/Interval: `{plan.get('period', '')}` / `{plan.get('interval', '')}`")
    st.markdown(f"- Chart: `{plan.get('chart_type', 'auto')}`")
    if plan.get("indicators"):
        st.markdown(f"- Indicators: `{', '.join(plan['indicators'])}`")
    st.markdown("#### Tools To Execute")
    st.code("\n".join(plan.get("tools_to_run", [])), language="text")
    if plan.get("risk_notes"):
        st.markdown("#### Risk / Data Notes")
        for note in plan["risk_notes"]:
            st.warning(note)

    approve_col, cancel_col = st.columns(2)
    with approve_col:
        if st.button("✅ Approve and Execute", type="primary", key="approve_execute_btn"):
            if st.session_state.pending_payload:
                approved_payload = dict(st.session_state.pending_payload)
                approved_payload["approved"] = True
                st.session_state.execution_state = "Approved"
                with st.spinner("Executing approved plan..."):
                    st.session_state.execution_state = "Running"
                    approved_result = call_api("chat/", approved_payload)
                st.session_state.last_result = approved_result
                st.session_state.messages.append({"role": "assistant", "content": approved_result})
                st.session_state.pending_payload = None
                if isinstance(approved_result, dict) and "error" not in approved_result:
                    st.session_state.execution_state = "Completed"
                else:
                    st.session_state.execution_state = "Error"
                st.rerun()
    with cancel_col:
        if st.button("❌ Cancel", key="approve_cancel_btn"):
            st.session_state.pending_payload = None
            st.session_state.execution_state = "Cancelled"
            st.info("Execution cancelled by user.")


with st.sidebar:
    st.header("🤖 AI Provider")
    
    selected_provider = st.selectbox(
        "Provider", 
        list(PROVIDER_MODELS.keys()), 
        format_func=lambda x: PROVIDER_MODELS[x]["name"],
        index=0
    )
    st.session_state.provider = selected_provider
    
    provider_info = PROVIDER_MODELS[selected_provider]
    
    if provider_info.get("requires_url", False):
        ollama_url = st.text_input(
            "Ollama URL",
            value=st.session_state.ollama_url,
            placeholder=provider_info.get("url_placeholder", "http://localhost:11434")
        )
        st.session_state.ollama_url = ollama_url
    
    if provider_info.get("requires_key", False):
        api_key = st.text_input(
            f"{provider_info['name']} API Key",
            value=st.session_state.api_keys.get(selected_provider, ""),
            placeholder=f"Enter your API key",
            type="password"
        )
        st.session_state.api_keys[selected_provider] = api_key
        
        if not api_key:
            st.warning(f"⚠️ Enter your API key to use {provider_info['name']}")
    
    models = provider_info["models"]
    if models:
        selected_model = st.selectbox("Model", models, index=0)
        st.session_state.model = selected_model
    else:
        st.warning("No models available")
        st.session_state.model = None
    
    with st.expander("⚙️ Advanced Settings"):
        reasoning = st.toggle("Reasoning Mode", value=True)
        streaming = st.toggle("Streaming", value=False)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
        max_tokens = st.number_input("Max Tokens", 512, 8192, 2048)
        st.session_state.require_approval = st.toggle("Human-in-the-loop approval", value=st.session_state.require_approval)
    
    st.divider()
    
    st.header("System Status")
    health = check_health()
    if health:
        st.success("✅ API Online")
        st.write(f"- Ollama: {health.get('ollama', 'Unknown')}")
        st.write(f"- yfinance: {health.get('yfinance', 'Unknown')}")
    else:
        st.error("❌ API Offline")
        st.write("Start: `uvicorn app.main:app --reload`")

    st.divider()
    
    st.header("Example Queries")
    examples = [
        "Show me AAPL price",
        "Compare AAPL vs MSFT over 1 year",
        "Plot NVDA candlestick with RSI",
        "Analyze HK electric (2638.HK) for last 3 years with 50MA and 200MA",
        "Compare bond ETFs TLT, IEF, SHY over 5 years",
        "Analyze GLD vs BTC-USD volatility over 2 years",
        "What's TSLA's max drawdown?",
        "Show me NVDA vs TSLA volatility",
        "Compare META, AMZN, and GOOGL YTD",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state.query_input = ex

    st.divider()
    
    st.header("Chart Types")
    st.write("""
- Line Chart
- Candlestick
- Normalized Comparison
- Rolling Volatility
- Drawdown
- RSI Panel
- MACD Panel
- Correlation Heatmap
- Returns Histogram
    """)

st.sidebar.header("About")
st.sidebar.info("""
This agent supports multiple LLM providers:
- Ollama (local models)
- OpenAI (GPT models)
- Anthropic (Claude models)
- Google Gemini
- DeepSeek

Data provided by Yahoo Finance.
Charts rendered with Plotly.
""")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Ask about stocks:",
        placeholder="e.g., 'Show me AAPL vs MSFT 6-month return'",
        key="query_input"
    )
with col2:
    st.write("")  
    st.write("")
    analyze_button = st.button("🔍 Analyze", type="primary")

if analyze_button:
    if query:
        provider_config = {
            "provider": st.session_state.provider,
            "model": st.session_state.model,
        }
        
        if st.session_state.provider == "ollama":
            provider_config["base_url"] = st.session_state.ollama_url
        elif st.session_state.provider in st.session_state.api_keys:
            provider_config["api_key"] = st.session_state.api_keys[st.session_state.provider]
        
        with st.spinner("Analyzing..."):
            st.session_state.execution_state = "Planning"
            payload = {
                "message": query,
                "session_id": st.session_state.session_id,
                "provider_config": provider_config,
                "require_approval": st.session_state.require_approval,
                "approved": False
            }
            result = call_api("chat/", payload)
            st.session_state.last_result = result
        
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.session_state.clear_query_input = True
        
        if isinstance(result, dict) and "error" in result:
            st.session_state.execution_state = "Error"
            st.error(result["error"])
        else:
            if not isinstance(result, dict):
                st.session_state.execution_state = "Error"
                st.error("Unexpected response format.")
            else:
                plan = result.get("execution_plan")
                if result.get("approval_required") and isinstance(plan, dict):
                    st.session_state.pending_payload = payload
                    render_approval_panel(result)
                    st.stop()

                render_analysis_result(result)
                st.session_state.execution_state = "Completed"
    else:
        st.warning("Please enter a query before clicking Analyze.")

# Render latest result when no new query is submitted (e.g., after approval rerun)
if isinstance(st.session_state.last_result, dict):
    result = st.session_state.last_result
    if result and result.get("approval_required"):
        render_approval_panel(result)
    elif result:
        render_analysis_result(result)

with st.expander("💬 Chat History", expanded=False):
    for msg in st.session_state.messages:
        if isinstance(msg["content"], dict):
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']['content']}")
            else:
                st.markdown(f"**Agent:** {msg['content'].get('answer', str(msg['content']))[:500]}")
        else:
            st.markdown(f"**{msg['role'].title()}:** {msg['content']}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
⚠️ This tool is for informational purposes only. Yahoo Finance data may be delayed.
Not suitable for real-time trading decisions. Not investment advice.
</div>
""", unsafe_allow_html=True)
