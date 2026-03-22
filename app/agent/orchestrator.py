"""Agent orchestrator - main coordination logic."""
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

from app.agent.schemas import Intent, ChartType, ChartResponse, TableData, ToolPlan
from app.agent.router import route_request, needs_clarification
from app.agent.guardrails import apply_guardrails, Guardrails
from app.agent.prompts import (
    TOOL_PLAN_PROMPT,
    ANSWER_SUMMARY_PROMPT,
    ANSWER_WRITER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    DATABASE_WARNING
)
from app.services.providers import gateway, Message, GenerationSettings
from app.services.ticker_mapper import normalize_ticker, suggest_ticker_format
from app.services.response_builder import (
    build_text_response, build_error_response, build_table_response,
    format_summary_stats, add_disclaimer, generate_summary
)
from app.tools import (
    get_price_history, get_multi_ticker_history, get_company_info,
    compute_indicators, compute_summary_stats, compute_correlation_matrix,
    build_chart, run_safe_analysis, generate_analysis_code
)
from app.tools.cache_tool import cache_price_data, get_cached_price_data


class StockAgent:
    """Main orchestrator for the stock analysis agent."""
    
    def __init__(self):
        gateway.initialize_providers()
        gateway.set_default_provider()
    
    def generate_with_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        json_mode: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using the LLM gateway."""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]
        
        settings = GenerationSettings(
            temperature=0.3,
            json_mode=json_mode,
            stream=False
        )
        
        result = gateway.generate(
            messages=messages,
            provider_type=provider,
            model=model,
            settings=settings,
            api_key=api_key,
            base_url=base_url
        )
        
        return result
    
    def generate_analysis(
        self,
        question: str,
        data: Dict[str, Any],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> str:
        """Generate natural language analysis using the LLM."""
        system_prompt = ANSWER_WRITER_SYSTEM_PROMPT
        user_prompt = f"""Question: {question}

Data Summary:
{json.dumps(data, indent=2, default=str)[:3000]}

Provide a clear, concise answer that:
1. Directly answers the question
2. Highlights key findings and metrics
3. Mentions any data limitations"""
        
        result = self.generate_with_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
            json_mode=False,
            api_key=api_key,
            base_url=base_url
        )
        
        if result.get("status") == "success":
            return result.get("content", "")
        return ""

    def _plan_request_with_llm(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Use the LLM to build a dynamic tool plan."""
        result = self.generate_with_llm(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=TOOL_PLAN_PROMPT.format(question=message),
            provider=provider,
            model=model,
            json_mode=True,
            api_key=api_key,
            base_url=base_url
        )

        if result.get("status") != "success":
            return None

        content = result.get("content", "")
        if not content:
            return None

        try:
            plan_data = content if isinstance(content, dict) else json.loads(content)
            plan = ToolPlan(**plan_data)
            return {
                "intent": plan.intent,
                "tickers": plan.tickers,
                "period": plan.period,
                "interval": plan.interval,
                "start_date": plan.start_date,
                "end_date": plan.end_date,
                "chart_type": plan.chart_type,
                "indicators": plan.indicators,
                "needs_chart": plan.chart_type is not None or len(plan.tickers) > 1
            }
        except Exception:
            return None

    def _fallback_plan_for_local_model(self, message: str) -> Optional[Dict[str, Any]]:
        """Use deterministic routing as fallback for weaker local planning quality."""
        try:
            return route_request(message)
        except Exception:
            return None

    def _serialize_route_result(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize route result for response metadata."""
        serialized = dict(route_result)
        intent = serialized.get("intent")
        chart_type = serialized.get("chart_type")
        if intent is not None and isinstance(intent, Intent):
            serialized["intent"] = intent.value
        if chart_type is not None and isinstance(chart_type, ChartType):
            serialized["chart_type"] = chart_type.value
        return serialized

    def _tools_for_intent(self, intent: Any, indicators: List[str], needs_chart: bool) -> List[str]:
        """Return tools executed for a routed intent."""
        tools: List[str] = []
        if intent == Intent.QUOTE_LOOKUP:
            tools = ["get_price_history", "generate_with_llm"]
        elif intent == Intent.HISTORICAL_PERFORMANCE:
            tools = ["get_cached_price_data", "get_price_history", "cache_price_data", "compute_summary_stats", "generate_with_llm"]
            if needs_chart:
                tools.append("build_chart")
        elif intent == Intent.TECHNICAL_ANALYSIS:
            tools = ["get_price_history", "compute_indicators", "build_chart", "generate_with_llm"]
        elif intent == Intent.CROSS_TICKER_COMPARISON:
            tools = ["get_multi_ticker_history", "compute_summary_stats", "build_chart", "generate_with_llm"]
        elif intent == Intent.COMPANY_FUNDAMENTALS:
            tools = ["get_company_info", "generate_with_llm"]
        elif intent == Intent.VISUALIZATION_REQUEST:
            tools = ["get_price_history", "build_chart", "generate_with_llm"]
            if indicators:
                tools.insert(1, "compute_indicators")
        elif intent == Intent.CODING_REQUEST:
            tools = ["generate_analysis_code"]
        else:
            tools = ["get_price_history", "compute_summary_stats", "generate_with_llm"]
        return tools
    
    def process_request(
        self,
        message: str,
        session_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        require_approval: bool = False,
        approved: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user request end-to-end.
        
        Args:
            message: User's natural language query
            session_id: Optional session identifier
        
        Returns:
            Structured response dict
        """
        start_time = time.time()
        
        passed, result = apply_guardrails(message)
        if not passed:
            return build_error_response(
                error_code="GUARDRAIL_BLOCKED",
                message=result,
                suggestion="Please rephrase your question without investment advice requests."
            )
        
        effective_provider = provider
        effective_model = model
        effective_api_key = None
        effective_base_url = None
        if provider_config:
            effective_provider = provider_config.get("provider", provider)
            effective_model = provider_config.get("model", model)
            effective_api_key = provider_config.get("api_key")
            effective_base_url = provider_config.get("base_url")

        llm_plan = self._plan_request_with_llm(
            message=message,
            provider=effective_provider,
            model=effective_model,
            api_key=effective_api_key,
            base_url=effective_base_url
        )

        if llm_plan and llm_plan.get("tickers"):
            llm_plan["tickers"] = [normalize_ticker(t) for t in llm_plan["tickers"] if t]

        local_model = (effective_provider == "ollama") or (effective_model and isinstance(effective_model, str) and ":" in effective_model)
        if local_model:
            fallback = self._fallback_plan_for_local_model(message)
            if fallback and fallback.get("tickers"):
                fallback["tickers"] = [normalize_ticker(t) for t in fallback["tickers"] if t]
                if llm_plan is None:
                    llm_plan = fallback
                else:
                    llm_tickers = llm_plan.get("tickers", [])
                    if len(llm_tickers) == 0 or any(t in {"HK", "YTD", "RSI", "MACD"} for t in llm_tickers):
                        llm_plan = fallback

        route_result = llm_plan if llm_plan else route_request(message)

        if route_result.get("tickers"):
            route_result["tickers"] = [normalize_ticker(t) for t in route_result["tickers"] if t]
        
        needs_clar, question = needs_clarification(route_result)
        if needs_clar:
            return build_text_response(
                answer=question or "Please provide more specific information.",
                warnings=["Please provide more specific information."]
            )
        
        tickers = route_result.get("tickers", [])
        period = route_result.get("period", "6mo")
        interval = route_result.get("interval", "1d")
        intent = route_result.get("intent")
        chart_type = route_result.get("chart_type")
        indicators = route_result.get("indicators", [])
        needs_chart = route_result.get("needs_chart", False)

        if require_approval and not approved:
            trace_intent = intent if isinstance(intent, Intent) else Intent.GENERAL_QA
            tools = self._tools_for_intent(trace_intent, indicators, needs_chart)
            plan = {
                "intent": trace_intent.value,
                "tickers": tickers,
                "period": period,
                "interval": interval,
                "chart_type": chart_type.value if isinstance(chart_type, ChartType) else chart_type,
                "indicators": indicators,
                "tools_to_run": tools,
                "explanation": (
                    "I will fetch market data, compute requested analytics, "
                    "and generate visualizations with a written summary. "
                    "No order execution or brokerage actions are performed."
                ),
                "risk_notes": [
                    "Market data can be delayed.",
                    "Data source availability may vary by symbol/market.",
                    "Output is informational, not investment advice."
                ]
            }
            return {
                "answer": "I prepared an execution plan. Please review and approve to continue.",
                "summary": "Approval required before executing tools.",
                "approval_required": True,
                "execution_plan": plan,
                "tickers": tickers,
                "charts": [],
                "tables": [],
                "warnings": [],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "execution_trace": {
                        "planner_source": "llm" if llm_plan else "rule_based",
                        "llm_plan_used": llm_plan is not None,
                        "routed_request": self._serialize_route_result(route_result),
                        "tools_executed": []
                    }
                },
                "runtime_ms": int((time.time() - start_time) * 1000)
            }
        
        try:
            if intent == Intent.QUOTE_LOOKUP:
                response = self._handle_quote_lookup(
                    tickers,
                    message,
                    effective_provider,
                    effective_model,
                    effective_api_key,
                    effective_base_url
                )
            elif intent == Intent.HISTORICAL_PERFORMANCE:
                response = self._handle_historical(
                    tickers,
                    period,
                    interval,
                    message,
                    effective_provider,
                    effective_model,
                    effective_api_key,
                    effective_base_url
                )
            elif intent == Intent.TECHNICAL_ANALYSIS:
                response = self._handle_technical(
                    tickers,
                    period,
                    interval,
                    indicators,
                    message,
                    effective_provider,
                    effective_model,
                    effective_api_key,
                    effective_base_url
                )
            elif intent == Intent.CROSS_TICKER_COMPARISON:
                response = self._handle_comparison(
                    tickers,
                    period,
                    interval,
                    chart_type,
                    message,
                    effective_provider,
                    effective_model,
                    effective_api_key,
                    effective_base_url
                )
            elif intent == Intent.COMPANY_FUNDAMENTALS:
                response = self._handle_fundamentals(
                    tickers,
                    message,
                    effective_provider,
                    effective_model,
                    effective_api_key,
                    effective_base_url
                )
            elif intent == Intent.VISUALIZATION_REQUEST:
                response = self._handle_visualization(
                    tickers,
                    period,
                    interval,
                    chart_type,
                    indicators,
                    message,
                    effective_provider,
                    effective_model,
                    effective_api_key,
                    effective_base_url
                )
            elif intent == Intent.CODING_REQUEST:
                response = self._handle_coding(message, tickers)
            else:
                response = self._handle_general(
                    message,
                    tickers,
                    period,
                    interval,
                    effective_provider,
                    effective_model,
                    effective_api_key,
                    effective_base_url
                )
            
            if isinstance(response, dict) and "status" not in response:
                response["runtime_ms"] = int((time.time() - start_time) * 1000)
                metadata = response.get("metadata", {})
                trace_intent = intent if isinstance(intent, Intent) else Intent.GENERAL_QA
                metadata["execution_trace"] = {
                    "planner_source": "llm" if llm_plan else "rule_based",
                    "llm_plan_used": llm_plan is not None,
                    "routed_request": self._serialize_route_result(route_result),
                    "tools_executed": self._tools_for_intent(trace_intent, indicators, needs_chart),
                }
                response["metadata"] = metadata
            elif isinstance(response, dict) and response.get("status") == "error":
                response["runtime_ms"] = int((time.time() - start_time) * 1000)
            
            return response
            
        except Exception as e:
            return build_error_response(
                error_code="PROCESSING_ERROR",
                message=f"An error occurred: {str(e)}",
                suggestion="Please try again with a different query."
            )
    
    def _handle_quote_lookup(
        self,
        tickers: List[str],
        question: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle quote lookup requests."""
        if not tickers:
            return build_text_response(
                answer="No ticker specified. Please provide a ticker symbol.",
                warnings=["Please provide a valid stock ticker."]
            )
        
        ticker = tickers[0]
        result = get_price_history(ticker, period="5d", interval="5m")
        
        if result["status"] != "success":
            msg = result.get("message", "No data found")
            if result["status"] == "RATE_LIMIT_OR_SOURCE_ERROR":
                msg = f"Yahoo Finance is temporarily unavailable. {msg}. Please try again in a few minutes."
            else:
                msg = f"No data found for {ticker}. The ticker may be delisted or unavailable. Please check the ticker symbol."
            return build_text_response(
                answer=msg,
                tickers=[ticker],
                warnings=["Yahoo Finance data may be delayed or unavailable."]
            )
        
        df = pd.DataFrame(result["data"])
        current_price = df["Close"].iloc[-1] if "Close" in df.columns else None
        prev_close = df["Close"].iloc[0] if len(df) > 0 and "Close" in df.columns else None
        
        answer = f"**{ticker} Current Quote**\n"
        if current_price:
            answer += f"- Current Price: ${current_price:.2f}\n"
        if prev_close:
            change = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0
            sign = "+" if change >= 0 else ""
            answer += f"- Change: {sign}{change:.2f}%\n"
        answer += f"- Data as of: {result['fetched_at']}\n"
        answer += DATABASE_WARNING
        
        llm_analysis = self.generate_analysis(
            question=question or f"What is {ticker} current quote?",
            data={"ticker": ticker, "quote": {"current_price": current_price, "previous_price": prev_close}},
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url
        )

        if llm_analysis:
            answer = f"{llm_analysis}\n\n{answer}"

        summary = llm_analysis if llm_analysis else generate_summary("quote_lookup", [ticker], "5d", {
            "current_price": current_price
        })

        return build_text_response(
            answer,
            tickers=[ticker],
            warnings=["Yahoo Finance data may be delayed."],
            summary=summary
        )
    
    def _handle_historical(
        self,
        tickers: List[str],
        period: str,
        interval: str,
        question: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle historical performance requests."""
        if not tickers:
            return build_text_response(
                answer="No ticker specified. Please provide a stock ticker.",
                warnings=["Please specify a stock ticker."]
            )
        
        ticker = tickers[0]
        
        cached = get_cached_price_data(ticker, period, interval)
        if cached:
            result = cached["data"]
        else:
            result = get_price_history(ticker, period=period, interval=interval)
            if result["status"] == "success":
                cache_price_data(ticker, period, interval, result)
        
        if result["status"] != "success":
            msg = result.get("message", "No data found")
            if result["status"] == "RATE_LIMIT_OR_SOURCE_ERROR":
                msg = f"Yahoo Finance is temporarily unavailable. Please try again."
            return build_text_response(
                answer=msg,
                tickers=[ticker],
                warnings=["Data temporarily unavailable."]
            )
        
        df = pd.DataFrame(result["data"])
        stats = compute_summary_stats(df, ticker)
        
        answer = format_summary_stats(stats, ticker)
        answer += f"\n\nData period: {period}\n"
        answer += DATABASE_WARNING
        
        llm_analysis = self.generate_analysis(
            question=question or f"Analyze {ticker} historical performance for {period}",
            data={"ticker": ticker, "period": period, "stats": stats},
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url
        )

        if llm_analysis:
            answer = f"{llm_analysis}\n\n{answer}"

        summary = generate_summary("historical_performance", tickers, period, stats)
        if llm_analysis:
            summary = llm_analysis
        
        charts = []
        chart_result = build_chart(df, "price_line", [ticker])
        if chart_result.get("status") == "success":
            charts.append(ChartResponse(
                chart_id=chart_result["chart_id"],
                chart_type=chart_result["chart_type"],
                html_path=chart_result.get("html_path"),
                figure_json=chart_result.get("figure_json"),
                tickers=[ticker]
            ))
        
        return build_text_response(
            answer,
            tickers=[ticker],
            charts=charts,
            warnings=["Yahoo Finance data may be delayed."],
            summary=summary
        )
    
    def _handle_technical(
        self,
        tickers: List[str],
        period: str,
        interval: str,
        indicators: List[str],
        question: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle technical analysis requests."""
        if not tickers:
            return build_text_response(
                answer="No ticker specified. Please provide a stock ticker.",
                warnings=["Please specify a stock ticker."]
            )
        
        ticker = tickers[0]
        
        result = get_price_history(ticker, period=period, interval=interval)
        if result["status"] != "success":
            msg = result.get("message", "No data found")
            return build_text_response(
                answer=msg,
                tickers=[ticker],
                warnings=["Data temporarily unavailable."]
            )
        
        df = pd.DataFrame(result["data"])
        df_with_indicators = compute_indicators(df, indicators if indicators else ["sma_20", "sma_50", "rsi_14"])
        
        answer = f"**{ticker} Technical Analysis** ({period})\n\n"
        
        for ind in indicators if indicators else ["sma_20", "sma_50", "rsi_14"]:
            col_name = ind.upper() if ind.startswith("rsi") or ind.startswith("macd") else ind.title()
            if col_name in df_with_indicators.columns:
                last_val = df_with_indicators[col_name].iloc[-1]
                if pd.notna(last_val):
                    answer += f"- {col_name}: {last_val:.2f}\n"
        
        charts = []
        chart_type = "macd" if "macd" in indicators else "rsi" if "rsi" in indicators else "candlestick"
        chart_result = build_chart(df_with_indicators, chart_type, [ticker])
        if chart_result.get("status") == "success":
            charts.append(ChartResponse(
                chart_id=chart_result["chart_id"],
                chart_type=chart_result["chart_type"],
                html_path=chart_result.get("html_path"),
                figure_json=chart_result.get("figure_json"),
                tickers=[ticker]
            ))
        
        llm_analysis = self.generate_analysis(
            question=question or f"Technical analysis for {ticker}",
            data={"ticker": ticker, "period": period, "indicators": indicators, "latest": df_with_indicators.tail(5).to_dict(orient="records")},
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url
        )

        if llm_analysis:
            answer = f"{llm_analysis}\n\n{answer}"

        summary = generate_summary("technical_analysis", tickers, period)
        if llm_analysis:
            summary = llm_analysis
        elif indicators:
            summary += f" using indicators: {', '.join(indicators)}"
        return build_text_response(answer, tickers=[ticker], charts=charts, summary=summary)
    
    def _handle_comparison(
        self,
        tickers: List[str],
        period: str,
        interval: str,
        chart_type: Optional[str],
        question: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle cross-ticker comparison requests."""
        if len(tickers) < 2:
            return build_text_response(
                answer="At least 2 tickers required for comparison. Please provide multiple stock tickers to compare.",
                warnings=["Please provide at least 2 tickers."]
            )
        
        result = get_multi_ticker_history(tickers, period=period, interval=interval)
        if result["status"] != "success":
            # Fallback: fetch each ticker individually when bulk endpoint fails.
            recovered_data: Dict[str, Any] = {}
            for ticker in tickers:
                single = get_price_history(ticker, period=period, interval=interval)
                if single.get("status") == "success" and single.get("data"):
                    recovered_data[ticker] = {
                        "records": single["data"],
                        "columns": single.get("columns", [])
                    }
                else:
                    recovered_data[ticker] = {"error": single.get("message", "No data available for this ticker")}

            if any(isinstance(v, dict) and v.get("records") for v in recovered_data.values()):
                result = {
                    "status": "success",
                    "tickers": tickers,
                    "data": recovered_data,
                    "period": period,
                    "interval": interval,
                }
            else:
                msg = result.get("message", "No data found for provided tickers")
                return build_text_response(
                    answer=msg,
                    tickers=tickers,
                    warnings=["Data temporarily unavailable."]
                )
        
        stats_list = []
        valid_frames: Dict[str, pd.DataFrame] = {}
        missing_tickers: List[str] = []
        for ticker, data in result["data"].items():
            if isinstance(data, dict) and "records" in data and data["records"]:
                df = pd.DataFrame(data["records"])
                if not df.empty:
                    valid_frames[ticker] = df
                    stats = compute_summary_stats(df, ticker)
                    stats_list.append(stats)
                else:
                    missing_tickers.append(ticker)
            else:
                missing_tickers.append(ticker)

        if not stats_list:
            return build_text_response(
                answer="No data found for provided tickers",
                tickers=tickers,
                warnings=["Data temporarily unavailable."]
            )
        
        llm_analysis = self.generate_analysis(
            question=question or f"Compare {', '.join(tickers)} over {period}",
            data={"tickers": tickers, "period": period, "stats": stats_list},
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        
        answer = f"**Comparison ({period})**\n\n"
        if llm_analysis:
            answer += llm_analysis + "\n\n"
        
        answer += "| Ticker | Price | Return | Volatility | Max Drawdown |\n"
        answer += "|--------|-------|--------|------------|-------------|\n"
        
        tables = []
        table_rows = [["Ticker", "Price", "Return %", "Volatility %", "Max DD %"]]
        
        for stats in stats_list:
            ticker = stats["ticker"]
            price = stats.get("current_price", 0) or 0
            ret = stats.get("cumulative_return_pct", 0) or 0
            vol = stats.get("annualized_volatility_pct", 0) or 0
            dd = stats.get("max_drawdown_pct", 0) or 0
            
            answer += f"| {ticker} | ${price:.2f} | {ret:+.2f} | {vol:.2f} | {dd:.2f} |\n"
            table_rows.append([ticker, f"${price:.2f}", f"{ret:+.2f}", f"{vol:.2f}", f"{dd:.2f}"])
        
        tables.append(TableData(
            name="comparison",
            headers=table_rows[0],
            rows=table_rows[1:]
        ))
        
        summary = generate_summary("cross_ticker_comparison", list(valid_frames.keys()), period, stats_list[0] if stats_list else None)
        if llm_analysis:
            summary = llm_analysis
        
        charts = []
        first_ticker = next(iter(valid_frames.keys()))
        chart_result = build_chart(
            valid_frames,
            "normalized_compare",
            list(valid_frames.keys())
        )
        if chart_result.get("status") == "success":
            charts.append(ChartResponse(
                chart_id=chart_result["chart_id"],
                chart_type=chart_result["chart_type"],
                html_path=chart_result.get("html_path"),
                figure_json=chart_result.get("figure_json"),
                tickers=tickers
            ))
        
        warnings = []
        if missing_tickers:
            warnings.append(f"No data for: {', '.join(missing_tickers)}")

        return build_text_response(
            answer,
            tickers=list(valid_frames.keys()),
            charts=charts,
            tables=tables,
            summary=summary,
            warnings=warnings
        )
    
    def _handle_fundamentals(
        self,
        tickers: List[str],
        question: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle company fundamentals requests."""
        if not tickers:
            return build_text_response(
                answer="No ticker specified. Please provide a stock ticker.",
                warnings=["Please specify a stock ticker."]
            )
        
        ticker = tickers[0]
        result = get_company_info(ticker)
        
        if result["status"] != "success":
            msg = result.get("message", "No data found")
            return build_text_response(
                answer=msg,
                tickers=[ticker],
                warnings=["Company data temporarily unavailable."]
            )
        
        data = result["data"]
        answer = f"**{data.get('shortName', ticker)} Fundamentals**\n\n"
        
        if data.get("sector"):
            answer += f"- Sector: {data['sector']}\n"
        if data.get("industry"):
            answer += f"- Industry: {data['industry']}\n"
        if data.get("marketCap"):
            market_cap = data['marketCap']
            if market_cap >= 1e12:
                answer += f"- Market Cap: ${market_cap/1e12:.2f}T\n"
            elif market_cap >= 1e9:
                answer += f"- Market Cap: ${market_cap/1e9:.2f}B\n"
        if data.get("currentPrice"):
            answer += f"- Current Price: ${data['currentPrice']:.2f}\n"
        if data.get("trailingPE"):
            answer += f"- P/E Ratio (TTM): {data['trailingPE']:.2f}\n"
        if data.get("forwardPE"):
            answer += f"- P/E Ratio (Forward): {data['forwardPE']:.2f}\n"
        if data.get("priceToBook"):
            answer += f"- Price/Book: {data['priceToBook']:.2f}\n"
        if data.get("priceToSalesTrailing12Months"):
            answer += f"- Price/Sales (TTM): {data['priceToSalesTrailing12Months']:.2f}\n"
        if data.get("dividendYield"):
            dy = data['dividendYield']
            if 0 < dy <= 1:
                dy = dy * 100
            answer += f"- Dividend Yield: {dy:.2f}%\n"
        if data.get("beta"):
            answer += f"- Beta: {data['beta']:.2f}\n"
        if data.get("totalRevenue"):
            answer += f"- Revenue (latest): ${data['totalRevenue']/1e9:.2f}B\n"
        if data.get("netIncome"):
            answer += f"- Net Income (latest): ${data['netIncome']/1e9:.2f}B\n"
        if data.get("operatingCashFlow"):
            answer += f"- Operating Cash Flow (latest): ${data['operatingCashFlow']/1e9:.2f}B\n"
        if data.get("freeCashFlow"):
            answer += f"- Free Cash Flow (latest): ${data['freeCashFlow']/1e9:.2f}B\n"
        
        llm_analysis = self.generate_analysis(
            question=question or f"Company fundamentals for {ticker}",
            data={"ticker": ticker, "fundamentals": data},
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url
        )

        if llm_analysis:
            answer = f"{llm_analysis}\n\n{answer}"

        summary = generate_summary("company_fundamentals", tickers, "snapshot")
        if llm_analysis:
            summary = llm_analysis
        return build_text_response(answer, tickers=[ticker], summary=summary)
    
    def _handle_visualization(
        self,
        tickers: List[str],
        period: str,
        interval: str,
        chart_type: Optional[str],
        indicators: List[str],
        question: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle visualization requests."""
        if not tickers:
            return build_text_response(
                answer="No ticker specified. Please provide a stock ticker.",
                warnings=["Please specify a stock ticker."]
            )
        
        ticker = tickers[0]
        
        result = get_price_history(ticker, period=period, interval=interval)
        if result["status"] != "success":
            msg = result.get("message", "No data found")
            return build_text_response(
                answer=msg,
                tickers=[ticker],
                warnings=["Data temporarily unavailable."]
            )
        
        df = pd.DataFrame(result["data"])
        
        if indicators:
            df = compute_indicators(df, indicators)
        
        chart_type_str = chart_type if isinstance(chart_type, str) else (chart_type.value if chart_type else "price_line")
        chart_result = build_chart(df, chart_type_str, tickers)
        
        charts = []
        if chart_result.get("status") == "success":
            charts.append(ChartResponse(
                chart_id=chart_result["chart_id"],
                chart_type=chart_result["chart_type"],
                html_path=chart_result.get("html_path"),
                figure_json=chart_result.get("figure_json"),
                tickers=tickers
            ))
        
        llm_analysis = self.generate_analysis(
            question=question or f"Visualize {ticker} with {chart_type_str}",
            data={"ticker": ticker, "period": period, "chart_type": chart_type_str, "indicators": indicators},
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url
        )

        answer = f"**{ticker} Chart** ({period})\n"
        if llm_analysis:
            answer = f"{llm_analysis}\n\n" + answer
        answer += f"Chart type: {chart_type_str}\n"
        answer += f"Generated at: {datetime.now().isoformat()}\n"
        if chart_result.get("html_path"):
            answer += f"File: {chart_result['html_path']}\n"
        
        summary = generate_summary("visualization_request", tickers, period)
        if llm_analysis:
            summary = llm_analysis
        return build_text_response(answer, tickers=tickers, charts=charts, summary=summary)
    
    def _handle_coding(self, message: str, tickers: List[str]) -> Dict[str, Any]:
        """Handle coding requests."""
        ticker = tickers[0] if tickers else "AAPL"
        
        code = generate_analysis_code(message, ticker)
        
        return {
            "generated_code": code,
            "execution_status": "generated",
            "artifacts": [],
            "summary": f"Generated Python analysis code for {ticker}"
        }
    
    def _handle_general(
        self,
        message: str,
        tickers: List[str],
        period: str,
        interval: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle general Q&A requests."""
        if not tickers:
            return build_text_response(
                "I can help you with stock analysis. Please specify a ticker symbol. "
                "For example: 'Show me AAPL price' or 'Compare NVDA and TSLA'",
                warnings=["Please provide a specific stock ticker."]
            )
        
        ticker = tickers[0]
        result = get_price_history(ticker, period=period, interval=interval)
        
        if result["status"] != "success":
            msg = result.get("message", "No data found")
            return build_text_response(
                answer=msg,
                tickers=[ticker],
                warnings=["Data temporarily unavailable."]
            )
        
        df = pd.DataFrame(result["data"])
        stats = compute_summary_stats(df, ticker)
        
        llm_analysis = self.generate_analysis(
            question=message,
            data={"ticker": ticker, "period": period, "stats": stats},
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url
        )

        answer = f"**{ticker} Overview** ({period})\n\n"
        if llm_analysis:
            answer = llm_analysis + "\n\n" + answer
        answer += format_summary_stats(stats, ticker)
        answer += "\n\n" + DATABASE_WARNING

        summary = generate_summary("general_qa", tickers, period, stats)
        if llm_analysis:
            summary = llm_analysis
        return build_text_response(answer, tickers=[ticker], summary=summary)


stock_agent = StockAgent()
