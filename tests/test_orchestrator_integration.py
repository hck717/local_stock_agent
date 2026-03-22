"""Integration-style tests for orchestrator planning flow."""

from app.agent.orchestrator import StockAgent
from app.agent.schemas import Intent, ChartType
from app.services.response_builder import build_text_response


def test_process_request_uses_llm_plan_and_sets_execution_trace(monkeypatch):
    """LLM plan path should be used and exposed in metadata trace."""
    agent = StockAgent()

    monkeypatch.setattr("app.agent.orchestrator.apply_guardrails", lambda text: (True, text))
    monkeypatch.setattr("app.agent.orchestrator.needs_clarification", lambda route_result: (False, None))

    llm_plan = {
        "intent": Intent.HISTORICAL_PERFORMANCE,
        "tickers": ["AAPL"],
        "period": "1y",
        "interval": "1d",
        "start_date": None,
        "end_date": None,
        "chart_type": ChartType.PRICE_LINE,
        "indicators": [],
        "needs_chart": True,
    }
    monkeypatch.setattr(agent, "_plan_request_with_llm", lambda *args, **kwargs: llm_plan)

    captured = {}

    def fake_handle_historical(tickers, period, interval, question, provider, model, api_key, base_url):
        captured["tickers"] = tickers
        captured["period"] = period
        captured["interval"] = interval
        captured["provider"] = provider
        captured["model"] = model
        captured["api_key"] = api_key
        return build_text_response("historical-ok", tickers=tickers, summary="historical-summary")

    monkeypatch.setattr(agent, "_handle_historical", fake_handle_historical)

    response = agent.process_request(
        message="Analyze AAPL over 1 year",
        provider_config={"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
    )

    assert response["answer"] == "historical-ok"
    assert response["summary"] == "historical-summary"
    assert response["runtime_ms"] >= 0

    trace = response["metadata"]["execution_trace"]
    assert trace["planner_source"] == "llm"
    assert trace["llm_plan_used"] is True
    assert trace["routed_request"]["intent"] == "historical_performance"
    assert trace["routed_request"]["chart_type"] == "price_line"
    assert "get_price_history" in trace["tools_executed"]

    assert captured["tickers"] == ["AAPL"]
    assert captured["period"] == "1y"
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-4o"
    assert captured["api_key"] == "test-key"


def test_process_request_falls_back_to_router_when_llm_plan_missing(monkeypatch):
    """If LLM planning fails, router output should be used with trace."""
    agent = StockAgent()

    monkeypatch.setattr("app.agent.orchestrator.apply_guardrails", lambda text: (True, text))
    monkeypatch.setattr(agent, "_plan_request_with_llm", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.agent.orchestrator.needs_clarification", lambda route_result: (False, None))

    monkeypatch.setattr(
        "app.agent.orchestrator.route_request",
        lambda text: {
            "intent": Intent.QUOTE_LOOKUP,
            "tickers": ["MSFT"],
            "period": "6mo",
            "interval": "1d",
            "start_date": None,
            "end_date": None,
            "chart_type": None,
            "indicators": [],
            "needs_chart": False,
        },
    )

    monkeypatch.setattr(
        agent,
        "_handle_quote_lookup",
        lambda tickers, question, provider, model, api_key, base_url: build_text_response("quote-ok", tickers=tickers),
    )

    response = agent.process_request("Show me MSFT price")

    assert response["answer"] == "quote-ok"
    assert response["runtime_ms"] >= 0

    trace = response["metadata"]["execution_trace"]
    assert trace["planner_source"] == "rule_based"
    assert trace["llm_plan_used"] is False
    assert trace["routed_request"]["intent"] == "quote_lookup"
    assert "get_price_history" in trace["tools_executed"]
