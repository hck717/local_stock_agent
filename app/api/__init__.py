"""API package."""
from app.api.routes_chat import router as chat_router
from app.api.routes_charts import router as charts_router
from app.api.routes_health import router as health_router
from app.api.routes_providers import router as providers_router

__all__ = ["chat_router", "charts_router", "health_router", "providers_router"]
