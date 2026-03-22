"""Caching tool using diskcache."""
import json
import hashlib
from typing import Optional, Any
from datetime import datetime, timedelta

import diskcache

from app.config import config


cache = diskcache.Cache(str(config.DATA_DIR / "cache"))


def _make_key(prefix: str, **kwargs) -> str:
    """Generate a cache key from prefix and kwargs."""
    sorted_args = json.dumps(kwargs, sort_keys=True, default=str)
    hash_val = hashlib.md5(sorted_args.encode()).hexdigest()
    return f"{prefix}:{hash_val}"


def get_cached(prefix: str, **kwargs) -> Optional[Any]:
    """
    Get a value from cache.
    
    Args:
        prefix: Cache key prefix
        **kwargs: Key-value pairs for cache lookup
    
    Returns:
        Cached value or None if not found
    """
    key = _make_key(prefix, **kwargs)
    result = cache.get(key)
    
    if result is not None:
        cached_at = cache.get(f"{key}:timestamp")
        return {
            "data": result,
            "cached_at": cached_at,
            "from_cache": True
        }
    
    return None


def set_cached(prefix: str, value: Any, ttl: Optional[int] = None, **kwargs) -> str:
    """
    Set a value in cache.
    
    Args:
        prefix: Cache key prefix
        value: Value to cache
        ttl: Time to live in seconds
        **kwargs: Key-value pairs for cache lookup
    
    Returns:
        Cache key
    """
    key = _make_key(prefix, **kwargs)
    timestamp = datetime.now().isoformat()
    
    cache.set(key, value, expire=ttl)
    cache.set(f"{key}:timestamp", timestamp)
    
    return key


def invalidate_cache(prefix: str = None, **kwargs) -> int:
    """
    Invalidate cache entries.
    
    Args:
        prefix: Optional prefix to filter by
        **kwargs: Key-value pairs to match
    
    Returns:
        Number of entries invalidated
    """
    count = 0
    
    if prefix and not kwargs:
        for key in list(cache.iterkeys()):
            if key.startswith(f"{prefix}:"):
                cache.delete(key)
                count += 1
    else:
        key = _make_key(prefix, **kwargs) if prefix else None
        if key and cache.get(key):
            cache.delete(key)
            count = 1
    
    return count


def clear_all_cache() -> int:
    """Clear all cache entries."""
    count = len(cache)
    cache.clear()
    return count


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "size": len(cache),
        "volume": cache.volume(),
        "path": str(cache.directory)
    }


def cache_price_data(ticker: str, period: str, interval: str, data: dict) -> str:
    """Cache price data with appropriate TTL."""
    ttl = config.INTRADAY_CACHE_TTL if interval in ["1m", "2m", "5m", "15m", "30m", "60m"] else config.CACHE_TTL_SECONDS
    return set_cached("price", data, ttl, ticker=ticker, period=period, interval=interval)


def get_cached_price_data(ticker: str, period: str, interval: str) -> Optional[dict]:
    """Get cached price data if available."""
    return get_cached("price", ticker=ticker, period=period, interval=interval)


def cache_company_info(ticker: str, data: dict) -> str:
    """Cache company info with 1-day TTL."""
    return set_cached("company", data, ttl=86400, ticker=ticker)


def get_cached_company_info(ticker: str) -> Optional[dict]:
    """Get cached company info if available."""
    return get_cached("company", ticker=ticker)


def cache_chart(chart_type: str, tickers: list, params: dict, figure_json: str) -> str:
    """Cache chart data."""
    return set_cached("chart", {"figure_json": figure_json, "params": params}, ttl=config.CACHE_TTL_SECONDS,
                      chart_type=chart_type, tickers=sorted(tickers))


def get_cached_chart(chart_type: str, tickers: list) -> Optional[dict]:
    """Get cached chart if available."""
    return get_cached("chart", chart_type=chart_type, tickers=sorted(tickers))
