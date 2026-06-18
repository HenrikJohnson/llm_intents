"""Tests for the SQLite cache module."""

import time
from unittest.mock import patch

from custom_components.llm_intents.cache import SQLiteCache


def _fresh_cache() -> SQLiteCache:
    """Return a fresh SQLiteCache, bypassing the singleton."""
    SQLiteCache._instance = None
    return SQLiteCache()


def test_singleton_returns_same_instance() -> None:
    """Two calls to SQLiteCache() return the same object."""
    SQLiteCache._instance = None
    a = SQLiteCache()
    b = SQLiteCache()
    assert a is b
    SQLiteCache._instance = None


def test_set_and_get() -> None:
    """Stored values are retrievable."""
    cache = _fresh_cache()
    cache.set("tool_a", {"q": "hello"}, {"answer": 42})
    result = cache.get("tool_a", {"q": "hello"})
    assert result == {"answer": 42}


def test_get_miss_returns_none() -> None:
    """Cache miss returns None."""
    cache = _fresh_cache()
    assert cache.get("nonexistent", {"x": 1}) is None


def test_set_overwrites_existing() -> None:
    """A second set for the same key overwrites the first."""
    cache = _fresh_cache()
    cache.set("tool_b", None, {"v": 1})
    cache.set("tool_b", None, {"v": 2})
    assert cache.get("tool_b", None) == {"v": 2}


def test_cleanup_removes_expired_entries() -> None:
    """Expired entries are purged during get."""
    cache = _fresh_cache()
    cache.set("tool_c", None, {"old": True})

    # Simulate time passing beyond the max age
    future = int(time.time()) + SQLiteCache.DEFAULT_MAX_AGE + 1
    with patch("custom_components.llm_intents.cache.time") as mock_time:
        mock_time.time.return_value = future
        result = cache.get("tool_c", None)

    assert result is None


def test_make_key_deterministic() -> None:
    """Same inputs produce the same cache key."""
    cache = _fresh_cache()
    key1 = cache._make_key("tool", {"a": 1, "b": 2})
    key2 = cache._make_key("tool", {"b": 2, "a": 1})
    assert key1 == key2


def test_make_key_none_params() -> None:
    """None params produce a consistent key."""
    cache = _fresh_cache()
    key1 = cache._make_key("tool", None)
    key2 = cache._make_key("tool", None)
    assert key1 == key2


def test_different_tools_different_keys() -> None:
    """Different tool names produce different keys."""
    cache = _fresh_cache()
    key1 = cache._make_key("tool_x", None)
    key2 = cache._make_key("tool_y", None)
    assert key1 != key2
