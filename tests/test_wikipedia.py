"""Tests for the Wikipedia search tool."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.llm_intents.const import CONF_WIKIPEDIA_NUM_RESULTS
from custom_components.llm_intents.wikipedia import SearchWikipediaTool

from .utils import MockContext, mock_response, mock_session


def _make_llm_context() -> llm.LLMContext:
    return llm.LLMContext(
        platform="test",
        context=None,
        user_prompt=None,
        language="en",
        assistant=None,
        device_id=None,
    )


@pytest.fixture
def wikipedia_tool(mock_hass: HomeAssistant) -> SearchWikipediaTool:
    """Create SearchWikipediaTool instance."""
    config = {CONF_WIKIPEDIA_NUM_RESULTS: 1}
    mock_entry = MagicMock()
    mock_entry.options = {}
    mock_hass.config_entries = MagicMock()
    mock_hass.config_entries.async_entries.return_value = [mock_entry]
    return SearchWikipediaTool(config, mock_hass)


async def test_wikipedia_search_success(
    mock_hass: HomeAssistant, wikipedia_tool: SearchWikipediaTool
) -> None:
    """Test successful Wikipedia search returns results."""
    search_data = {
        "query": {
            "search": [
                {
                    "title": "Python (programming language)",
                    "snippet": "A <b>programming</b> language",
                },
            ]
        }
    }
    summary_data = {
        "extract": "Python is a high-level programming language.",
    }

    session = AsyncMock()
    call_count = 0

    def mock_get(*args: object, **kwargs: Any) -> MockContext:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockContext(mock_response(200, search_data))
        return MockContext(mock_response(200, summary_data))

    session.get = Mock(side_effect=mock_get)

    with patch(
        "custom_components.llm_intents.wikipedia.async_get_clientsession",
        return_value=session,
    ):
        tool_input = llm.ToolInput(
            tool_name="search_wikipedia", tool_args={"query": "Python"}
        )
        result = await wikipedia_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Python (programming language)"
    assert (
        result["results"][0]["summary"]
        == "Python is a high-level programming language."
    )


async def test_wikipedia_search_no_results(
    mock_hass: HomeAssistant, wikipedia_tool: SearchWikipediaTool
) -> None:
    """Test Wikipedia search with no results."""
    with patch(
        "custom_components.llm_intents.wikipedia.async_get_clientsession",
        return_value=mock_session(200, {"query": {"search": []}}),
    ):
        tool_input = llm.ToolInput(
            tool_name="search_wikipedia",
            tool_args={"query": "xyznonexistent"},
        )
        result = await wikipedia_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "result" in result
    assert "No Wikipedia articles found" in result["result"]


async def test_wikipedia_search_http_error(
    mock_hass: HomeAssistant, wikipedia_tool: SearchWikipediaTool
) -> None:
    """Test Wikipedia search with HTTP error."""
    with patch(
        "custom_components.llm_intents.wikipedia.async_get_clientsession",
        return_value=mock_session(500, {}),
    ):
        tool_input = llm.ToolInput(
            tool_name="search_wikipedia", tool_args={"query": "test"}
        )
        result = await wikipedia_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "error" in result
    assert "500" in result["error"]


async def test_wikipedia_search_exception(
    mock_hass: HomeAssistant, wikipedia_tool: SearchWikipediaTool
) -> None:
    """Test Wikipedia search with unexpected exception."""
    session = AsyncMock()
    session.get = MagicMock(side_effect=Exception("Network error"))

    with patch(
        "custom_components.llm_intents.wikipedia.async_get_clientsession",
        return_value=session,
    ):
        tool_input = llm.ToolInput(
            tool_name="search_wikipedia", tool_args={"query": "test"}
        )
        result = await wikipedia_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "error" in result
    assert "Network error" in result["error"]


async def test_wikipedia_search_summary_failure_falls_back_to_snippet(
    mock_hass: HomeAssistant, wikipedia_tool: SearchWikipediaTool
) -> None:
    """Test that when summary API fails, the snippet is used instead."""
    search_data = {
        "query": {
            "search": [
                {"title": "Test Article", "snippet": "A <b>test</b> snippet"},
            ]
        }
    }

    session = AsyncMock()
    call_count = 0

    def mock_get(*args: object, **kwargs: Any) -> MockContext:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockContext(mock_response(200, search_data))
        return MockContext(mock_response(404, {}))

    session.get = Mock(side_effect=mock_get)

    with patch(
        "custom_components.llm_intents.wikipedia.async_get_clientsession",
        return_value=session,
    ):
        tool_input = llm.ToolInput(
            tool_name="search_wikipedia", tool_args={"query": "test"}
        )
        result = await wikipedia_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "results" in result
    assert result["results"][0]["summary"] == "A test snippet"


async def test_wikipedia_cached_response(
    mock_hass: HomeAssistant, wikipedia_tool: SearchWikipediaTool
) -> None:
    """Test that cached responses are returned."""
    cached = {"results": [{"title": "Cached", "summary": "From cache"}]}
    with (
        patch("custom_components.llm_intents.wikipedia.SQLiteCache") as mock_cache_cls,
        patch("custom_components.llm_intents.wikipedia.async_get_clientsession"),
    ):
        mock_cache_cls.return_value.get.return_value = cached
        tool_input = llm.ToolInput(
            tool_name="search_wikipedia", tool_args={"query": "cached"}
        )
        result = await wikipedia_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert result == cached
