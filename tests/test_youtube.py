"""Tests for the YouTube search tool."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.llm_intents.const import (
    CONF_PROVIDER_API_KEYS,
    DOMAIN,
    PROVIDER_GOOGLE,
)
from custom_components.llm_intents.youtube import SearchYouTubeTool

from .utils import MockContext, mock_session


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
def youtube_tool(mock_hass: HomeAssistant) -> SearchYouTubeTool:
    """Create SearchYouTubeTool instance."""
    config = {CONF_PROVIDER_API_KEYS: {PROVIDER_GOOGLE: "test_google_key"}}
    mock_entry = MagicMock()
    mock_entry.options = {}
    mock_hass.config_entries = MagicMock()
    mock_hass.config_entries.async_entries.return_value = [mock_entry]
    mock_hass.data = {DOMAIN: {"config": config}}
    return SearchYouTubeTool(config, mock_hass)


@pytest.fixture
def youtube_tool_no_key(mock_hass: HomeAssistant) -> SearchYouTubeTool:
    """Create SearchYouTubeTool with no API key."""
    config = {CONF_PROVIDER_API_KEYS: {}}
    mock_entry = MagicMock()
    mock_entry.options = {}
    mock_hass.config_entries = MagicMock()
    mock_hass.config_entries.async_entries.return_value = [mock_entry]
    mock_hass.data = {DOMAIN: {"config": config}}
    return SearchYouTubeTool(config, mock_hass)


async def test_youtube_search_success(
    mock_hass: HomeAssistant, youtube_tool: SearchYouTubeTool
) -> None:
    """Test successful YouTube search."""
    api_data = {
        "items": [
            {
                "id": {"videoId": "abc123"},
                "snippet": {
                    "title": "Test Video",
                    "channelTitle": "Test Channel",
                    "description": "A great video",
                    "publishedAt": "2024-01-01T00:00:00Z",
                },
            }
        ]
    }

    with patch(
        "custom_components.llm_intents.youtube.async_get_clientsession",
        return_value=mock_session(200, api_data),
    ):
        tool_input = llm.ToolInput(
            tool_name="search_youtube",
            tool_args={"query": "test video", "num_results": 1},
        )
        result = await youtube_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Test Video"
    assert result["results"][0]["url"] == "https://www.youtube.com/watch?v=abc123"
    assert result["results"][0]["channel"] == "Test Channel"


async def test_youtube_search_no_api_key(
    mock_hass: HomeAssistant, youtube_tool_no_key: SearchYouTubeTool
) -> None:
    """Test YouTube search without API key returns error."""
    tool_input = llm.ToolInput(
        tool_name="search_youtube",
        tool_args={"query": "test"},
    )
    result = await youtube_tool_no_key.async_call(
        mock_hass, tool_input, _make_llm_context()
    )

    assert "error" in result
    assert "not configured" in result["error"]


async def test_youtube_search_no_results(
    mock_hass: HomeAssistant, youtube_tool: SearchYouTubeTool
) -> None:
    """Test YouTube search with no results."""
    with patch(
        "custom_components.llm_intents.youtube.async_get_clientsession",
        return_value=mock_session(200, {"items": []}),
    ):
        tool_input = llm.ToolInput(
            tool_name="search_youtube",
            tool_args={"query": "nonexistent video"},
        )
        result = await youtube_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "result" in result
    assert result["result"] == "No videos found"


async def test_youtube_search_http_error(
    mock_hass: HomeAssistant, youtube_tool: SearchYouTubeTool
) -> None:
    """Test YouTube search with HTTP error."""
    resp = AsyncMock()
    resp.status = 403
    resp.json = AsyncMock(return_value={})
    resp.text = AsyncMock(return_value="Forbidden")

    session = AsyncMock()
    session.get = Mock(return_value=MockContext(resp))

    with patch(
        "custom_components.llm_intents.youtube.async_get_clientsession",
        return_value=session,
    ):
        tool_input = llm.ToolInput(
            tool_name="search_youtube",
            tool_args={"query": "test"},
        )
        result = await youtube_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "error" in result
    assert "403" in result["error"]


async def test_youtube_search_exception(
    mock_hass: HomeAssistant, youtube_tool: SearchYouTubeTool
) -> None:
    """Test YouTube search with unexpected exception."""
    session = AsyncMock()
    session.get = MagicMock(side_effect=Exception("Connection failed"))

    with patch(
        "custom_components.llm_intents.youtube.async_get_clientsession",
        return_value=session,
    ):
        tool_input = llm.ToolInput(
            tool_name="search_youtube",
            tool_args={"query": "test"},
        )
        result = await youtube_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "error" in result


async def test_youtube_cached_response(
    mock_hass: HomeAssistant, youtube_tool: SearchYouTubeTool
) -> None:
    """Test that cached responses are returned."""
    cached = {"results": [{"title": "Cached Video"}]}
    with (
        patch("custom_components.llm_intents.youtube.SQLiteCache") as mock_cache_cls,
        patch("custom_components.llm_intents.youtube.async_get_clientsession"),
    ):
        mock_cache_cls.return_value.get.return_value = cached
        tool_input = llm.ToolInput(
            tool_name="search_youtube",
            tool_args={"query": "cached"},
        )
        result = await youtube_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert result == cached
