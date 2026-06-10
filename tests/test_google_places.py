"""Tests for the Google Places search tool."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.llm_intents.const import (
    CONF_GOOGLE_PLACES_LATITUDE,
    CONF_GOOGLE_PLACES_LONGITUDE,
    CONF_GOOGLE_PLACES_NUM_RESULTS,
    CONF_GOOGLE_PLACES_RADIUS,
    CONF_GOOGLE_PLACES_RANKING,
    CONF_PROVIDER_API_KEYS,
    DOMAIN,
    PROVIDER_GOOGLE,
    SERVICE_DEFAULTS,
)
from custom_components.llm_intents.google_places import FindPlacesTool

from .utils import MockContext, mock_post_session


def _make_llm_context() -> llm.LLMContext:
    return llm.LLMContext(
        platform="test",
        context=None,
        user_prompt=None,
        language="en",
        assistant=None,
        device_id=None,
    )


def _make_config(api_key: str = "test_key") -> dict:
    return {
        CONF_PROVIDER_API_KEYS: {PROVIDER_GOOGLE: api_key},
        CONF_GOOGLE_PLACES_NUM_RESULTS: 2,
        CONF_GOOGLE_PLACES_LATITUDE: 40.7128,
        CONF_GOOGLE_PLACES_LONGITUDE: -74.0060,
        CONF_GOOGLE_PLACES_RADIUS: SERVICE_DEFAULTS.get(CONF_GOOGLE_PLACES_RADIUS, 10),
        CONF_GOOGLE_PLACES_RANKING: "RELEVANCE",
    }


@pytest.fixture
def places_tool(mock_hass: HomeAssistant) -> FindPlacesTool:
    """Create FindPlacesTool instance."""
    config = _make_config()
    mock_entry = MagicMock()
    mock_entry.options = {}
    mock_hass.config_entries = MagicMock()
    mock_hass.config_entries.async_entries.return_value = [mock_entry]
    mock_hass.data = {DOMAIN: {"config": config}}
    return FindPlacesTool(config, mock_hass)


@pytest.fixture
def places_tool_no_key(mock_hass: HomeAssistant) -> FindPlacesTool:
    """Create FindPlacesTool with no API key."""
    config = _make_config(api_key="")
    mock_entry = MagicMock()
    mock_entry.options = {}
    mock_hass.config_entries = MagicMock()
    mock_hass.config_entries.async_entries.return_value = [mock_entry]
    mock_hass.data = {DOMAIN: {"config": config}}
    return FindPlacesTool(config, mock_hass)


async def test_places_search_success(
    mock_hass: HomeAssistant, places_tool: FindPlacesTool
) -> None:
    """Test successful places search."""
    api_data = {
        "places": [
            {
                "displayName": {"text": "Best Pizza"},
                "shortFormattedAddress": "123 Main St",
                "rating": 4.5,
                "nationalPhoneNumber": "555-1234",
                "regularOpeningHours": {
                    "openNow": True,
                    "weekdayDescriptions": [
                        "Monday: 9:00\u202fAM - 10:00\u202fPM",
                    ],
                },
            }
        ]
    }

    with patch(
        "custom_components.llm_intents.google_places.async_get_clientsession",
        return_value=mock_post_session(200, api_data),
    ):
        tool_input = llm.ToolInput(
            tool_name="find_places", tool_args={"query": "pizza near me"}
        )
        result = await places_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "Best Pizza"
    assert result["results"][0]["address"] == "123 Main St"
    assert result["results"][0]["phone"] == "555-1234"
    assert result["results"][0]["open_now"] is True
    assert len(result["results"][0]["regular_open_hours"]) == 1


async def test_places_search_no_api_key(
    mock_hass: HomeAssistant, places_tool_no_key: FindPlacesTool
) -> None:
    """Test places search without API key returns error."""
    tool_input = llm.ToolInput(tool_name="find_places", tool_args={"query": "pizza"})
    result = await places_tool_no_key.async_call(
        mock_hass, tool_input, _make_llm_context()
    )

    assert "error" in result
    assert "not configured" in result["error"]


async def test_places_search_no_results(
    mock_hass: HomeAssistant, places_tool: FindPlacesTool
) -> None:
    """Test places search with no results."""
    with patch(
        "custom_components.llm_intents.google_places.async_get_clientsession",
        return_value=mock_post_session(200, {"places": []}),
    ):
        tool_input = llm.ToolInput(
            tool_name="find_places", tool_args={"query": "nonexistent"}
        )
        result = await places_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "result" in result
    assert result["result"] == "No places found"


async def test_places_search_http_error(
    mock_hass: HomeAssistant, places_tool: FindPlacesTool
) -> None:
    """Test places search with HTTP error."""
    resp = AsyncMock()
    resp.status = 500
    resp.json = AsyncMock(return_value={})
    resp.text = AsyncMock(return_value="Internal Server Error")

    session = AsyncMock()
    session.post = Mock(return_value=MockContext(resp))

    with patch(
        "custom_components.llm_intents.google_places.async_get_clientsession",
        return_value=session,
    ):
        tool_input = llm.ToolInput(tool_name="find_places", tool_args={"query": "test"})
        result = await places_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "error" in result
    assert "500" in result["error"]


async def test_places_search_exception(
    mock_hass: HomeAssistant, places_tool: FindPlacesTool
) -> None:
    """Test places search with unexpected exception."""
    session = AsyncMock()
    session.post = MagicMock(side_effect=Exception("Timeout"))

    with patch(
        "custom_components.llm_intents.google_places.async_get_clientsession",
        return_value=session,
    ):
        tool_input = llm.ToolInput(tool_name="find_places", tool_args={"query": "test"})
        result = await places_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "error" in result
    assert "Timeout" in result["error"]


async def test_places_search_cached_response(
    mock_hass: HomeAssistant, places_tool: FindPlacesTool
) -> None:
    """Test that cached responses are returned."""
    cached = {"results": [{"name": "Cached Place"}]}
    with (
        patch(
            "custom_components.llm_intents.google_places.SQLiteCache"
        ) as mock_cache_cls,
        patch("custom_components.llm_intents.google_places.async_get_clientsession"),
    ):
        mock_cache_cls.return_value.get.return_value = cached
        tool_input = llm.ToolInput(
            tool_name="find_places", tool_args={"query": "cached"}
        )
        result = await places_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert result == cached


async def test_places_search_place_without_opening_hours(
    mock_hass: HomeAssistant, places_tool: FindPlacesTool
) -> None:
    """Test a place without opening hours info."""
    api_data = {
        "places": [
            {
                "displayName": {"text": "Park"},
                "shortFormattedAddress": "Central Park",
            }
        ]
    }

    with patch(
        "custom_components.llm_intents.google_places.async_get_clientsession",
        return_value=mock_post_session(200, api_data),
    ):
        tool_input = llm.ToolInput(tool_name="find_places", tool_args={"query": "park"})
        result = await places_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert "results" in result
    assert result["results"][0]["name"] == "Park"
    assert "open_now" not in result["results"][0]


async def test_places_search_place_with_no_rating(
    mock_hass: HomeAssistant, places_tool: FindPlacesTool
) -> None:
    """Test a place without a rating."""
    api_data = {
        "places": [
            {
                "displayName": {"text": "New Place"},
                "shortFormattedAddress": "456 New St",
            }
        ]
    }

    with patch(
        "custom_components.llm_intents.google_places.async_get_clientsession",
        return_value=mock_post_session(200, api_data),
    ):
        tool_input = llm.ToolInput(
            tool_name="find_places", tool_args={"query": "new place"}
        )
        result = await places_tool.async_call(
            mock_hass, tool_input, _make_llm_context()
        )

    assert result["results"][0]["rating"] == "Not rated"
