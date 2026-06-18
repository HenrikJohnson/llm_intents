"""Tests for the date info tool."""

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.llm_intents.date_info import DateInfoTool


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
def date_tool(mock_hass: HomeAssistant) -> DateInfoTool:
    """Create DateInfoTool instance."""
    return DateInfoTool({}, mock_hass)


async def test_date_info_known_date(
    mock_hass: HomeAssistant, date_tool: DateInfoTool
) -> None:
    """Test a known date returns correct day of week."""
    tool_input = llm.ToolInput(
        tool_name="calendar_day_info",
        tool_args={"day": 25, "month": 12, "year": 2024},
    )
    result = await date_tool.async_call(mock_hass, tool_input, _make_llm_context())

    assert result["day"] == "Wednesday"
    assert result["date"] == "December 25, 2024"
    assert "Wednesday" in result["message"]


async def test_date_info_leap_year(
    mock_hass: HomeAssistant, date_tool: DateInfoTool
) -> None:
    """Test Feb 29 on a leap year works."""
    tool_input = llm.ToolInput(
        tool_name="calendar_day_info",
        tool_args={"day": 29, "month": 2, "year": 2024},
    )
    result = await date_tool.async_call(mock_hass, tool_input, _make_llm_context())

    assert result["day"] == "Thursday"
    assert "February 29" in result["date"]


async def test_date_info_invalid_date(
    mock_hass: HomeAssistant, date_tool: DateInfoTool
) -> None:
    """Test invalid date returns error."""
    tool_input = llm.ToolInput(
        tool_name="calendar_day_info",
        tool_args={"day": 31, "month": 2, "year": 2024},
    )
    result = await date_tool.async_call(mock_hass, tool_input, _make_llm_context())

    assert "error" in result
    assert "Invalid date" in result["error"]


async def test_date_info_defaults_to_current_year(
    mock_hass: HomeAssistant, date_tool: DateInfoTool
) -> None:
    """Test that year defaults to the current year when not provided."""
    tool_input = llm.ToolInput(
        tool_name="calendar_day_info",
        tool_args={"day": 1, "month": 1},
    )
    result = await date_tool.async_call(mock_hass, tool_input, _make_llm_context())

    assert "day" in result
    assert "date" in result
    assert "message" in result


async def test_date_info_new_years_2000(
    mock_hass: HomeAssistant, date_tool: DateInfoTool
) -> None:
    """Test Y2K date."""
    tool_input = llm.ToolInput(
        tool_name="calendar_day_info",
        tool_args={"day": 1, "month": 1, "year": 2000},
    )
    result = await date_tool.async_call(mock_hass, tool_input, _make_llm_context())

    assert result["day"] == "Saturday"
    assert result["date"] == "January 01, 2000"
