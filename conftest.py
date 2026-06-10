"""Pytest fixture to disable Home Assistant custom integration loading."""

import homeassistant.helpers.llm as _llm_mod

if not hasattr(_llm_mod, "selector_serializer") and hasattr(
    _llm_mod, "_selector_serializer"
):
    _llm_mod.selector_serializer = _llm_mod._selector_serializer  # noqa: SLF001

from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def enable_custom_integrations() -> Generator[None, None, None]:
    """Override HA-CC plugin's enable_custom_integrations so it does nothing."""
    return
