import pytest


_PYRAMID_LEVELS = (
    "unit",
    "component",
    "integration",
    "event_driven",
    "e2e",
    "replay",
    "chaos",
    "performance",
)


def pytest_collection_modifyitems(items):
    for item in items:
        if not any(item.get_closest_marker(level) for level in _PYRAMID_LEVELS):
            item.add_marker(pytest.mark.unit)