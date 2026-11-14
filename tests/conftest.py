"""Shared pytest configuration: the `slow` marker (full plan counts) runs only with FULL=1."""

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: full plan-count checks; run with FULL=1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("FULL") == "1":
        return
    skip = pytest.mark.skip(reason="slow: set FULL=1 to run the full plan counts")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
