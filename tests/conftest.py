"""Shared pytest configuration for Biomni tests."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that call live LLM APIs (requires API keys).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: marks tests that call live LLM APIs (skip unless --live)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return  # --live given: run everything
    skip_live = pytest.mark.skip(reason="needs --live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
