"""Shared fixtures for femlab tests."""

import pytest
import warp as wp


@pytest.fixture(scope="session", autouse=True)
def warp_initialized():
    """Ensure Warp is initialized once for the test session."""
    wp.init()
    yield


@pytest.fixture
def device():
    """Return the current Warp device string."""
    return str(wp.get_device())
