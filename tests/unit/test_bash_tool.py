"""
Unit tests for the persistent-container BashTool.
Run with:  pytest -q
"""

from typing import Generator

import pytest
from fastagents.envs import BashTool


@pytest.fixture(scope="function")  # type: ignore[misc]
def env() -> Generator[BashTool, None, None]:
    """Spin up a fresh BashTool and tear it down after each test."""
    tool = BashTool(timeout=2)
    tool.reset()  # start the container
    yield tool
    tool.close()  # ensure cleanup


def test_echo_ok(env: BashTool) -> None:
    """Basic happy-path: command succeeds, output matches, reward==0."""
    obs, reward, done, info = env.step({"tool": "bash", "cmd": "echo 42"})
    assert obs["exit"] == 0
    assert obs["stdout"].strip() == "42"
    assert reward == 0.0
    assert done is False
    assert not info.get("truncated")


def test_timeout(env: BashTool) -> None:
    """Timeout handling: a sleep longer than env.timeout should exit -1."""
    obs, *_ = env.step({"tool": "bash", "cmd": "sleep 5"})
    assert obs["exit"] == -1
    assert "TIMEOUT" in obs["stderr"]


def test_truncation(env: BashTool) -> None:
    """Stdout longer than 4 KB is truncated and flagged."""
    long_text = "x" * (5 * 1024)
    obs, _, _, info = env.step({"tool": "bash", "cmd": f"printf '{long_text}'"})
    assert len(obs["stdout"]) == 4 * 1024
    assert info["truncated"] is True
