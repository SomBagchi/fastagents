"""
Common TypedDicts and type aliases shared across env/tool implementations.

These TypedDict classes are available at runtime for type checking and validation.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple, TypedDict

# ── Types available at runtime ──────────────────────────────────────
# Alias for the standard Gym-style step return:
StepReturn = Tuple["BashObs", float, bool, Dict[str, Any]]


class BashAction(TypedDict):
    """Action schema: one Bash command to execute in the sandbox."""

    tool: Literal["bash"]
    cmd: str


class BashObs(TypedDict):
    """Observation schema returned by BashTool.step()."""

    stdout: str
    stderr: str
    exit: int  # process exit-code (0 = success)


# Public export surface ------------------------------------------------
__all__ = [
    "StepReturn",
    "BashAction",
    "BashObs",
]
