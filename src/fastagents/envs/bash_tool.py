"""
BashTool — persistent-container variant
• reset()  → spins up one detached container
• step()   → docker exec into that container for every Bash command
• close()  → force-removes the container

Stage-0: reward = 0.0, done = False, 4 KB stdout/err cap
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict

import docker
from docker import DockerClient  # type: ignore
from docker.models.containers import Container

from ._types import BashAction, BashObs, StepReturn

KB = 4 * 1024  # 4 KiB cap for stdout/stderr


class BashTool:
    """Gym-style wrapper around **one** long-lived Docker bash session."""

    def __init__(self, image: str = "fastagents-bash:0.1", timeout: int = 2) -> None:
        self.image = image
        self.timeout = timeout
        self._client: DockerClient | None = None
        self._ctr: Container | None = None

    # ─────────────────────────────────────────────────────
    # Episode control
    # ─────────────────────────────────────────────────────
    def reset(self) -> Dict[str, Any]:
        """Start a new episode by spinning up a detached container."""
        self.close()  # kill any prior episode
        self._client = docker.from_env()  # type: ignore
        self._ctr = self._client.containers.run(
            self.image,
            detach=True,
            tty=True,  # allocate a pseudo-TTY
            user="agent",  # stay non-root
            working_dir="/workspace",
        )
        return {}  # no initial observation yet

    def close(self) -> None:
        """Tear down the container (called manually or via __del__)."""
        if self._ctr is not None:
            try:
                self._ctr.remove(force=True)
            except docker.errors.APIError:  # type: ignore
                pass
            self._ctr = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        self.close()

    # ─────────────────────────────────────────────────────
    # Core step
    # ─────────────────────────────────────────────────────
    def step(self, action: BashAction) -> StepReturn:
        assert action["tool"] == "bash", f"Unsupported tool: {action}"

        # Allow users to call step() without an explicit reset()
        if self._ctr is None:
            self.reset()

        assert self._ctr is not None  # type narrowing
        ctr_id = self._ctr.id  # persistent container ID
        cmd_str = action["cmd"]

        try:
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-u",
                    "agent",  # run as the non-root user
                    "-w",
                    "/workspace",  # working dir inside container
                    ctr_id,
                    "bash",
                    "-lc",
                    cmd_str,
                ],
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )
            obs: BashObs = {
                "stdout": proc.stdout[:KB],
                "stderr": proc.stderr[:KB],
                "exit": proc.returncode,
            }
            info = {
                "truncated": len(proc.stdout) > KB or len(proc.stderr) > KB,
            }

        except subprocess.TimeoutExpired as e:
            # Handle the case where stdout might be bytes or str
            stdout_str = ""
            if e.stdout is not None:
                if isinstance(e.stdout, bytes):
                    stdout_str = e.stdout.decode("utf-8", errors="replace")
                else:
                    stdout_str = e.stdout

            timeout_obs: BashObs = {
                "stdout": stdout_str[:KB],
                "stderr": "TIMEOUT",
                "exit": -1,
            }
            obs = timeout_obs
            info = {"timeout": True}

        reward: float = 0.0
        done: bool = False
        return obs, reward, done, info
