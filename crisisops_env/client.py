"""Client helpers for interacting with a remote CrisisOps environment."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from .models import Action, BuddyAction, Observation, State

try:
    from openenv.core.env_client import EnvClient
except ImportError:
    EnvClient = None  # type: ignore[assignment]


if EnvClient is not None:

    class CrisisOpsClient(EnvClient[BuddyAction, Observation, State]):  # type: ignore[misc]
        """OpenEnv WebSocket client for CrisisOps when openenv-core is installed."""

        pass

else:

    class CrisisOpsClient:
        """Small HTTP fallback client used before openenv-core is installed."""

        def __init__(self, base_url: str) -> None:
            """Store the base URL for a running CrisisOps server."""

            self.base_url = base_url.rstrip("/")

        def reset(self, **kwargs: Any) -> Observation:
            """Reset the remote environment over HTTP."""

            response = requests.post(f"{self.base_url}/reset", json=kwargs, timeout=30)
            response.raise_for_status()
            payload = response.json()
            return Observation.model_validate(payload["observation"])

        def step(self, action: Action | BuddyAction | Dict[str, Any]) -> Observation:
            """Take one remote environment step over HTTP."""

            if isinstance(action, (Action, BuddyAction)):
                action_model = action
            elif "primary_action" in action:
                action_model = BuddyAction.model_validate(action)
            else:
                action_model = Action.model_validate(action)
            response = requests.post(
                f"{self.base_url}/step",
                json={"action": action_model.model_dump()},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            return Observation.model_validate(payload["observation"])

        def state(self) -> State:
            """Fetch full remote state over HTTP."""

            response = requests.get(f"{self.base_url}/state", timeout=30)
            response.raise_for_status()
            return State.model_validate(response.json())

        def close(self) -> None:
            """Match the OpenEnv client close API for local scripts."""

            return None
