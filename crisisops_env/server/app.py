"""FastAPI app for serving CrisisOps through OpenEnv or local HTTP."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI

from crisisops_env.env import CrisisOpsEnv
from crisisops_env.models import Action, BuddyAction, Observation

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    create_app = None  # type: ignore[assignment]


if create_app is not None:
    app = create_app(
        CrisisOpsEnv,
        BuddyAction,
        Observation,
        env_name="crisisops_env",
    )
else:
    app = FastAPI(title="CrisisOps Env", version="0.1.0")
    _env = CrisisOpsEnv()

    @app.get("/health")
    def health() -> Dict[str, str]:
        """Return local server health."""

        return {"status": "healthy"}

    @app.post("/reset")
    def reset(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Reset the local fallback environment."""

        obs = _env.reset(**(payload or {}))
        return {
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.post("/step")
    def step(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Step the local fallback environment."""

        raw_action = payload["action"]
        action = (
            BuddyAction.model_validate(raw_action)
            if "primary_action" in raw_action
            else Action.model_validate(raw_action)
        )
        obs = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.get("/state")
    def state() -> Dict[str, Any]:
        """Return local fallback environment state."""

        return _env.state.model_dump()

    @app.get("/schema")
    def schema() -> Dict[str, Any]:
        """Return model schemas for local fallback clients."""

        return {
            "action": BuddyAction.model_json_schema(),
            "legacy_action": Action.model_json_schema(),
            "observation": Observation.model_json_schema(),
        }


def main() -> None:
    """Run the CrisisOps server with Uvicorn."""

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
