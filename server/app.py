"""FastAPI server for multi-channel retail environment."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from environment.grader import evaluate_seeded_summaries, score_episode
from environment.models import (
    ActionType,
    AllocateAction,
    NoOpAction,
    OrderAction,
    PromoteAction,
    RetailAction,
    SetPriceAction,
)
from environment.retail_env import MultiChannelRetailEnv
from environment.tasks import get_task_config, list_tasks

app = FastAPI(
    title="Multi-Channel Retail Environment",
    description="OpenEnv compliant multi-channel retail with disruption recovery",
)

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"

# Global environment instance
env = MultiChannelRetailEnv()


class ResetRequest(BaseModel):
    task_config: Optional[Dict[str, Any]] = None
    task_name: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class EvaluateRequest(BaseModel):
    summary: Optional[Dict[str, Any]] = None
    summaries: Optional[List[Dict[str, Any]]] = None
    variance_penalty: float = 0.0


def _parse_action(payload: Dict[str, Any]) -> RetailAction:
    """Parse action from JSON payload."""
    if not isinstance(payload, dict):
        raise ValueError("Action must be a JSON object")

    action_type = payload.get("action")
    if action_type is None:
        raise ValueError("Missing 'action' field")

    action_value = str(action_type).strip().lower()

    try:
        if action_value == ActionType.ALLOCATE.value:
            return AllocateAction(**payload)
        if action_value == ActionType.SET_PRICE.value:
            return SetPriceAction(**payload)
        if action_value == ActionType.ORDER.value:
            return OrderAction(**payload)
        if action_value == ActionType.PROMOTE.value:
            return PromoteAction(**payload)
        if action_value == ActionType.NOOP.value:
            return NoOpAction(**payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid action: {exc}") from exc

    raise ValueError(f"Unknown action type: {action_type}")


def _resolve_task_config(request: ResetRequest) -> Dict[str, Any]:
    """Resolve task configuration."""
    if request.task_config is not None:
        return request.task_config
    if request.task_name is not None:
        return get_task_config(request.task_name)
    raise ValueError("Either task_config or task_name must be provided")


@app.post("/reset")
async def reset_environment(request: ResetRequest):
    """Reset the environment."""
    try:
        task_config = _resolve_task_config(request)
        if request.seed is not None:
            env.seed = int(request.seed)
        elif "seed" in task_config:
            env.seed = int(task_config["seed"])

        observation = env.reset(task_config)

        return {
            "observation": observation.model_dump(),
            "reward": 0.0,
            "done": False,
            "info": {
                "task_name": task_config.get("name", request.task_name),
                "horizon": int(task_config.get("horizon", 30)),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step_environment(request: StepRequest):
    """Take a step in the environment."""
    try:
        action = _parse_action(request.action)
        observation, reward, done, info = env.step(action)

        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def get_state():
    """Get current internal state."""
    try:
        state = env.get_state()
        return {"state": state}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
async def get_tasks_list():
    """List available tasks."""
    try:
        return {"tasks": list_tasks()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/evaluate")
async def evaluate_episode(request: EvaluateRequest):
    """Evaluate episode(s)."""
    try:
        if request.summaries is not None:
            result = evaluate_seeded_summaries(
                request.summaries,
                variance_penalty=request.variance_penalty,
            )
            return {
                "mode": "multi_seed",
                "result": result,
            }

        if request.summary is not None:
            result = score_episode(request.summary)
            return {
                "mode": "single_summary",
                "result": result,
            }

        raise ValueError("Either summary or summaries must be provided")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check."""
    return {"status": "healthy"}


@app.get("/")
async def ui_home():
    """Serve the terminal UI."""
    ui_file = UI_DIR / "index.html"
    if not ui_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(ui_file)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
