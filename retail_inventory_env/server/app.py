import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from environment.grader import evaluate_seeded_summaries, score_episode
from environment.models import ActionType, NoOpAction, OrderAction, RetailAction, SetPriceAction
from environment.retail_env import RetailInventoryEnv
from environment.tasks import get_task_config, list_tasks

app = FastAPI(
    title="Retail Inventory Env",
    description="OpenEnv compliant retail inventory management environment",
)

# Global environment instance
env = RetailInventoryEnv()


class ResetRequest(BaseModel):
    task_config: Optional[Dict[str, Any]] = None
    task_name: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class EvaluateRequest(BaseModel):
    summary: Optional[Dict[str, Any]] = None
    summaries: Optional[List[Dict[str, Any]]] = None
    variance_penalty: float = Field(default=0.0, ge=0.0)


class StateResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


def _parse_action(action_payload: Dict[str, Any]) -> RetailAction:
    if not isinstance(action_payload, dict):
        raise ValueError("Action payload must be an object")

    action_type = action_payload.get("action")
    if action_type is None:
        raise ValueError("Missing action field")

    action_value = str(action_type).strip().lower()

    try:
        if action_value == ActionType.ORDER.value:
            return OrderAction(**action_payload)
        if action_value == ActionType.SET_PRICE.value:
            return SetPriceAction(**action_payload)
        if action_value == ActionType.NO_OP.value:
            return NoOpAction(**action_payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid payload for action '{action_value}': {exc}") from exc

    raise ValueError(f"Unsupported action type: {action_type}")


def _resolve_task_config(request: ResetRequest) -> Dict[str, Any]:
    if request.task_config is not None:
        return request.task_config
    if request.task_name is not None:
        return get_task_config(request.task_name)
    raise ValueError("Either task_config or task_name must be provided")


def _build_current_episode_summary() -> Dict[str, Any]:
    if env.state is None:
        raise RuntimeError("Environment not initialized. Call /reset first.")

    summary = {k: float(v) for k, v in env.episode_metrics.items()}

    inventory_value = 0.0
    max_capacity = 0.0
    used_capacity = 0.0
    for product, qty in env.state.inventory.items():
        inventory_value += float(qty) * float(env.state.product_costs[product])
        cap = float(env.state.max_inventory[product])
        if cap != float("inf"):
            max_capacity += cap
            used_capacity += float(qty)

    if max_capacity > 0:
        summary["ending_inventory_ratio"] = min(1.0, used_capacity / max_capacity)
    else:
        summary["ending_inventory_ratio"] = 0.0

    summary["profit"] = float(env.state.cash) + inventory_value - float(env.initial_cash)
    summary["horizon"] = float(getattr(env, "_horizon", summary.get("horizon", 30.0)))
    summary["num_products"] = float(len(env.state.inventory))

    return summary


@app.post("/reset")
async def reset_environment(request: ResetRequest):
    """Reset the environment with the given task configuration."""
    try:
        if request.seed is not None:
            env.seed = request.seed

        task_config = _resolve_task_config(request)
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
    """Get safe internal state (masks hidden variables)."""
    try:
        state = env.get_state()
        return {"state": state}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
async def get_tasks():
    """List deterministic built-in task presets."""
    try:
        return {"tasks": list_tasks()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/evaluate")
async def evaluate_episode(request: EvaluateRequest):
    """Return deterministic grader output for one or many episode summaries."""
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

        summary = _build_current_episode_summary()
        result = score_episode(summary)
        return {
            "mode": "current_episode",
            "summary": summary,
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
