"""FastAPI server for multi-channel retail environment."""
import asyncio
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

from environment.grader import evaluate_seeded_summaries, score_episode
from environment.models import (
    ActionType,
    AllocateAction,
    CompositeAction,
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


class LiveStartRequest(BaseModel):
    task_name: Optional[str] = None
    seed: Optional[int] = None
    mode: str = "heuristic"  # heuristic | noop
    interval_ms: int = 600


class LiveStopRequest(BaseModel):
    reason: Optional[str] = None


def _parse_action(payload: Dict[str, Any]) -> RetailAction:
    """Parse action from JSON payload.

    Supports both legacy single actions and the new CompositeAction
    format that allows multiple actions per timestep.
    """
    if not isinstance(payload, dict):
        raise ValueError("Action must be a JSON object")

    action_type = payload.get("action")
    if action_type is None:
        raise ValueError("Missing 'action' field")

    action_value = str(action_type).strip().lower()

    try:
        if action_value == ActionType.COMPOSITE.value:
            return CompositeAction(**payload)
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


env_lock = threading.Lock()


class LiveRunner:
    """Realtime step runner for continuous wall-clock execution."""

    def __init__(self, env_ref, lock_ref) -> None:
        self.env = env_ref
        self.env_lock = lock_ref
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._mode = "heuristic"
        self._interval_s = 0.6
        self._tick = 0
        self._latest: Dict[str, Any] = {
            "tick": 0,
            "running": False,
            "done": False,
            "reward": 0.0,
            "observation": None,
            "info": {},
            "timestamp": 0.0,
            "error": None,
        }
        self._status_lock = threading.Lock()

    def _heuristic_action(self, observation: Any) -> Dict[str, Any]:
        products = list(observation.inventory.keys())
        if not products:
            return {"action": "noop"}

        target = min(products, key=lambda p: int(observation.inventory.get(p, 0)))
        min_inv = int(observation.inventory.get(target, 0))
        if min_inv <= 2:
            return {"action": "order", "product": target, "quantity": 3}

        if getattr(observation, "disruption_active", False) and float(observation.cash) > 20.0:
            return {"action": "promote", "product": target, "budget_allocated": 8.0}

        return {"action": "noop"}

    def _run_loop(self) -> None:
        while self._running:
            try:
                with self.env_lock:
                    if self.env.state is None:
                        self._set_latest(error="Environment not initialized. Use /live/start with task_name.")
                        time.sleep(self._interval_s)
                        continue

                    obs = self.env._get_observation()
                    if self._mode == "noop":
                        action_payload = {"action": "noop"}
                    else:
                        action_payload = self._heuristic_action(obs)

                    action = _parse_action(action_payload)
                    observation, reward, done, info = self.env.step(action)

                self._tick += 1
                self._set_latest(
                    tick=self._tick,
                    running=True,
                    done=bool(done),
                    reward=float(reward),
                    observation=observation.model_dump(),
                    info=info,
                    timestamp=time.time(),
                    error=None,
                )

                if done:
                    self._running = False
                    self._set_latest(running=False)
                    break

            except Exception as exc:
                self._set_latest(error=str(exc), running=False)
                self._running = False
                break

            time.sleep(self._interval_s)

    def _set_latest(self, **kwargs: Any) -> None:
        with self._status_lock:
            self._latest.update(kwargs)

    def start(self, mode: str, interval_ms: int) -> None:
        if self._running:
            return
        self._mode = mode if mode in {"heuristic", "noop"} else "heuristic"
        self._interval_s = max(0.1, float(interval_ms) / 1000.0)
        self._running = True
        self._set_latest(running=True, error=None)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, reason: Optional[str] = None) -> None:
        self._running = False
        self._set_latest(running=False, info={"reason": reason or "stopped"})

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            return {
                "running": bool(self._running),
                "mode": self._mode,
                "interval_ms": int(self._interval_s * 1000),
                "tick": int(self._latest.get("tick", 0)),
                "done": bool(self._latest.get("done", False)),
                "timestamp": float(self._latest.get("timestamp", 0.0)),
                "error": self._latest.get("error"),
            }

    def latest(self) -> Dict[str, Any]:
        with self._status_lock:
            return dict(self._latest)


live_runner = LiveRunner(env, env_lock)


@app.post("/reset")
async def reset_environment(request: ResetRequest):
    """Reset the environment."""
    try:
        with env_lock:
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
        with env_lock:
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
    """Get current internal state and episode metrics."""
    try:
        with env_lock:
            state = env.get_state()
            metrics = dict(env.episode_metrics)
        return {"state": state, "episode_metrics": metrics}
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


@app.post("/live/start")
async def live_start(request: LiveStartRequest):
    """Start realtime continuous stepping."""
    try:
        if request.task_name:
            with env_lock:
                task_cfg = get_task_config(request.task_name)
                if request.seed is not None:
                    env.seed = int(request.seed)
                elif "seed" in task_cfg:
                    env.seed = int(task_cfg["seed"])
                env.reset(task_cfg)

        live_runner.start(mode=request.mode, interval_ms=int(request.interval_ms))
        return {
            "ok": True,
            "status": live_runner.status(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/live/stop")
async def live_stop(request: LiveStopRequest):
    """Stop realtime stepping."""
    live_runner.stop(request.reason)
    return {
        "ok": True,
        "status": live_runner.status(),
    }


@app.get("/live/status")
async def live_status():
    """Get realtime runner status."""
    return live_runner.status()


@app.get("/live/latest")
async def live_latest():
    """Get latest realtime tick payload."""
    return live_runner.latest()


@app.get("/live/stream")
async def live_stream():
    """Server-Sent Events stream for realtime live runner updates.

    Clients receive a ``data:`` event containing a JSON-encoded payload
    identical to ``/live/latest`` every 500 ms while the connection is open.
    This removes the need for client-side polling.
    """

    async def _event_generator():
        try:
            while True:
                data = live_runner.latest()
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.5)
        except GeneratorExit:
            return
        except Exception:
            # Yield a generic error event so connected clients can react, then close.
            yield f"data: {json.dumps({'error': 'stream error'})}\n\n"
            return

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/health")
async def health_check():
    """Health check."""
    return {"status": "healthy"}


@app.get("/dashboard")
async def ui_dashboard():
    """Serve the modern visual dashboard UI."""
    ui_file = UI_DIR / "dashboard.html"
    if not ui_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(ui_file, media_type="text/html")


@app.get("/static/chart.js")
async def serve_chartjs():
    """Serve bundled Chart.js for the dashboard."""
    js_file = UI_DIR / "chart.umd.min.js"
    if not js_file.exists():
        raise HTTPException(status_code=404, detail="Chart.js not found")
    return FileResponse(js_file, media_type="application/javascript")


@app.get("/")
async def ui_home():
    """Serve the terminal UI."""
    ui_file = UI_DIR / "index.html"
    if not ui_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(ui_file)


def main():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
