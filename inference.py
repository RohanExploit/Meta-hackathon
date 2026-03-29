import json
import os
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from environment.tasks import TASKS


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

TEMPERATURE = 0.0
MAX_TOKENS = 200
REQUEST_TIMEOUT = 30


SYSTEM_PROMPT = (
    "You manage a retail inventory environment. Return exactly one JSON object action. "
    "Allowed actions: "
    "{\"action\":\"order\",\"product\":str,\"quantity\":int}, "
    "{\"action\":\"set_price\",\"product\":str,\"new_price\":float}, "
    "{\"action\":\"noop\"}. "
    "Never add extra text."
)


def _safe_json_dict(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _sanitize_action(action: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    products = list(observation.get("inventory", {}).keys())
    if not products:
        return {"action": "noop"}

    cash = float(observation.get("cash", 0.0))
    prices = observation.get("price", {}) or {}

    a = str(action.get("action", "noop")).strip().lower()
    if a == "order":
        product = str(action.get("product", products[0]))
        if product not in products:
            product = products[0]

        quantity = action.get("quantity", 0)
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            quantity = 0

        if quantity <= 0:
            return {"action": "noop"}

        unit_cost_proxy = max(0.1, float(prices.get(product, 1.0)) * 0.65)
        max_affordable = int(cash // unit_cost_proxy)
        if max_affordable <= 0:
            return {"action": "noop"}

        quantity = max(1, min(quantity, max_affordable, 25))
        return {"action": "order", "product": product, "quantity": quantity}

    if a == "set_price":
        product = str(action.get("product", products[0]))
        if product not in products:
            product = products[0]

        try:
            new_price = float(action.get("new_price", prices.get(product, 1.0)))
        except (TypeError, ValueError):
            return {"action": "noop"}

        if new_price <= 0:
            return {"action": "noop"}

        return {"action": "set_price", "product": product, "new_price": round(new_price, 2)}

    return {"action": "noop"}


def _build_user_prompt(task_name: str, step: int, observation: Dict[str, Any], history: List[str]) -> str:
    inv = observation.get("inventory", {})
    sales = observation.get("sales_history", {})
    prices = observation.get("price", {})

    lines: List[str] = []
    lines.append(f"Task: {task_name}")
    lines.append(f"Day: {observation.get('day', 0)}")
    lines.append(f"Cash: {float(observation.get('cash', 0.0)):.2f}")
    lines.append("Products:")
    for p in inv.keys():
        lines.append(
            f"- {p}: inventory={int(inv.get(p, 0))}, last_sales={int(sales.get(p, 0))}, price={float(prices.get(p, 0.0)):.2f}"
        )
    lines.append(f"Recent history: {' | '.join(history[-4:]) if history else 'none'}")
    lines.append(
        f"Step {step}: choose one valid action JSON. Prefer avoiding stockouts and avoid overspending."
    )
    return "\n".join(lines)


def _call_model_action(client: OpenAI, task_name: str, step: int, observation: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    user_prompt = _build_user_prompt(task_name, step, observation, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = completion.choices[0].message.content or ""
    except Exception:
        return {"action": "noop"}

    parsed = _safe_json_dict(text)
    if parsed is None:
        return {"action": "noop"}
    return _sanitize_action(parsed, observation)


def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{ENV_BASE_URL}{path}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def run_task(client: OpenAI, task_name: str) -> Dict[str, Any]:
    task_cfg = TASKS[task_name]
    reset_payload = {"task_name": task_name, "seed": int(task_cfg["seed"])}
    reset_out = _post_json("/reset", reset_payload)

    observation = reset_out["observation"]
    done = bool(reset_out.get("done", False))
    total_reward = 0.0
    history: List[str] = []
    final_info: Dict[str, Any] = {}

    max_steps = int(task_cfg["horizon"])
    for step in range(1, max_steps + 1):
        if done:
            break

        action = _call_model_action(client, task_name, step, observation, history)

        try:
            step_out = _post_json("/step", {"action": action})
        except Exception:
            step_out = _post_json("/step", {"action": {"action": "noop"}})
            action = {"action": "noop"}

        observation = step_out["observation"]
        reward = float(step_out.get("reward", 0.0))
        done = bool(step_out.get("done", False))
        info = step_out.get("info", {}) or {}

        total_reward += reward
        history.append(f"step={step}, action={action}, reward={reward:.3f}, done={done}")
        final_info = info

    grader = (final_info.get("grader") or {}) if isinstance(final_info, dict) else {}

    return {
        "task": task_name,
        "total_reward": total_reward,
        "steps_executed": len(history),
        "score": float(grader.get("score", 0.0)),
        "grader": grader,
    }


def main() -> None:
    if not MODEL_NAME:
        raise ValueError("MODEL_NAME is required")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN or API_KEY is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    task_order = ["easy", "medium", "hard"]
    results: List[Dict[str, Any]] = []
    for task_name in task_order:
        result = run_task(client, task_name)
        results.append(result)
        print(
            f"Task={task_name} score={result['score']:.4f} "
            f"reward={result['total_reward']:.3f} steps={result['steps_executed']}"
        )

    mean_score = sum(r["score"] for r in results) / max(1, len(results))
    print(f"Mean score across tasks: {mean_score:.4f}")
    print(json.dumps({"results": results, "mean_score": mean_score}, indent=2))


if __name__ == "__main__":
    main()
