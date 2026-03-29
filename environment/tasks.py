from __future__ import annotations

from typing import Any, Dict, List


TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "name": "easy",
        "seed": 42,
        "horizon": 7,
        "products": ["Widget"],
        "initial_inventory": {"Widget": 10},
        "initial_cash": 120.0,
        "fixed_demand": {"Widget": 2.5},
        "demand_elasticity": {"Widget": 1.0},
        "seasonality": {"Widget": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
        "product_costs": {"Widget": 5.0},
        "holding_costs": {"Widget": 0.08},
        "max_inventory": {"Widget": 120},
        "lead_time": 0,
        "order_fixed_fee": 0.75,
        "price_bounds": {"Widget": {"min_multiplier": 1.10, "max_multiplier": 1.90}},
        "reference_prices": {"Widget": 7.5},
        "initial_prices": {"Widget": 7.5},
        "target_profit": 70.0,
        "baseline_profit": 60.0,
        "price_volatility_penalty_weight": 0.05,
        "invalid_action_penalty": 4.0,
        "loop_penalty": 2.5,
        "service_penalty": 0.8,
        "grading_seeds": [42],
    },
    "medium": {
        "name": "medium",
        "seed": 123,
        "horizon": 14,
        "products": ["Stapler", "Notebook"],
        "initial_inventory": {"Stapler": 6, "Notebook": 10},
        "initial_cash": 240.0,
        "demand_ranges": {"Stapler": [1.8, 3.0], "Notebook": [2.4, 3.8]},
        "demand_elasticity": {"Stapler": 0.9, "Notebook": 1.2},
        "seasonality": {
            "Stapler": [1.0, 1.1, 1.0, 0.95, 1.0, 1.05, 1.0, 0.9, 1.0, 1.15, 1.0, 0.95, 1.0, 1.05],
            "Notebook": [1.0, 0.95, 1.0, 1.1, 1.0, 1.05, 1.15, 1.0, 0.9, 1.0, 1.05, 1.0, 1.1, 1.0],
        },
        "product_costs": {"Stapler": 6.0, "Notebook": 3.5},
        "holding_costs": {"Stapler": 0.12, "Notebook": 0.10},
        "max_inventory": {"Stapler": 35, "Notebook": 45},
        "lead_time": 1,
        "order_fixed_fee": 2.0,
        "price_bounds": {
            "Stapler": {"min_multiplier": 1.10, "max_multiplier": 2.00},
            "Notebook": {"min_multiplier": 1.10, "max_multiplier": 2.10},
        },
        "reference_prices": {"Stapler": 9.2, "Notebook": 5.6},
        "initial_prices": {"Stapler": 9.2, "Notebook": 5.6},
        "target_profit": 180.0,
        "baseline_profit": 150.0,
        "price_volatility_penalty_weight": 0.08,
        "invalid_action_penalty": 6.0,
        "loop_penalty": 3.0,
        "service_penalty": 1.2,
        "grading_seeds": [123],
    },
    "hard": {
        "name": "hard",
        "seed": 999,
        "horizon": 30,
        "products": ["Cable", "Mouse", "Keyboard"],
        "initial_inventory": {"Cable": 8, "Mouse": 6, "Keyboard": 4},
        "initial_cash": 360.0,
        "demand_ranges": {"Cable": [3.2, 4.8], "Mouse": [1.9, 2.8], "Keyboard": [1.3, 2.0]},
        "demand_elasticity": {"Cable": 1.4, "Mouse": 1.1, "Keyboard": 0.8},
        "seasonality": {
            "Cable": [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.15, 1.05, 1.0, 0.95, 0.9, 0.88, 0.92, 1.0, 1.08, 1.15, 1.2, 1.25, 1.3, 1.22, 1.12, 1.0, 0.95, 0.9, 0.88, 0.9, 0.95, 1.0],
            "Mouse": [1.0, 0.98, 0.96, 0.95, 0.93, 0.92, 0.9, 0.88, 0.9, 0.95, 1.0, 1.05, 1.08, 1.1, 1.12, 1.15, 1.18, 1.2, 1.22, 1.2, 1.15, 1.1, 1.05, 1.0, 0.96, 0.94, 0.92, 0.95, 0.98, 1.0],
            "Keyboard": [1.05, 1.08, 1.1, 1.12, 1.15, 1.18, 1.2, 1.18, 1.15, 1.12, 1.1, 1.08, 1.05, 1.02, 1.0, 0.98, 0.96, 0.95, 0.94, 0.95, 0.97, 1.0, 1.03, 1.06, 1.1, 1.12, 1.14, 1.12, 1.08, 1.05],
        },
        "product_costs": {"Cable": 4.0, "Mouse": 12.0, "Keyboard": 18.0},
        "holding_costs": {"Cable": 0.09, "Mouse": 0.18, "Keyboard": 0.25},
        "max_inventory": {"Cable": 60, "Mouse": 30, "Keyboard": 24},
        "lead_time": 2,
        "order_fixed_fee": 3.5,
        "price_bounds": {
            "Cable": {"min_multiplier": 1.08, "max_multiplier": 2.20},
            "Mouse": {"min_multiplier": 1.10, "max_multiplier": 2.10},
            "Keyboard": {"min_multiplier": 1.12, "max_multiplier": 2.00},
        },
        "reference_prices": {"Cable": 7.2, "Mouse": 18.0, "Keyboard": 27.0},
        "initial_prices": {"Cable": 7.2, "Mouse": 18.0, "Keyboard": 27.0},
        "target_profit": 520.0,
        "baseline_profit": 430.0,
        "price_volatility_penalty_weight": 0.12,
        "invalid_action_penalty": 8.0,
        "loop_penalty": 4.0,
        "service_penalty": 1.5,
        "grading_seeds": [999],
    },
}

# Backward-compatible aliases
TASK_PRESETS: Dict[str, Dict[str, Any]] = {
    **TASKS,
    "easy_single_product": TASKS["easy"],
    "medium_two_product_variable_demand": TASKS["medium"],
    "hard_multi_product_nonstationary": TASKS["hard"],
}


def get_task_config(task_name: str) -> Dict[str, Any]:
    if task_name not in TASK_PRESETS:
        raise KeyError(f"Unknown task preset: {task_name}")

    preset = TASK_PRESETS[task_name]
    # Return a shallow-cloned config with copied nested structures for safe mutation
    cloned: Dict[str, Any] = {}
    for key, value in preset.items():
        if isinstance(value, dict):
            cloned[key] = dict(value)
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value

    # Deep-copy nested dicts where necessary
    if "price_bounds" in cloned:
        cloned["price_bounds"] = {
            product: dict(bounds) for product, bounds in cloned["price_bounds"].items()
        }
    if "seasonality" in cloned:
        cloned["seasonality"] = {
            product: list(values) for product, values in cloned["seasonality"].items()
        }

    return cloned


def list_tasks() -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for name, cfg in TASK_PRESETS.items():
        tasks.append(
            {
                "name": name,
                "horizon": int(cfg.get("horizon", 30)),
                "products": list(cfg.get("products", [])),
                "lead_time": int(cfg.get("lead_time", 0)),
                "order_fixed_fee": float(cfg.get("order_fixed_fee", 0.0)),
                "grading_seeds": list(cfg.get("grading_seeds", [])),
            }
        )
    return tasks


def get_task_seed_set(task_name: str) -> List[int]:
    config = get_task_config(task_name)
    return [int(seed) for seed in config.get("grading_seeds", [])]
