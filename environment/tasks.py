"""Five progressively harder tasks for multi-channel retail."""
from typing import Any, Dict, List


TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "name": "easy",
        "seed": 42,
        "horizon": 10,
        "description": "Single product, stable demand, no disruptions",
        "products": ["Product_A"],
        "initial_inventory": {"Product_A": 20},
        "initial_cash": 200.0,
        "base_demand_luxury": {"Product_A": 1.0},
        "base_demand_budget": {"Product_A": 2.0},
        "demand_elasticity": {"Product_A": 1.0},
        "product_costs": {"Product_A": 5.0},
        "holding_costs": {"Product_A": 0.05},
        "max_inventory": {"Product_A": 100.0},
        "lead_time_mean": 1,
        "lead_time_variance": 0,
        "supplier_reliability": 1.0,
        "initial_prices_luxury": {"Product_A": 12.0},
        "initial_prices_budget": {"Product_A": 8.0},
        "price_bounds": {"Product_A": {"min": 5.5, "max": 15.0}},
        "baseline_profit": 50.0,
        "grading_seeds": [42],
    },
    
    "medium_simple": {
        "name": "medium_simple",
        "seed": 123,
        "horizon": 14,
        "description": "Two products, variable demand, occasional disruptions",
        "products": ["Product_A", "Product_B"],
        "initial_inventory": {"Product_A": 15, "Product_B": 20},
        "initial_cash": 300.0,
        "base_demand_luxury": {"Product_A": 1.2, "Product_B": 0.8},
        "base_demand_budget": {"Product_A": 2.5, "Product_B": 3.0},
        "demand_elasticity": {"Product_A": 1.1, "Product_B": 1.3},
        "product_costs": {"Product_A": 5.0, "Product_B": 8.0},
        "holding_costs": {"Product_A": 0.06, "Product_B": 0.08},
        "max_inventory": {"Product_A": 80.0, "Product_B": 80.0},
        "lead_time_mean": 2,
        "lead_time_variance": 1,
        "supplier_reliability": 0.95,
        "initial_prices_luxury": {"Product_A": 12.0, "Product_B": 18.0},
        "initial_prices_budget": {"Product_A": 8.0, "Product_B": 12.0},
        "price_bounds": {
            "Product_A": {"min": 5.5, "max": 15.0},
            "Product_B": {"min": 8.8, "max": 20.0},
        },
        "baseline_profit": 100.0,
        "grading_seeds": [123],
    },
    
    "medium_challenge": {
        "name": "medium_challenge",
        "seed": 456,
        "horizon": 14,
        "description": "Two products with demand shifts and supplier delays",
        "products": ["Product_A", "Product_B"],
        "initial_inventory": {"Product_A": 12, "Product_B": 18},
        "initial_cash": 280.0,
        "base_demand_luxury": {"Product_A": 1.5, "Product_B": 0.7},
        "base_demand_budget": {"Product_A": 2.0, "Product_B": 3.5},
        "demand_elasticity": {"Product_A": 1.3, "Product_B": 1.5},
        "product_costs": {"Product_A": 5.5, "Product_B": 9.0},
        "holding_costs": {"Product_A": 0.08, "Product_B": 0.10},
        "max_inventory": {"Product_A": 70.0, "Product_B": 70.0},
        "lead_time_mean": 3,
        "lead_time_variance": 2,
        "supplier_reliability": 0.88,
        "initial_prices_luxury": {"Product_A": 13.0, "Product_B": 20.0},
        "initial_prices_budget": {"Product_A": 9.0, "Product_B": 13.0},
        "price_bounds": {
            "Product_A": {"min": 6.0, "max": 16.0},
            "Product_B": {"min": 9.9, "max": 22.0},
        },
        "baseline_profit": 80.0,
        "grading_seeds": [456],
    },
    
    "hard": {
        "name": "hard",
        "seed": 789,
        "horizon": 21,
        "description": "Three products, high disruption frequency, tight margins",
        "products": ["Product_A", "Product_B", "Product_C"],
        "initial_inventory": {"Product_A": 10, "Product_B": 12, "Product_C": 8},
        "initial_cash": 400.0,
        "base_demand_luxury": {"Product_A": 1.3, "Product_B": 0.9, "Product_C": 1.1},
        "base_demand_budget": {"Product_A": 2.8, "Product_B": 3.2, "Product_C": 2.5},
        "demand_elasticity": {"Product_A": 1.2, "Product_B": 1.4, "Product_C": 1.3},
        "product_costs": {"Product_A": 6.0, "Product_B": 10.0, "Product_C": 7.5},
        "holding_costs": {"Product_A": 0.10, "Product_B": 0.12, "Product_C": 0.08},
        "max_inventory": {"Product_A": 60.0, "Product_B": 50.0, "Product_C": 55.0},
        "lead_time_mean": 3,
        "lead_time_variance": 2,
        "supplier_reliability": 0.85,
        "initial_prices_luxury": {"Product_A": 14.0, "Product_B": 22.0, "Product_C": 16.0},
        "initial_prices_budget": {"Product_A": 9.5, "Product_B": 14.0, "Product_C": 11.0},
        "price_bounds": {
            "Product_A": {"min": 6.6, "max": 18.0},
            "Product_B": {"min": 11.0, "max": 25.0},
            "Product_C": {"min": 8.25, "max": 20.0},
        },
        "baseline_profit": 120.0,
        "grading_seeds": [789],
    },
    
    "expert": {
        "name": "expert",
        "seed": 999,
        "horizon": 30,
        "description": "Four products, extreme disruptions, multi-segment optimization",
        "products": ["Product_A", "Product_B", "Product_C", "Product_D"],
        "initial_inventory": {"Product_A": 8, "Product_B": 10, "Product_C": 6, "Product_D": 12},
        "initial_cash": 500.0,
        "base_demand_luxury": {"Product_A": 1.5, "Product_B": 1.0, "Product_C": 0.8, "Product_D": 1.2},
        "base_demand_budget": {"Product_A": 3.0, "Product_B": 3.5, "Product_C": 2.8, "Product_D": 3.2},
        "demand_elasticity": {"Product_A": 1.2, "Product_B": 1.5, "Product_C": 1.4, "Product_D": 1.1},
        "product_costs": {"Product_A": 6.5, "Product_B": 11.0, "Product_C": 8.0, "Product_D": 9.5},
        "holding_costs": {"Product_A": 0.11, "Product_B": 0.14, "Product_C": 0.09, "Product_D": 0.12},
        "max_inventory": {"Product_A": 50.0, "Product_B": 45.0, "Product_C": 48.0, "Product_D": 55.0},
        "lead_time_mean": 3,
        "lead_time_variance": 3,
        "supplier_reliability": 0.80,
        "initial_prices_luxury": {"Product_A": 15.0, "Product_B": 24.0, "Product_C": 18.0, "Product_D": 19.0},
        "initial_prices_budget": {"Product_A": 10.0, "Product_B": 15.0, "Product_C": 12.0, "Product_D": 13.0},
        "price_bounds": {
            "Product_A": {"min": 7.15, "max": 19.5},
            "Product_B": {"min": 12.1, "max": 27.5},
            "Product_C": {"min": 8.8, "max": 22.0},
            "Product_D": {"min": 10.45, "max": 23.0},
        },
        "baseline_profit": 200.0,
        "grading_seeds": [999],
    },
}

TASK_PRESETS = {**TASKS}


def get_task_config(task_name: str) -> Dict[str, Any]:
    """Get task configuration by name."""
    if task_name not in TASK_PRESETS:
        raise KeyError(f"Unknown task: {task_name}")
    
    # Return a deep copy for safety
    import copy
    return copy.deepcopy(TASK_PRESETS[task_name])


def list_tasks() -> List[Dict[str, Any]]:
    """List available tasks with metadata."""
    return [
        {
            "name": name,
            "horizon": cfg.get("horizon", 30),
            "num_products": len(cfg.get("products", [])),
            "description": cfg.get("description", ""),
            "baseline_profit": cfg.get("baseline_profit", 0.0),
        }
        for name, cfg in TASK_PRESETS.items()
    ]


def get_task_seed_set(task_name: str) -> List[int]:
    """Get grading seeds for a task."""
    config = get_task_config(task_name)
    return config.get("grading_seeds", [])
