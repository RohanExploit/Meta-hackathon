from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_WEIGHTS = {
    "profit": 0.40,
    "fill_rate": 0.20,
    "stockout_severity": 0.15,
    "inventory_health": 0.15,
    "price_stability": 0.10,
}

DEFAULT_GUARDRAILS = {
    "min_service_threshold": 0.60,
    "hard_service_floor": 0.40,
    "profit_floor": 0.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return numerator / denominator


def score_episode(
    summary: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    guardrails: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Deterministic episode grader with anti-exploit guardrails.

    Returns a normalized final score in [0, 1] with component scores.
    """

    component_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        component_weights.update(weights)

    guardrail_cfg = dict(DEFAULT_GUARDRAILS)
    if guardrails:
        guardrail_cfg.update(guardrails)

    total_weight = sum(max(0.0, float(v)) for v in component_weights.values())
    if total_weight <= 0.0:
        component_weights = dict(DEFAULT_WEIGHTS)
        total_weight = sum(component_weights.values())

    # Normalize weights deterministically
    component_weights = {
        key: max(0.0, float(value)) / total_weight for key, value in component_weights.items()
    }

    profit = float(summary.get("profit", 0.0))
    target_profit = float(summary.get("target_profit", summary.get("baseline_profit", 1.0)))

    total_demand = float(summary.get("total_demand", 0.0))
    total_sales = float(summary.get("total_sales", 0.0))
    total_unmet = float(summary.get("total_unmet_demand", max(total_demand - total_sales, 0.0)))

    total_holding_cost = float(summary.get("total_holding_cost", 0.0))
    max_possible_holding = float(summary.get("max_possible_holding_cost", 0.0))
    ending_inventory_ratio = _clamp01(float(summary.get("ending_inventory_ratio", 0.0)))

    horizon = max(int(summary.get("horizon", 1)), 1)
    num_products = max(int(summary.get("num_products", 1)), 1)
    change_budget_scale = float(summary.get("price_change_budget_scale", 0.35))
    change_budget = max(1.0, horizon * num_products * change_budget_scale)
    price_change_magnitude = float(summary.get("price_change_magnitude", 0.0))

    if target_profit <= 0:
        profit_score = 1.0 if profit > 0 else 0.0
    else:
        profit_score = _clamp01(profit / target_profit)

    fill_rate = _clamp01(_safe_div(total_sales, total_demand, default=1.0 if total_demand <= 0 else 0.0))
    fill_rate_score = fill_rate

    stockout_ratio = _clamp01(_safe_div(total_unmet, total_demand, default=0.0))
    stockout_severity_score = _clamp01(1.0 - stockout_ratio)

    if max_possible_holding > 0.0:
        holding_ratio = _clamp01(total_holding_cost / max_possible_holding)
    else:
        holding_ratio = 0.0
    inventory_health_score = _clamp01(1.0 - 0.7 * holding_ratio - 0.3 * ending_inventory_ratio)

    price_stability_score = _clamp01(1.0 - (price_change_magnitude / change_budget))

    raw_score = (
        component_weights.get("profit", 0.0) * profit_score
        + component_weights.get("fill_rate", 0.0) * fill_rate_score
        + component_weights.get("stockout_severity", 0.0) * stockout_severity_score
        + component_weights.get("inventory_health", 0.0) * inventory_health_score
        + component_weights.get("price_stability", 0.0) * price_stability_score
    )

    min_service_threshold = float(guardrail_cfg["min_service_threshold"])
    hard_service_floor = float(guardrail_cfg["hard_service_floor"])
    profit_floor = float(guardrail_cfg["profit_floor"])

    guardrail_applied = 0.0
    if fill_rate < hard_service_floor:
        raw_score = min(raw_score, 0.30)
        guardrail_applied = 1.0
    elif fill_rate < min_service_threshold:
        raw_score *= 0.75
        guardrail_applied = 1.0

    if profit <= profit_floor and fill_rate < 0.90:
        raw_score = min(raw_score, 0.50)
        guardrail_applied = 1.0

    final_score = _clamp01(raw_score)

    return {
        "score": final_score,
        "profit_score": profit_score,
        "fill_rate_score": fill_rate_score,
        "stockout_severity_score": stockout_severity_score,
        "inventory_health_score": inventory_health_score,
        "price_stability_score": price_stability_score,
        "fill_rate": fill_rate,
        "stockout_ratio": stockout_ratio,
        "guardrail_applied": guardrail_applied,
    }


def compute_deterministic_score(summary: Dict[str, Any]) -> Dict[str, float]:
    """Backward-compatible deterministic single-episode scoring alias."""

    return score_episode(summary)


def aggregate_seed_scores(
    scored_episodes: Iterable[Dict[str, Any]],
    variance_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Aggregate multiple deterministic episode scores with optional variance penalty."""

    score_list: List[float] = []
    for episode in scored_episodes:
        if "score" in episode:
            score_list.append(float(episode["score"]))
        else:
            score_list.append(float(score_episode(episode)["score"]))

    if not score_list:
        return {
            "mean_score": 0.0,
            "score_variance": 0.0,
            "variance_penalty": float(variance_penalty),
            "adjusted_score": 0.0,
            "num_seeds": 0,
        }

    mean_score = float(statistics.fmean(score_list))
    variance = float(statistics.pvariance(score_list)) if len(score_list) > 1 else 0.0
    adjusted_score = _clamp01(mean_score - float(variance_penalty) * variance)

    return {
        "mean_score": mean_score,
        "score_variance": variance,
        "variance_penalty": float(variance_penalty),
        "adjusted_score": adjusted_score,
        "num_seeds": len(score_list),
        "seed_scores": score_list,
    }


def evaluate_seeded_summaries(
    summaries: Iterable[Dict[str, Any]],
    variance_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Score each summary then aggregate across seeds."""

    per_seed = [score_episode(summary) for summary in summaries]
    aggregate = aggregate_seed_scores(per_seed, variance_penalty=variance_penalty)
    aggregate["per_seed"] = per_seed
    return aggregate
