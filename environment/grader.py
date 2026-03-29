"""Deterministic production grader for retail episodes."""
from typing import Any, Dict


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_episode(summary: Dict[str, Any]) -> Dict[str, float]:
    """Compute a bounded deterministic score in [0, 1]."""

    profit = float(summary.get("profit", 0.0))
    baseline_profit = float(summary.get("baseline_profit", 100.0))
    total_demand = float(summary.get("total_demand", 0.0))
    total_sales = float(summary.get("total_sales", 0.0))
    total_holding_cost = float(summary.get("total_holding_cost", summary.get("total_cost", 0.0)))
    max_possible_holding_cost = float(summary.get("max_possible_holding_cost", 0.0))

    safe_baseline = baseline_profit if baseline_profit > 0.0 else 1.0
    profit_score = _clamp01(profit / safe_baseline)

    if total_demand > 0.0:
        fill_rate = _clamp01(total_sales / total_demand)
    else:
        fill_rate = 1.0
    fill_rate_score = fill_rate

    if max_possible_holding_cost > 0.0:
        efficiency = _clamp01(1.0 - (total_holding_cost / max_possible_holding_cost))
    else:
        efficiency = 1.0
    efficiency_score = efficiency

    score = (
        0.5 * profit_score
        + 0.3 * fill_rate_score
        + 0.2 * efficiency_score
    )

    if fill_rate < 0.6:
        score *= 0.5

    final_score = _clamp01(score)

    return {
        "score": final_score,
        "profit_score": profit_score,
        "fill_rate_score": fill_rate_score,
        "efficiency_score": efficiency_score,
        "fill_rate": fill_rate,
        "profit": profit,
        "baseline_profit": baseline_profit,
    }


def evaluate_seeded_summaries(
    summaries: list[Dict[str, Any]],
    variance_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Aggregate scores across multiple seeds."""
    
    scores = [score_episode(s)["score"] for s in summaries]
    
    if not scores:
        return {
            "mean_score": 0.0,
            "variance": 0.0,
            "adjusted_score": 0.0,
            "num_seeds": 0,
        }
    
    import statistics
    mean = statistics.fmean(scores)
    variance = statistics.pvariance(scores) if len(scores) > 1 else 0.0
    adjusted = max(0.0, min(1.0, mean - variance_penalty * variance))
    
    return {
        "mean_score": mean,
        "variance": variance,
        "adjusted_score": adjusted,
        "num_seeds": len(scores),
        "seed_scores": scores,
    }
