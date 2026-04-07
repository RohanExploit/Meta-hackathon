"""Deterministic production grader for retail episodes.

Scoring Architecture (Meta/OpenEnv hardened):
─────────────────────────────────────────────
• Non-linear penalties prevent reward hacking via over-ordering
• Multiplicative bankruptcy gate collapses score for negative-profit runs
• Over-order exploit detection penalises high fill-rate + low efficiency
• Output is strictly bounded to [0.0, 1.0]
"""
from __future__ import annotations

import math
from typing import Any, Dict


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_episode(summary: Dict[str, Any]) -> Dict[str, float]:
    """Compute a bounded deterministic score in [0, 1].

    Scoring formula (non-linear, exploit-resistant):
        1. Base components: profit_score, fill_rate, efficiency
        2. Overstocking penalty: if efficiency < 0.5 the profit contribution
           is exponentially reduced  (scales as (eff/0.5)^1.5)
        3. Weighted sum:  0.45·profit + 0.35·fill_rate + 0.20·efficiency
        4. Critical-failure gates (multiplicative):
           a. fill_rate < 0.6  →  score *= 0.5
           b. profit < 0       →  score *= max(0.05, 1 + profit/baseline)
           c. high fill + low efficiency (over-order exploit) → score *= 0.6
    """

    profit = float(summary.get("profit", 0.0))
    baseline_profit = float(summary.get("baseline_profit", 100.0))
    total_demand = float(summary.get("total_demand", 0.0))
    total_sales = float(summary.get("total_sales", 0.0))
    total_holding_cost = float(
        summary.get("total_holding_cost", summary.get("total_cost", 0.0))
    )
    max_possible_holding_cost = float(
        summary.get("max_possible_holding_cost", 0.0)
    )

    safe_baseline = baseline_profit if baseline_profit > 0.0 else 1.0

    # ── Base component scores ───────────────────────────────────────
    profit_score = _clamp01(profit / safe_baseline)

    if total_demand > 0.0:
        fill_rate = _clamp01(total_sales / total_demand)
    else:
        fill_rate = 1.0
    fill_rate_score = fill_rate

    if max_possible_holding_cost > 0.0:
        efficiency = _clamp01(
            1.0 - (total_holding_cost / max_possible_holding_cost)
        )
    else:
        efficiency = 1.0
    efficiency_score = efficiency

    # ── NON-LINEAR: overstocking penalty on profit ──────────────────
    # If efficiency drops below 0.5 (massive overstocking), the profit
    # contribution decays exponentially.  This prevents brute-force
    # over-ordering to inflate fill-rate while ignoring warehouse costs.
    adjusted_profit_score = profit_score
    if efficiency < 0.5:
        decay = (efficiency / 0.5) ** 1.5  # e.g. eff=0.25 → decay≈0.35
        adjusted_profit_score = profit_score * decay

    # ── Weighted combination ────────────────────────────────────────
    raw_score = (
        0.45 * adjusted_profit_score
        + 0.35 * fill_rate_score
        + 0.20 * efficiency_score
    )

    # ── Critical-failure gates (multiplicative) ─────────────────────

    # Gate 1: Low fill-rate penalty (agent is failing to meet demand)
    if fill_rate < 0.6:
        raw_score *= 0.5

    # Gate 2: Bankruptcy penalty (profit is negative → going bankrupt)
    # A company with 100% fill-rate but negative profit is NOT succeeding.
    if profit < 0:
        bankruptcy_factor = max(0.05, 1.0 + (profit / safe_baseline))
        raw_score *= bankruptcy_factor

    # Gate 3: Over-ordering exploit (high fill + low efficiency)
    # Catches agents that max out fill-rate by flooding the warehouse.
    if fill_rate > 0.9 and efficiency < 0.3:
        raw_score *= 0.6

    final_score = _clamp01(raw_score)

    return {
        "score": final_score,
        "profit_score": profit_score,
        "fill_rate_score": fill_rate_score,
        "efficiency_score": efficiency_score,
        "fill_rate": fill_rate,
        "profit": profit,
        "baseline_profit": baseline_profit,
        # Diagnostic fields for debugging
        "adjusted_profit_score": adjusted_profit_score,
        "overstocking_penalty_applied": efficiency < 0.5,
        "bankruptcy_penalty_applied": profit < 0,
        "overorder_exploit_detected": fill_rate > 0.9 and efficiency < 0.3,
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
