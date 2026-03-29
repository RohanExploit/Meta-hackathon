"""Unit tests for deterministic multi-metric grader."""

import pytest

from environment.retail_env import compute_deterministic_score


def test_grader_profit_only_baseline():
    out = compute_deterministic_score(
        {
            "profit": 50.0,
            "baseline_profit": 100.0,
            "total_sales": 0.0,
            "total_demand": 0.0,
            "total_holding_cost": 0.0,
            "max_possible_holding_cost": 0.0,
        }
    )
    assert out["profit_score"] == pytest.approx(0.5)
    assert out["fill_rate"] == 1.0
    assert out["score"] == pytest.approx(0.5 * 0.5 + 0.3 * 1.0 + 0.2 * 1.0)


def test_grader_fill_rate_antigaming():
    out = compute_deterministic_score(
        {
            "profit": 100.0,
            "baseline_profit": 100.0,
            "total_sales": 40.0,
            "total_demand": 100.0,
            "total_holding_cost": 0.0,
            "max_possible_holding_cost": 1.0,
        }
    )
    assert out["fill_rate"] == pytest.approx(0.4)
    raw = 0.5 * 1.0 + 0.3 * 0.4 + 0.2 * 1.0
    assert out["score"] == pytest.approx(raw * 0.5)


def test_grader_clamped_to_one():
    out = compute_deterministic_score(
        {
            "profit": 200.0,
            "baseline_profit": 100.0,
            "total_sales": 10.0,
            "total_demand": 10.0,
            "total_holding_cost": 0.0,
            "max_possible_holding_cost": 1.0,
        }
    )
    assert out["score"] == 1.0


def test_grader_zero_baseline_safe():
    out = compute_deterministic_score(
        {
            "profit": 0.0,
            "baseline_profit": 0.0,
            "total_sales": 0.0,
            "total_demand": 0.0,
            "total_holding_cost": 0.0,
            "max_possible_holding_cost": 0.0,
        }
    )
    assert out["profit_score"] == 0.0
    assert 0.0 <= out["score"] <= 1.0
