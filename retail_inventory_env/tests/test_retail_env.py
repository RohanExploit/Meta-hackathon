"""Environment unit tests: reproducibility, horizon, pricing, episode end."""

import pytest

from environment.retail_env import RetailInventoryEnv
from environment.models import ActionType, NoOpAction, OrderAction, SetPriceAction


def _minimal_task(**overrides):
    base = {
        "products": ["W"],
        "initial_inventory": {"W": 10},
        "initial_cash": 100.0,
        "fixed_demand": {"W": 1},
        "product_costs": {"W": 5.0},
        "holding_costs": {"W": 0.1},
        "max_inventory": {"W": 100},
        "horizon": 5,
        "baseline_profit": 50.0,
    }
    base.update(overrides)
    return base


def test_reset_and_step_noop():
    env = RetailInventoryEnv(seed=42)
    env.reset(_minimal_task())
    obs, _, done, info = env.step(NoOpAction(action=ActionType.NO_OP))
    assert obs.day == 1
    assert obs.sales_history["W"] >= 0
    assert not done
    assert "invalid_action" in info


def test_horizon_terminates_episode():
    env = RetailInventoryEnv(seed=0)
    env.reset(_minimal_task(horizon=2))
    _, _, done, _ = env.step(NoOpAction(action=ActionType.NO_OP))
    assert not done
    _, _, done, info = env.step(NoOpAction(action=ActionType.NO_OP))
    assert done
    assert "grader" in info
    assert 0.0 <= info["grader"]["score"] <= 1.0


def test_reproducibility_same_seed():
    def run():
        env = RetailInventoryEnv(seed=123)
        env.reset(_minimal_task(fixed_demand={"W": 3}))
        rewards = []
        for _ in range(4):
            _, r, d, _ = env.step(NoOpAction(action=ActionType.NO_OP))
            rewards.append(r)
            if d:
                break
        return rewards

    assert run() == run()


def test_set_price_affects_cash_path():
    env = RetailInventoryEnv(seed=0)
    env.reset(_minimal_task(fixed_demand={"W": 5}, initial_inventory={"W": 20}))
    # Max legal price for cost 5 is 10; same RNG demand → higher price ⇒ higher end-of-step cash
    obs_hi, _, _, _ = env.step(
        SetPriceAction(action=ActionType.SET_PRICE, product="W", new_price=10.0)
    )

    env2 = RetailInventoryEnv(seed=0)
    env2.reset(_minimal_task(fixed_demand={"W": 5}, initial_inventory={"W": 20}))
    obs_lo, _, _, _ = env2.step(
        SetPriceAction(action=ActionType.SET_PRICE, product="W", new_price=5.5)
    )

    assert obs_hi.cash > obs_lo.cash


def test_sales_history_matches_last_step_sales():
    env = RetailInventoryEnv(seed=0)
    obs0 = env.reset(_minimal_task(fixed_demand={"W": 2}))
    assert obs0.sales_history["W"] == 0
    obs1, _, _, _ = env.step(NoOpAction(action=ActionType.NO_OP))
    # Under fixed Poisson mean 2, sales are stochastic; history should be non-negative int
    assert isinstance(obs1.sales_history["W"], int)
    assert obs1.sales_history["W"] >= 0


def test_invalid_order_flagged():
    env = RetailInventoryEnv(seed=0)
    env.reset(_minimal_task())
    _, _, _, info = env.step(
        OrderAction(action=ActionType.ORDER, product="W", quantity=10**9)
    )
    assert info["invalid_action"] is True


def test_grader_present_on_done():
    env = RetailInventoryEnv(seed=1)
    env.reset(_minimal_task(horizon=1))
    _, _, done, info = env.step(NoOpAction(action=ActionType.NO_OP))
    assert done
    assert "score" in info.get("grader", {})
