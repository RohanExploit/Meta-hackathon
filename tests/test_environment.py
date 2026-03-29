"""Tests for multi-channel retail environment."""
import pytest
from environment.models import AllocateAction, NoOpAction, OrderAction, SetPriceAction
from environment.retail_env import MultiChannelRetailEnv
from environment.tasks import get_task_config, list_tasks
from environment.grader import score_episode


def test_task_configs_exist():
    """Test that all tasks are accessible."""
    tasks = list_tasks()
    assert len(tasks) >= 5, "Should have at least 5 tasks"
    
    task_names = [t["name"] for t in tasks]
    for name in ["easy", "medium_simple", "medium_challenge", "hard", "expert"]:
        assert name in task_names, f"Task {name} not found"


def test_environment_reset():
    """Test environment reset."""
    env = MultiChannelRetailEnv(seed=42)
    task_cfg = get_task_config("easy")
    obs = env.reset(task_cfg)
    
    assert obs.day == 0
    assert obs.cash > 0
    assert len(obs.inventory) > 0
    assert not obs.disruption_active


def test_environment_step():
    """Test environment step."""
    env = MultiChannelRetailEnv(seed=42)
    task_cfg = get_task_config("easy")
    env.reset(task_cfg)
    
    action = NoOpAction(action="noop")
    obs, reward, done, info = env.step(action)
    
    assert obs.day == 1
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_allocate_action():
    """Test allocate action."""
    env = MultiChannelRetailEnv(seed=42)
    task_cfg = get_task_config("easy")
    env.reset(task_cfg)
    
    product = list(env.state.inventory.keys())[0]
    action = AllocateAction(
        action="allocate",
        product=product,
        luxury_units=5,
        budget_units=5
    )
    obs, reward, done, info = env.step(action)
    
    assert obs.day == 1
    assert info["valid_action"] == True


def test_order_action():
    """Test order action."""
    env = MultiChannelRetailEnv(seed=42)
    task_cfg = get_task_config("easy")
    env.reset(task_cfg)
    
    product = list(env.state.inventory.keys())[0]
    action = OrderAction(
        action="order",
        product=product,
        quantity=2
    )
    obs, reward, done, info = env.step(action)
    
    assert obs.day == 1
    assert isinstance(info, dict)


def test_set_price_action():
    """Test set price action."""
    env = MultiChannelRetailEnv(seed=42)
    task_cfg = get_task_config("easy")
    env.reset(task_cfg)
    
    product = list(env.state.inventory.keys())[0]
    action = SetPriceAction(
        action="set_price",
        product=product,
        segment="budget",
        new_price=9.5
    )
    obs, reward, done, info = env.step(action)
    
    assert obs.day == 1


def test_full_episode():
    """Test a full episode."""
    env = MultiChannelRetailEnv(seed=42)
    task_cfg = get_task_config("easy")
    env.reset(task_cfg)
    
    total_reward = 0.0
    for _ in range(10):
        action = NoOpAction(action="noop")
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    assert obs.day > 0
    assert obs.day <= 10


def test_terminal_grader_present_on_done():
    env = MultiChannelRetailEnv(seed=42)
    task_cfg = get_task_config("easy")
    task_cfg["horizon"] = 1
    env.reset(task_cfg)

    _, _, done, info = env.step(NoOpAction(action="noop"))
    assert done is True
    assert "grader" in info
    assert "terminal_summary" in info
    assert 0.0 <= info["grader"]["score"] <= 1.0


def test_grader():
    """Test grader function."""
    summary = {
        "profit": 100.0,
        "baseline_profit": 100.0,
        "total_demand": 100.0,
        "total_sales": 100.0,
        "disruption_events": 0,
        "recovery_success_rate": 0.0,
    }
    
    result = score_episode(summary)
    
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
    assert result["score"] == 1.0  # Perfect profit and fill rate


def test_grader_low_fill_rate():
    """Test grader with low fill rate."""
    summary = {
        "profit": 100.0,
        "baseline_profit": 100.0,
        "total_demand": 100.0,
        "total_sales": 40.0,  # 40% fill rate
        "disruption_events": 0,
        "recovery_success_rate": 0.0,
    }
    
    result = score_episode(summary)
    
    # Score should be penalized
    assert result["score"] < 1.0
    assert result["fill_rate"] == 0.4


def test_reproducibility():
    """Test that seeded runs are reproducible."""
    env1 = MultiChannelRetailEnv(seed=42)
    task_cfg = get_task_config("easy")
    env1.reset(task_cfg)
    
    env2 = MultiChannelRetailEnv(seed=42)
    env2.reset(task_cfg)
    
    # Both should have same initial state
    assert env1.state.cash == env2.state.cash
    assert env1.state.inventory == env2.state.inventory


def test_observation_deterministic_given_seed():
    env1 = MultiChannelRetailEnv(seed=123)
    cfg = get_task_config("easy")
    o1 = env1.reset(cfg)

    env2 = MultiChannelRetailEnv(seed=123)
    o2 = env2.reset(cfg)

    assert o1.recent_demand_luxury == o2.recent_demand_luxury
    assert o1.recent_demand_budget == o2.recent_demand_budget


def test_live_endpoints_basic_cycle():
    from fastapi.testclient import TestClient
    from server.app import app

    client = TestClient(app)

    start = client.post(
        "/live/start",
        json={"task_name": "easy", "mode": "noop", "interval_ms": 100},
    )
    assert start.status_code == 200

    status = client.get("/live/status")
    assert status.status_code == 200
    assert "running" in status.json()

    latest = client.get("/live/latest")
    assert latest.status_code == 200

    stop = client.post("/live/stop", json={"reason": "test"})
    assert stop.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
