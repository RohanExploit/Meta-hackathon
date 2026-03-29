"""Full HTTP API tests using Starlette TestClient."""

from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)

MIN_TASK = {
    "products": ["W"],
    "initial_inventory": {"W": 8},
    "initial_cash": 100.0,
    "fixed_demand": {"W": 2},
    "product_costs": {"W": 5.0},
    "holding_costs": {"W": 0.05},
    "max_inventory": {"W": 50},
    "horizon": 3,
    "baseline_profit": 25.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_reset_and_step_flow():
    r = client.post("/reset", json={"task_config": MIN_TASK, "seed": 7})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["done"] is False
    assert "W" in body["observation"]["inventory"]

    r2 = client.post("/step", json={"action": {"action": "noop"}})
    assert r2.status_code == 200, r2.text
    b2 = r2.json()
    assert b2["observation"]["day"] == 1
    assert "reward" in b2


def test_reset_invalid_task_400():
    r = client.post("/reset", json={"task_config": {"bad": True}, "seed": 1})
    assert r.status_code == 400


def test_step_unknown_action_400():
    client.post("/reset", json={"task_config": MIN_TASK, "seed": 1})
    r = client.post("/step", json={"action": {"action": "fly"}})
    assert r.status_code == 400


def test_state_masks_demand():
    client.post("/reset", json={"task_config": MIN_TASK, "seed": 99})
    r = client.get("/state")
    assert r.status_code == 200
    dp = r.json()["state"]["demand_pattern"]
    assert all(v == 0.0 for v in dp.values())
