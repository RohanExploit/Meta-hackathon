"""API contract tests for OpenEnv-style endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app


def test_reset_step_until_done_includes_grader():
    client = TestClient(app)

    reset = client.post("/reset", json={"task_name": "easy", "seed": 42})
    assert reset.status_code == 200
    assert reset.json()["done"] is False

    done = False
    last_info = {}
    for _ in range(20):
        step = client.post("/step", json={"action": {"action": "noop"}})
        assert step.status_code == 200
        body = step.json()
        done = bool(body["done"])
        last_info = body.get("info", {})
        if done:
            break

    assert done is True
    assert "grader" in last_info
    assert "terminal_summary" in last_info
    assert 0.0 <= float(last_info["grader"]["score"]) <= 1.0


def test_evaluate_single_summary():
    client = TestClient(app)
    response = client.post(
        "/evaluate",
        json={
            "summary": {
                "profit": 50.0,
                "baseline_profit": 100.0,
                "total_demand": 100.0,
                "total_sales": 70.0,
                "total_holding_cost": 10.0,
                "max_possible_holding_cost": 100.0,
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "single_summary"
    assert 0.0 <= float(body["result"]["score"]) <= 1.0


def test_evaluate_multi_seed_summary():
    client = TestClient(app)
    response = client.post(
        "/evaluate",
        json={
            "summaries": [
                {
                    "profit": 40.0,
                    "baseline_profit": 100.0,
                    "total_demand": 100.0,
                    "total_sales": 60.0,
                },
                {
                    "profit": 80.0,
                    "baseline_profit": 100.0,
                    "total_demand": 100.0,
                    "total_sales": 85.0,
                },
            ],
            "variance_penalty": 0.1,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "multi_seed"
    assert int(body["result"]["num_seeds"]) == 2
    assert 0.0 <= float(body["result"]["adjusted_score"]) <= 1.0
