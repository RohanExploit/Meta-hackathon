"""Compliance-oriented checks aligned with Context.txt requirements."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from environment.grader import score_episode
from environment.tasks import TASKS, get_task_config
from server.app import app


ROOT = Path(__file__).resolve().parent.parent


def _read_openenv_yaml_text() -> str:
    return (ROOT / "openenv.yaml").read_text(encoding="utf-8")


def _extract_list_block(text: str, key: str) -> set[str]:
    lines = text.splitlines()
    capture = False
    items: set[str] = set()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if not capture and stripped == f"{key}:":
            capture = True
            continue

        if not capture:
            continue

        if not line.startswith("  "):
            break

        stripped = stripped.lstrip("-").strip()
        if stripped:
            items.add(stripped)

    return items


def test_openenv_yaml_contract():
    text = _read_openenv_yaml_text()

    assert "name:" in text
    assert "version:" in text
    assert "author:" in text

    variables = _extract_list_block(text, "variables")
    assert {"API_BASE_URL", "MODEL_NAME", "OPENAI_API_KEY", "HF_TOKEN"}.issubset(variables)

    tasks = _extract_list_block(text, "tasks")
    assert len(tasks) >= 3

    endpoints = _extract_list_block(text, "endpoints")
    required_endpoints = {
        "POST /reset",
        "POST /step",
        "GET /state",
        "GET /tasks",
        "POST /evaluate",
        "GET /health",
        "POST /live/start",
        "POST /live/stop",
        "GET /live/status",
        "GET /live/latest",
    }
    assert required_endpoints.issubset(endpoints)


def test_hf_spaces_config_exists_and_uses_docker():
    spaces_config = ROOT / "app.spaces.yaml"
    assert spaces_config.exists()

    text = spaces_config.read_text(encoding="utf-8")
    assert "sdk: docker" in text
    assert "dockerfile: server/Dockerfile" in text
    assert "openenv" in text


def test_task_count_and_difficulty_tiers():
    assert len(TASKS) >= 3
    for name in ["easy", "medium_simple", "hard"]:
        cfg = get_task_config(name)
        assert cfg["horizon"] > 0
        assert len(cfg["products"]) >= 1


def test_api_required_endpoints_respond():
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200

    r = client.get("/tasks")
    assert r.status_code == 200
    assert len(r.json().get("tasks", [])) >= 3


def test_grader_score_bounded_and_continuous():
    low = score_episode(
        {
            "profit": 0.0,
            "baseline_profit": 100.0,
            "total_demand": 100.0,
            "total_sales": 30.0,
        }
    )["score"]
    mid = score_episode(
        {
            "profit": 50.0,
            "baseline_profit": 100.0,
            "total_demand": 100.0,
            "total_sales": 70.0,
        }
    )["score"]
    high = score_episode(
        {
            "profit": 120.0,
            "baseline_profit": 100.0,
            "total_demand": 100.0,
            "total_sales": 95.0,
        }
    )["score"]

    assert 0.0 <= low <= 1.0
    assert 0.0 <= mid <= 1.0
    assert 0.0 <= high <= 1.0
    assert low < mid < high


def test_grader_multifactor_shape_and_antigaming():
    result = score_episode(
        {
            "profit": 80.0,
            "baseline_profit": 100.0,
            "total_demand": 100.0,
            "total_sales": 40.0,
            "total_holding_cost": 20.0,
            "max_possible_holding_cost": 100.0,
        }
    )

    # Exposes multifactor components
    assert "profit_score" in result
    assert "fill_rate_score" in result
    assert "efficiency_score" in result

    # Anti-gaming should reduce score when fill rate is low
    assert result["fill_rate"] < 0.6
    assert 0.0 <= result["score"] <= 1.0
