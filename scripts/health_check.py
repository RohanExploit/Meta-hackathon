"""Run a quick local system health check for this repository.

Checks:
1) Python syntax compile for core modules
2) Test suite (with pytest host plugin autoload disabled)
3) Legacy smoke scripts
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(step: str, cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n[health] {step}")
    print(f"[health] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> int:
    python = sys.executable

    compile_targets = [
        "environment/models.py",
        "environment/retail_env.py",
        "environment/grader.py",
        "environment/tasks.py",
        "server/app.py",
        "inference.py",
        "test_env.py",
        "test_api.py",
        "scripts/run_tests.py",
    ]

    _run(
        "Compile core Python files",
        [python, "-m", "py_compile", *compile_targets],
    )

    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    _run(
        "Run test suite",
        [python, "scripts/run_tests.py"],
        env=env,
    )

    _run("Run environment smoke script", [python, "test_env.py"])
    _run("Run API smoke script", [python, "test_api.py"])
    _run(
        "Run pre-submission checks (skip Docker/OpenEnv CLI)",
        [python, "scripts/pre_submission_check.py"],
        env={
            **env,
            "API_BASE_URL": env.get("API_BASE_URL", "https://router.huggingface.co/v1"),
            "MODEL_NAME": env.get("MODEL_NAME", "health-check-placeholder"),
            "HF_TOKEN": env.get("HF_TOKEN", "health-check-placeholder"),
            "PRECHECK_SKIP_HEALTH": "1",
            "PRECHECK_SKIP_DOCKER": "1",
            "PRECHECK_SKIP_OPENENV": "1",
        },
    )

    print("\n[health] All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
