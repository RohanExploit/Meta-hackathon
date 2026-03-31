"""Pre-submission checks mapped to Context.txt requirements.

Runs local validations and prints a rubric-aligned PASS/FAIL report.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> tuple[bool, str]:
    try:
        out = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        text = (out.stdout or "") + (out.stderr or "")
        return True, text.strip()
    except subprocess.CalledProcessError as exc:
        text = (exc.stdout or "") + (exc.stderr or "")
        return False, text.strip()
    except FileNotFoundError as exc:
        return False, str(exc)


def _check_health_script() -> tuple[bool, str]:
    return _run_cmd([sys.executable, "scripts/health_check.py"])


def _check_task_count() -> tuple[bool, str]:
    code = (
        "from environment.tasks import TASKS; "
        "n=len(TASKS); "
        "print(f'tasks={n}'); "
        "raise SystemExit(0 if n>=3 else 1)"
    )
    return _run_cmd([sys.executable, "-c", code])


def _check_inference_contract() -> tuple[bool, str]:
    required = ["API_BASE_URL", "MODEL_NAME"]
    missing = [name for name in required if not os.getenv(name)]
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")):
        missing.append("OPENAI_API_KEY (or HF_TOKEN/API_KEY)")
    if missing:
        return False, f"Missing env vars for inference: {', '.join(missing)}"
    return True, "Inference env vars present"


def _check_docker_available() -> tuple[bool, str]:
    return _run_cmd(["docker", "--version"])


def _check_openenv_cli_available() -> tuple[bool, str]:
    return _run_cmd(["openenv", "validate", "."])


def main() -> int:
    skip_health = os.getenv("PRECHECK_SKIP_HEALTH", "0") == "1"
    skip_docker = os.getenv("PRECHECK_SKIP_DOCKER", "0") == "1"
    skip_openenv = os.getenv("PRECHECK_SKIP_OPENENV", "0") == "1"

    checks = [
        ("At least 3 tasks are defined", _check_task_count),
        ("Inference env vars configured", _check_inference_contract),
    ]

    if not skip_health:
        checks.insert(0, ("Core health checks (compile+tests+smoke)", _check_health_script))

    if not skip_docker:
        checks.append(("Docker CLI available", _check_docker_available))
    if not skip_openenv:
        checks.append(("OpenEnv validate CLI available", _check_openenv_cli_available))

    print("\n=== Pre-Submission Check ===")
    failures = 0

    for title, fn in checks:
        ok, details = fn()
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {title}")
        if details:
            print(details[:1200])
        if not ok:
            failures += 1

    print("\n=== Result ===")
    if failures == 0:
        print("All checks passed.")
        return 0

    print(f"{failures} check(s) failed. Resolve these before submission.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
