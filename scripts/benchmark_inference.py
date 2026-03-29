"""Run inference twice and compare outputs for reproducibility evidence.

Requires API_BASE_URL, MODEL_NAME, HF_TOKEN and a running env server.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run_once(index: int) -> tuple[bool, float, dict]:
    start = time.time()
    proc = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    duration = time.time() - start

    payload = {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    report = ROOT / f"inference_run_{index}.log"
    report.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")

    mean_score = None
    for line in proc.stdout.splitlines()[::-1]:
        stripped = line.strip()
        if stripped.startswith("{") and '"mean_score"' in stripped:
            try:
                mean_score = json.loads(stripped).get("mean_score")
            except json.JSONDecodeError:
                mean_score = None
            break

    return proc.returncode == 0, duration, {"mean_score": mean_score, **payload}


def main() -> int:
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}")
        return 1

    ok1, t1, r1 = _run_once(1)
    ok2, t2, r2 = _run_once(2)

    print("Benchmark summary:")
    print(f"run1_ok={ok1} runtime_sec={t1:.2f} mean_score={r1.get('mean_score')}")
    print(f"run2_ok={ok2} runtime_sec={t2:.2f} mean_score={r2.get('mean_score')}")
    print(f"runtime_under_20min={t1 < 1200 and t2 < 1200}")
    print(f"score_match={r1.get('mean_score') == r2.get('mean_score')}")
    print("Logs: inference_run_1.log, inference_run_2.log")

    if not (ok1 and ok2):
        return 1
    if not (t1 < 1200 and t2 < 1200):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
