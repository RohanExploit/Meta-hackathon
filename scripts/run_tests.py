"""Run pytest with host plugin autoload disabled.

This avoids failures from globally installed pytest plugins on this machine.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    import pytest

    args = sys.argv[1:] or ["tests"]
    return int(pytest.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
