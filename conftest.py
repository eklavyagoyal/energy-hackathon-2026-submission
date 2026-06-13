"""Pytest root conftest: put the repo root on sys.path so tests can
import the `src` package without an editable install."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
