from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CATALOG_DIR = ROOT / "catalog" / "v2"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
