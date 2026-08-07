"""Make factory_v1 modules importable when pytest starts at repository root."""

from pathlib import Path
import sys


FACTORY_DIR = Path(__file__).resolve().parents[1]
if str(FACTORY_DIR) not in sys.path:
    sys.path.insert(0, str(FACTORY_DIR))
