import sys
from pathlib import Path

# Make scripts/lab importable without packaging the scripts dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "lab"))
