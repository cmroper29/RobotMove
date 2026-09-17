import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
URDF = EXAMPLES / "urdf" / "kr6_r900.urdf"


@pytest.fixture
def write(tmp_path):
    def _write(name: str, text: str) -> Path:
        p = tmp_path / name
        p.write_text(text)
        return p

    return _write
