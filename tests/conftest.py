import os
import tempfile
from pathlib import Path

import pytest

from ham.memory_engine import MemoryEngine


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def engine(tmp_db):
    # Disable live API calls in tests.
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)
    eng = MemoryEngine(tmp_db)
    yield eng
    eng.close()
