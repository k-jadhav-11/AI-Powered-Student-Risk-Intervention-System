# tests/test_risk/test_model_baseline.py
"""STUB -- src/risk/model_baseline.py is itself a structural stub (see its module
docstring): the real logic for this pipeline stage lives in a notebook, not
an importable module, so there's nothing here to unit test yet. Fill this
in alongside promoting model_baseline.py from a stub to real code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.risk import model_baseline


def test_module_imports_cleanly():
    """The one thing this stub module needs to do correctly today."""
    assert model_baseline is not None
