# tests/test_risk/test_eda.py
"""STUB -- src/risk/eda.py is itself a structural stub (see its module
docstring): the real logic for this pipeline stage lives in a notebook, not
an importable module, so there's nothing here to unit test yet. Fill this
in alongside promoting eda.py from a stub to real code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.risk import eda


def test_module_imports_cleanly():
    """The one thing this stub module needs to do correctly today."""
    assert eda is not None
