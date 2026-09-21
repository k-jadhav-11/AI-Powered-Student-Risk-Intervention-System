# src/risk/__init__.py
"""
Flattens src/risk/'s submodules onto one namespace, so callers that need
the whole Sprint 3 surface (streamlit_app.py) can do `from src import risk
as rs` and keep using `rs.load_model(...)`, `rs.plot_precision_recall(...)`,
`rs.load_gru_model(...)`, etc. exactly as they did when all of this lived
in one file (the original risk_service.py).

Sprint 4 code (ai_assistant/tools.py, src/explainability.py) deliberately does NOT
import through this package -- it imports `from src.risk import
risk_queries` directly, to keep its dependency surface to exactly the
handful of functions it actually needs, not everything Sprint 3 has ever
defined. See risk_queries.py's module docstring for the full reasoning.
"""
from .risk_queries import *  # noqa: F401,F403
from .risk_charts import *  # noqa: F401,F403
from .clustering import *  # noqa: F401,F403
from .model_gru import *  # noqa: F401,F403
from .evaluation import *  # noqa: F401,F403
