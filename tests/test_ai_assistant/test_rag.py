# tests/test_ai_assistant/test_rag.py
"""ai_assistant/rag.py -- RAG retrieval using LangChain's Chroma wrapper.
No LLM/network needed for retrieval itself; embedding backend is whatever
vectorstore.get_embedding_function() resolves to in the current environment
(real sentence-transformers or the offline hash fallback)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import rag, tools, vectorstore


def test_retrieve_remedial_content_returns_real_grounded_text():
    """Every suggestion returned must be verbatim text from
    vectorstore.STARTER_REMEDIAL_CONTENT -- never generated or altered."""
    real_texts = {item["text"] for item in vectorstore.STARTER_REMEDIAL_CONTENT}
    result = rag.retrieve_remedial_content(["failures", "absences"], k=2)
    for suggestion in result["suggestions"]:
        assert suggestion in real_texts


def test_exact_tag_match_beats_similarity_search():
    """A factor with an exact factor_tags match should retrieve that exact
    item, not something else via similarity search -- this is the
    regression test for the hybrid-retrieval fix (pure embedding search on
    the offline hash fallback matched poorly on snake_case factor names)."""
    result = rag.retrieve_remedial_content(["pct_assessments_submitted"], k=1)
    assert result["backend"] == "exact factor-tag match"
    matching_item = next(
        item for item in vectorstore.STARTER_REMEDIAL_CONTENT
        if "pct_assessments_submitted" in item["factor_tags"]
    )
    assert result["suggestions"] == [matching_item["text"]]


def test_get_risk_explanation_includes_rag_suggestions():
    """End-to-end: the real tool output (not just rag.py in isolation)
    carries remedial suggestions, grounded in that student's real top
    factors, with a matching real evidence_sources ID per suggestion."""
    result = tools.get_risk_explanation("uci", "0")
    assert "mitigation_strategies" in result
    assert len(result["mitigation_strategies"]) > 0
    assert len(result["evidence_sources"]) == len(result["mitigation_strategies"])
    real_ids = {item["id"] for item in vectorstore.STARTER_REMEDIAL_CONTENT}
    assert set(result["evidence_sources"]) <= real_ids


def test_retriever_is_cached_across_calls():
    """Second call should reuse the cached retriever, not rebuild the
    vectorstore -- same caching pattern as src/explainability.py's SHAP
    explainer cache."""
    rag.build_retriever()  # warm the cache
    assert rag._RETRIEVER_CACHE is not None
