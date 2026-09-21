# tests/test_ai_assistant/test_vectorstore.py
"""ai_assistant/vectorstore.py -- Sprint 5 preparation. Proves the embedding
pipeline runs end to end (Sprint 4's actual job here, per the plan's
Section 5.3), regardless of which embedding backend is available in the
current environment -- real sentence-transformers model if the network
allows it, the labeled hash fallback if not."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import vectorstore


def test_vectorstore_pipeline_runs_end_to_end():
    result = vectorstore.verify_pipeline()
    assert result["chroma_count"] == result["starter_content_count"]
    assert result["faiss_count"] == result["starter_content_count"]


def test_embedding_function_returns_consistent_dimension():
    """Same embedding function, same output dimension for every input --
    required for either vector store to accept the vectors."""
    embed_fn, backend = vectorstore.get_embedding_function()
    vectors = embed_fn(["short text", "a longer piece of text with more words"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert backend  # never empty -- always reports which backend was used


def test_hash_embedding_is_deterministic():
    """The offline fallback must be reproducible -- same text in, same
    vector out, every time -- since it stands in for a real embedding
    model's determinism."""
    v1 = vectorstore._hash_embedding("a test sentence")
    v2 = vectorstore._hash_embedding("a test sentence")
    assert v1 == v2
