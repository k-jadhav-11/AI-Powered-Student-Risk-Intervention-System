
import hashlib
from functools import lru_cache

import numpy as np
from langsmith import traceable

STARTER_REMEDIAL_CONTENT = [
    {"id": "rem_001", "factor_tags": ["failures", "g1", "g2"],
     "text": "Struggling with algebra or core subject basics: review foundational concepts and practice "
             "with step-by-step worked examples before moving to word problems or advanced material."},
    {"id": "rem_002", "factor_tags": ["engagement_trend", "num_weeks_active", "days_active_mean"],
     "text": "Low or declining engagement in an online course: set a fixed weekly check-in time, "
             "even just 15 minutes, since consistency matters more than session length."},
    {"id": "rem_003", "factor_tags": ["pct_assessments_submitted"],
     "text": "Missed multiple assignment deadlines: talk to the instructor about a "
             "catch-up plan before the gap becomes too large to close in one push."},
    {"id": "rem_004", "factor_tags": ["failures"],
     "text": "Repeated course failures: consider a study group or peer tutor for the "
             "specific topics that keep coming up as weak points, rather than restarting cold."},
    {"id": "rem_005", "factor_tags": ["absences"],
     "text": "High or rising absences: identify the specific barrier (schedule "
             "conflict, health, motivation) before assuming more of the same intervention will help."},
    {"id": "rem_006", "factor_tags": ["g1", "g2"],
     "text": "Low first- or second-period grades (g1/g2): schedule a short check-in with the subject "
             "teacher to pinpoint exactly which topic the grade drop traces back to, rather than "
             "reviewing the whole syllabus broadly."},
    {"id": "rem_007", "factor_tags": ["studytime"],
     "text": "Low reported weekly study time: a fixed, short daily block (even 20-30 minutes) tends to "
             "stick better than an irregular longer session once or twice a week."},
    {"id": "rem_008", "factor_tags": ["goout", "dalc", "walc"],
     "text": "High social-outing or leisure-alcohol-use scores alongside falling grades: this is worth "
             "a supportive, non-judgmental conversation about balance rather than a purely academic "
             "intervention -- the risk factor here is about time and attention, not ability."},
    {"id": "rem_009", "factor_tags": ["num_weeks_active", "total_clicks_sum", "days_active_sum"],
     "text": "Very few active weeks on the course platform: a short welcome-back message plus a specific, "
             "small first task (not a full module) lowers the barrier to re-engaging more than a "
             "general reminder does."},
    {"id": "rem_010", "factor_tags": ["engagement_trend"],
     "text": "Engagement trending downward over the course: early terms usually predict this better than "
             "later ones, so a check-in as soon as a downward trend appears is more effective than "
             "waiting for a bigger drop."},
    {"id": "rem_011", "factor_tags": ["schoolsup_yes", "famsup_yes"],
     "text": "No school or family educational support currently in place: connecting the student with "
             "existing school support resources (tutoring, counseling) is often the highest-leverage "
             "single step available."},
    {"id": "rem_012", "factor_tags": ["absences", "num_weeks_active"],
     "text": "Persona note -- students matching a 'quietly disengaged' or 'repeat-attempt struggling' "
             "profile (see the Clustering Explorer / Performance Personas page) tend to respond better "
             "to a proactive, low-pressure check-in than to a formal warning, since the pattern is "
             "usually disengagement rather than active failure."},
]

_EMBED_DIM = 64


def _hash_embedding(text: str, dim: int = _EMBED_DIM) -> list[float]:
    """Creates a deterministic hash-based embedding for offline fallback.
    Returns a normalized vector with the specified dimensionality."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()


@lru_cache(maxsize=1)
@traceable(name="load_sentence_transformer_model", run_type="tool")
def _load_sentence_transformer():
    """Loads the SentenceTransformer model once and reuses it across calls.
    Raises an exception if the model cannot be loaded."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding_function():
    """Returns an embedding function and the backend used to generate embeddings.
    Uses SentenceTransformer when available, otherwise falls back to hash embeddings."""
    try:
        model = _load_sentence_transformer()

        def embed_fn(texts: list[str]) -> list[list[float]]:
            return model.encode(texts).tolist()

        return embed_fn, "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        def embed_fn(texts: list[str]) -> list[list[float]]:
            return [_hash_embedding(t) for t in texts]

        return embed_fn, "hash-fallback (offline placeholder, not semantic)"


def build_chroma_collection(collection_name: str = "remedial_content_starter"):
    """Builds an in-memory ChromaDB collection from the starter remedial content.
    Returns the collection and the embedding backend used."""
    import chromadb

    embed_fn, backend = get_embedding_function()
    client = chromadb.Client()  # ephemeral, in-memory client
   
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    embeddings = embed_fn([item["text"] for item in STARTER_REMEDIAL_CONTENT])
    collection.add(
        ids=[item["id"] for item in STARTER_REMEDIAL_CONTENT],
        documents=[item["text"] for item in STARTER_REMEDIAL_CONTENT],
        embeddings=embeddings,
    )
    return collection, backend


def build_faiss_index():
    """Builds a FAISS IndexFlatL2 index from the starter remedial content.
    Returns the index and the embedding backend used."""
    import faiss

    embed_fn, backend = get_embedding_function()
    embeddings = np.array(embed_fn([item["text"] for item in STARTER_REMEDIAL_CONTENT]), dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, backend


def verify_pipeline() -> dict:
    """Runs the ChromaDB and FAISS embedding pipelines as an end-to-end check.
    Returns backend names and document counts for both vector stores."""
    collection, chroma_backend = build_chroma_collection()
    index, faiss_backend = build_faiss_index()
    return {
        "chroma_backend": chroma_backend,
        "chroma_count": collection.count(),
        "faiss_backend": faiss_backend,
        "faiss_count": index.ntotal,
        "starter_content_count": len(STARTER_REMEDIAL_CONTENT),
    }