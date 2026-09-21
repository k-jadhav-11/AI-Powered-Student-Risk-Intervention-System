

from langsmith import traceable
from langchain_core.embeddings import Embeddings

from . import vectorstore


# ============================================================
# LangChain Embedding Wrapper
# ============================================================

class _CallableEmbeddings(Embeddings):
    """Wraps the existing embedding function as a LangChain Embeddings object.
    Supports document and query embeddings for the Chroma vector store."""

    def __init__(self, embed_fn):
        """Initializes the wrapper with the embedding function.
        The function is reused for both documents and queries."""
        self._embed_fn = embed_fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for multiple document texts.
        Returns one vector for each input document."""

        # Generate embeddings for all documents
        return self._embed_fn(texts)

    def embed_query(self, text: str) -> list[float]:
        """Generates an embedding for a single search query.
        Returns the resulting embedding vector."""

        # Generate one embedding and return the first result
        return self._embed_fn([text])[0]


# ============================================================
# Retriever Cache
# ============================================================

# Cache the retriever so the vector store does not need to be rebuilt
# for every user question.
_RETRIEVER_CACHE = None

# Store which embedding backend is being used
_RETRIEVER_BACKEND = None


# ============================================================
# Build Retriever
# ============================================================

@traceable(name="build_retriever", run_type="tool")
def build_retriever(k: int = 2):
    """Builds and caches a LangChain retriever using ChromaDB.
    Returns the retriever and the embedding backend in use."""

    # Use the cached retriever if it has already been created
    global _RETRIEVER_CACHE, _RETRIEVER_BACKEND
    if _RETRIEVER_CACHE is not None:
        return _RETRIEVER_CACHE, _RETRIEVER_BACKEND

    # Import Chroma only when the retriever needs to be created
    from langchain_chroma import Chroma

    # Get the embedding function and active backend
    embed_fn, backend = vectorstore.get_embedding_function()

    # Extract remedial text from the knowledge base
    texts = [item["text"] for item in vectorstore.STARTER_REMEDIAL_CONTENT]

    # Store factor tags and source IDs as Chroma metadata
    metadatas = [{"id": item["id"], "factor_tags": ",".join(item["factor_tags"])}
                 for item in vectorstore.STARTER_REMEDIAL_CONTENT]

    # Create the Chroma vector store
    store = Chroma.from_texts(
        texts=texts,
        embedding=_CallableEmbeddings(embed_fn),
        metadatas=metadatas,
        collection_name="remedial_content_rag",
    )

    # Convert the vector store into a retriever
    _RETRIEVER_CACHE = store.as_retriever(search_kwargs={"k": k})

    # Save the embedding backend used by the retriever
    _RETRIEVER_BACKEND = backend

    # Return the cached retriever and backend
    return _RETRIEVER_CACHE, _RETRIEVER_BACKEND


# ============================================================
# Warm Up
# ============================================================

def warm_up(k: int = 3) -> str:
    """Initializes the cached retriever and embedding model.
    Returns the embedding backend used by the retriever."""

    _, backend = build_retriever(k=k)
    return backend


# ============================================================
# Retrieve Remedial Content
# ============================================================

@traceable(name="retrieve_remedial_content", run_type="retriever")
def retrieve_remedial_content(top_factors: list[str], k: int = 3) -> dict:
    """Retrieves remedial suggestions using factor matching and vector similarity.
    Returns suggestions, source IDs, and the retrieval backend used."""

    # Map each remedial text to its real source ID
    id_by_text = {item["text"]: item["id"] for item in vectorstore.STARTER_REMEDIAL_CONTENT}

    # Find remedial content whose factor tags match the student's factors
    tagged_matches = [
        item["text"] for item in vectorstore.STARTER_REMEDIAL_CONTENT
        if set(item["factor_tags"]) & set(top_factors)
    ]

    # If enough exact matches exist, return them without similarity search
    if len(tagged_matches) >= k:
        suggestions = tagged_matches[:k]

        return {
            "suggestions": suggestions,
            "source_ids": [id_by_text[s] for s in suggestions],
            "backend": "exact factor-tag match",
        }

    # Build the retriever when more suggestions are required
    retriever, backend = build_retriever(k=k)

    # Convert the student's top factors into a search query
    query = " ".join(top_factors)

    # Retrieve the most relevant remedial documents
    docs = retriever.invoke(query)

    # Remove documents that were already found through exact factor matching
    remaining = [
        d.page_content
        for d in docs
        if d.page_content not in tagged_matches
    ]

    # Combine exact matches with similarity-based matches
    suggestions = tagged_matches + remaining[: k - len(tagged_matches)]

    # Report whether only exact matching was used or similarity retrieval was used
    reported_backend = (
        "exact factor-tag match"
        if not remaining[: k - len(tagged_matches)]
        else backend
    )

    # Return suggestions, their source IDs, and retrieval backend
    return {
        "suggestions": suggestions,
        "source_ids": [id_by_text[s] for s in suggestions],
        "backend": reported_backend,
    }