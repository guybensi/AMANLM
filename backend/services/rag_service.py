import numpy as np
from dataclasses import dataclass

from backend.models.chat import SourceProof
from backend.models.document import Chunk
from backend.services.embedding_service import embedding_service
from backend.services.vector_store import vector_store


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def retrieve(query: str, top_k: int = 5) -> list[ScoredChunk]:
    """Retrieve the top-k most relevant chunks for a query using cosine similarity.

    Encodes the query with the embedding service, computes dot-product similarity
    against all stored (L2-normalised) chunk embeddings, and returns the highest-
    scoring chunks in descending order.

    Args:
        query (str): The user's natural-language question or search string.
        top_k (int): Maximum number of chunks to return. Defaults to 5.
            Clamped to the total number of stored chunks when fewer exist.

    Returns:
        list[ScoredChunk]: Ranked list of ``ScoredChunk`` objects, each containing
            the matched ``Chunk`` and its float similarity ``score``.
            Returns an empty list when no documents have been uploaded.

    Example:
        >>> results = retrieve("What is the capital of France?", top_k=3)
        >>> results[0].score  # highest similarity score
        0.82
        >>> results[0].chunk.filename
        'europe_guide.pdf'
    """
    all_embeddings = vector_store.get_all_embeddings()
    if all_embeddings.shape[0] == 0:
        return []

    query_embedding = embedding_service.encode_single(query)
    scores = np.dot(all_embeddings, query_embedding)
    k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:k]

    return [
        ScoredChunk(chunk=vector_store.get_chunk(int(i)), score=float(scores[i]))
        for i in top_indices
    ]


def calculate_confidence(scored_chunks: list[ScoredChunk]) -> float:
    """Compute an aggregate confidence score (0–100) from a list of scored chunks.

    Combines three weighted sub-scores:

    * **Retrieval** (60 %): maps the top chunk's similarity into [0, 100] using a
      calibrated floor/range designed for multilingual models.
    * **Coverage** (25 %): fraction of chunks whose score exceeds a low relevance
      threshold, rewarding broad document coverage.
    * **Consistency** (15 %): penalises high score variance across chunks, rewarding
      topics that are well-represented throughout the source material.

    Args:
        scored_chunks (list[ScoredChunk]): Ranked chunks returned by ``retrieve()``.
            The list is expected to be non-empty; an empty list returns 0.0.

    Returns:
        float: Confidence percentage rounded to one decimal place, clamped to [0.0, 100.0].

    Example:
        >>> chunks = retrieve("photosynthesis", top_k=5)
        >>> score = calculate_confidence(chunks)
        >>> 0.0 <= score <= 100.0
        True

        >>> calculate_confidence([])
        0.0
    """
    if not scored_chunks:
        return 0.0

    top_score = scored_chunks[0].score
    scores = [sc.score for sc in scored_chunks]

    # Calibrated for multilingual models (Hebrew, Arabic, etc.) where
    # strong matches score 0.30-0.55 rather than 0.60-0.90 like English.
    # Map [0.05, 0.45] → [0, 100]  so a score of 0.30+ reads as High.
    FLOOR = 0.05
    RANGE = 0.40
    retrieval_score = max(0.0, (top_score - FLOOR) / RANGE) * 100

    # Coverage: chunks above a low realistic threshold
    relevant = sum(1 for s in scores if s > 0.12)
    coverage_score = (relevant / len(scores)) * 100

    # Consistency: low spread across top-k = topic is well-covered
    std = float(np.std(scores))
    consistency_score = max(0.0, (0.25 - std) / 0.25) * 100

    confidence = 0.60 * retrieval_score + 0.25 * coverage_score + 0.15 * consistency_score
    return round(min(100.0, max(0.0, confidence)), 1)


def confidence_label(score: float) -> str:
    """Map a numeric confidence score to a human-readable label.

    Thresholds:
        * **High** — score ≥ 65
        * **Medium** — 40 ≤ score < 65
        * **Low** — 20 ≤ score < 40
        * **Very Low** — score < 20

    Args:
        score (float): Confidence percentage in the range [0.0, 100.0], as
            returned by ``calculate_confidence()``.

    Returns:
        str: One of ``"High"``, ``"Medium"``, ``"Low"``, or ``"Very Low"``.

    Example:
        >>> confidence_label(80.0)
        'High'
        >>> confidence_label(45.0)
        'Medium'
        >>> confidence_label(10.0)
        'Very Low'
    """
    if score >= 65:
        return "High"
    elif score >= 40:
        return "Medium"
    elif score >= 20:
        return "Low"
    return "Very Low"


def build_source_proofs(scored_chunks: list[ScoredChunk]) -> list[SourceProof]:
    """Build deduplicated source-proof cards from a ranked list of scored chunks.

    Each unique ``(doc_id, chunk_index)`` pair produces one ``SourceProof``.
    Chunk text is truncated to 400 characters and suffixed with ``…`` when
    longer, giving the UI a compact verbatim quote.

    Args:
        scored_chunks (list[ScoredChunk]): Ranked chunks from ``retrieve()``.
            Duplicate ``(doc_id, chunk_index)`` pairs are silently skipped after
            the first occurrence.

    Returns:
        list[SourceProof]: Ordered list of source-proof objects, one per unique
            chunk, preserving the original ranking order.

    Example:
        >>> chunks = retrieve("climate change", top_k=5)
        >>> proofs = build_source_proofs(chunks)
        >>> proofs[0].filename
        'ipcc_report.pdf'
        >>> proofs[0].relevance_score
        0.741
    """
    seen = set()
    proofs = []
    for sc in scored_chunks:
        key = (sc.chunk.doc_id, sc.chunk.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        quote = sc.chunk.text[:400].strip()
        if len(sc.chunk.text) > 400:
            quote += "…"
        proofs.append(SourceProof(
            filename=sc.chunk.filename,
            page_number=sc.chunk.page_number,
            chunk_index=sc.chunk.chunk_index,
            quote=quote,
            relevance_score=round(sc.score, 3),
        ))
    return proofs
