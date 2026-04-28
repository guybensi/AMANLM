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
    if score >= 65:
        return "High"
    elif score >= 40:
        return "Medium"
    elif score >= 20:
        return "Low"
    return "Very Low"


def build_source_proofs(scored_chunks: list[ScoredChunk]) -> list[SourceProof]:
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
