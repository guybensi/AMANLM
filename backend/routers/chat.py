import time
from fastapi import APIRouter, HTTPException

from backend.models.chat import ChatRequest, ChatResponse
from backend.services.rag_service import retrieve, calculate_confidence, confidence_label, build_source_proofs
from backend.services.llm_service import build_prompt, chat_completion
from backend.services.vector_store import vector_store
from backend.config import settings

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    stats = vector_store.stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload documents before asking questions."
        )

    # Retrieval
    t0 = time.time()
    scored_chunks = retrieve(request.message, top_k=request.top_k)
    retrieval_ms = int((time.time() - t0) * 1000)

    if not scored_chunks:
        raise HTTPException(status_code=500, detail="Retrieval failed unexpectedly.")

    # Confidence + sources
    confidence = calculate_confidence(scored_chunks)
    label = confidence_label(confidence)
    sources = build_source_proofs(scored_chunks)

    # LLM call
    t1 = time.time()
    messages = build_prompt(request.message, scored_chunks, request.mode)
    answer = chat_completion(messages)
    llm_ms = int((time.time() - t1) * 1000)

    return ChatResponse(
        answer=answer,
        mode=request.mode,
        confidence=confidence,
        confidence_label=label,
        sources=sources,
        retrieval_time_ms=retrieval_ms,
        llm_time_ms=llm_ms,
        model_used=settings.groq_model,
    )
