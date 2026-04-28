from typing import Literal
from pydantic import BaseModel, Field


class SourceProof(BaseModel):
    filename: str
    page_number: int
    chunk_index: int
    quote: str
    relevance_score: float


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    mode: Literal["short", "long"] = "short"
    top_k: int = Field(5, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["short", "long"]
    confidence: float
    confidence_label: Literal["High", "Medium", "Low", "Very Low"]
    sources: list[SourceProof]
    retrieval_time_ms: int
    llm_time_ms: int
    model_used: str
