from datetime import datetime
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    upload_time: datetime
    size_bytes: int


class UploadResponse(BaseModel):
    doc_ids: list[str]
    filenames: list[str]
    chunk_counts: list[int]
    total_chunks: int
    processing_time_ms: int


class DocumentsStatus(BaseModel):
    total_docs: int
    total_chunks: int
    embedding_model: str
    memory_mb: float
    tesseract_available: bool
