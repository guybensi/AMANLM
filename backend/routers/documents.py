import time
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.models.document import UploadResponse, DocumentMeta, DocumentsStatus
from backend.services.document_processor import dispatch
from backend.services.embedding_service import embedding_service
from backend.services.vector_store import vector_store
from backend.config import settings, TESSERACT_AVAILABLE

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    start = time.time()
    doc_ids, filenames, chunk_counts = [], [], []
    total_chunks = 0

    for upload in files:
        file_bytes = await upload.read()
        content_type = upload.content_type or ""
        filename = upload.filename or "unknown"

        try:
            chunks, meta = dispatch(file_bytes, filename, content_type)
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))

        if chunks:
            texts = [c.text for c in chunks]
            embeddings = embedding_service.encode(texts)
            vector_store.add(chunks, embeddings, meta)
        else:
            # Still register the doc even with 0 chunks (e.g. image without Tesseract)
            from numpy import empty
            vector_store.add([], empty((0, 384)), meta)

        doc_ids.append(meta.doc_id)
        filenames.append(meta.filename)
        chunk_counts.append(meta.chunk_count)
        total_chunks += meta.chunk_count

    vector_store.save_cache()
    elapsed_ms = int((time.time() - start) * 1000)

    return UploadResponse(
        doc_ids=doc_ids,
        filenames=filenames,
        chunk_counts=chunk_counts,
        total_chunks=total_chunks,
        processing_time_ms=elapsed_ms,
    )


@router.get("", response_model=list[DocumentMeta])
async def list_documents():
    return vector_store.get_all_metas()


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    removed = vector_store.remove_doc(doc_id)
    if removed == 0 and doc_id not in [m.doc_id for m in vector_store.get_all_metas()]:
        raise HTTPException(status_code=404, detail="Document not found")
    vector_store.save_cache()
    return {"removed_chunks": removed}


@router.get("/status", response_model=DocumentsStatus)
async def get_status():
    stats = vector_store.stats()
    return DocumentsStatus(
        total_docs=stats["total_docs"],
        total_chunks=stats["total_chunks"],
        embedding_model=settings.embedding_model,
        memory_mb=stats["memory_mb"],
        tesseract_available=TESSERACT_AVAILABLE,
    )
