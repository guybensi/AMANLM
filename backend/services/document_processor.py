import uuid
from datetime import datetime

from backend.models.document import Chunk, DocumentMeta
from backend.utils.text_utils import recursive_chunk
from backend.config import settings, TESSERACT_AVAILABLE


def process_pdf(file_bytes: bytes, filename: str, doc_id: str) -> tuple[list[Chunk], int]:
    import fitz  # PyMuPDF
    chunks = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text.strip():
            continue
        page_chunks = recursive_chunk(text, settings.chunk_size, settings.chunk_overlap)
        for idx, chunk_text in enumerate(page_chunks):
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                filename=filename,
                page_number=page_num,
                chunk_index=idx,
                text=chunk_text,
            ))
    doc.close()
    return chunks, len(file_bytes)


def process_txt(file_bytes: bytes, filename: str, doc_id: str) -> tuple[list[Chunk], int]:
    text = file_bytes.decode("utf-8", errors="replace")
    raw_chunks = recursive_chunk(text, settings.chunk_size, settings.chunk_overlap)
    chunks = [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            filename=filename,
            page_number=1,
            chunk_index=idx,
            text=chunk_text,
        )
        for idx, chunk_text in enumerate(raw_chunks)
    ]
    return chunks, len(file_bytes)


def process_docx(file_bytes: bytes, filename: str, doc_id: str) -> tuple[list[Chunk], int]:
    import io
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    raw_chunks = recursive_chunk(full_text, settings.chunk_size, settings.chunk_overlap)
    chunks = [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            filename=filename,
            page_number=1,
            chunk_index=idx,
            text=chunk_text,
        )
        for idx, chunk_text in enumerate(raw_chunks)
    ]
    return chunks, len(file_bytes)


def process_image(file_bytes: bytes, filename: str, doc_id: str) -> tuple[list[Chunk], int]:
    if not TESSERACT_AVAILABLE:
        return [], len(file_bytes)
    import io
    import pytesseract
    from PIL import Image
    image = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(image)
    raw_chunks = recursive_chunk(text, settings.chunk_size, settings.chunk_overlap)
    chunks = [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            filename=filename,
            page_number=1,
            chunk_index=idx,
            text=chunk_text,
        )
        for idx, chunk_text in enumerate(raw_chunks)
    ]
    return chunks, len(file_bytes)


SUPPORTED_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
}

EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "txt",
    ".docx": "docx",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
}


def dispatch(
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> tuple[list[Chunk], DocumentMeta]:
    doc_id = str(uuid.uuid4())

    # Resolve type from content_type or extension
    ftype = SUPPORTED_TYPES.get(content_type)
    if not ftype:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        ftype = EXTENSION_MAP.get(ext)
    if not ftype:
        raise ValueError(f"Unsupported file type: {filename} ({content_type})")

    processors = {
        "pdf": process_pdf,
        "txt": process_txt,
        "docx": process_docx,
        "image": process_image,
    }
    chunks, size_bytes = processors[ftype](file_bytes, filename, doc_id)

    meta = DocumentMeta(
        doc_id=doc_id,
        filename=filename,
        file_type=ftype,
        chunk_count=len(chunks),
        upload_time=datetime.utcnow(),
        size_bytes=size_bytes,
    )
    return chunks, meta
