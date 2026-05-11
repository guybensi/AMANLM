import uuid
from datetime import datetime

from backend.models.document import Chunk, DocumentMeta
from backend.utils.text_utils import recursive_chunk
from backend.config import settings, TESSERACT_AVAILABLE


def process_pdf(file_bytes: bytes, filename: str, doc_id: str) -> tuple[list[Chunk], int]:
    """Parse a PDF and return its text chunks and raw byte size.

    Iterates over each page using PyMuPDF, skips blank pages, and passes each
    page's text through ``recursive_chunk()`` to produce overlapping word windows.

    Args:
        file_bytes (bytes): Raw PDF file content as read from an upload.
        filename (str): Original filename used to populate chunk metadata.
        doc_id (str): UUID string identifying the parent document.

    Returns:
        tuple[list[Chunk], int]: A pair of (chunks, size_in_bytes) where
            ``chunks`` is the ordered list of ``Chunk`` objects across all pages
            and ``size_in_bytes`` is ``len(file_bytes)``.

    Example:
        >>> with open("report.pdf", "rb") as f:
        ...     chunks, size = process_pdf(f.read(), "report.pdf", "doc-001")
        >>> len(chunks)
        42
        >>> chunks[0].page_number
        1
    """
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
    """Parse a plain-text or Markdown file and return its chunks and byte size.

    Decodes the bytes as UTF-8 (replacing invalid sequences) and passes the
    full text through ``recursive_chunk()``.  All chunks are assigned
    ``page_number=1`` since plain-text files have no page concept.

    Args:
        file_bytes (bytes): Raw file content.
        filename (str): Original filename stored in chunk metadata.
        doc_id (str): UUID string identifying the parent document.

    Returns:
        tuple[list[Chunk], int]: ``(chunks, len(file_bytes))``.

    Example:
        >>> chunks, size = process_txt(b"Hello world.", "notes.txt", "doc-002")
        >>> chunks[0].text
        'Hello world.'
        >>> chunks[0].page_number
        1
    """
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
    """Parse a DOCX file and return its text chunks and byte size.

    Extracts non-empty paragraph text with ``python-docx``, joins paragraphs
    with newlines, and passes the result through ``recursive_chunk()``.
    All chunks receive ``page_number=1`` since DOCX pagination is not
    extracted at this stage.

    Args:
        file_bytes (bytes): Raw DOCX file content.
        filename (str): Original filename stored in chunk metadata.
        doc_id (str): UUID string identifying the parent document.

    Returns:
        tuple[list[Chunk], int]: ``(chunks, len(file_bytes))``.

    Example:
        >>> with open("paper.docx", "rb") as f:
        ...     chunks, size = process_docx(f.read(), "paper.docx", "doc-003")
        >>> chunks[0].filename
        'paper.docx'
    """
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
    """Extract text from an image via Tesseract OCR and return chunks and byte size.

    Requires Tesseract to be installed and discoverable on ``PATH``
    (``TESSERACT_AVAILABLE`` must be ``True``).  Returns an empty chunk list
    silently when Tesseract is unavailable so the document is still registered.

    Args:
        file_bytes (bytes): Raw image content (JPEG, PNG, or WebP).
        filename (str): Original filename stored in chunk metadata.
        doc_id (str): UUID string identifying the parent document.

    Returns:
        tuple[list[Chunk], int]: ``(chunks, len(file_bytes))``.  ``chunks`` is
            empty when Tesseract is not installed or the image contains no text.

    Example:
        >>> with open("scan.png", "rb") as f:
        ...     chunks, size = process_image(f.read(), "scan.png", "doc-004")
        >>> chunks[0].text  # OCR-extracted text
        'Invoice total: $1,200'
    """
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
    """Route a file to the appropriate processor and return its chunks and metadata.

    Resolves the file type first from ``content_type``, then from the file
    extension when the MIME type is unknown or generic.  Raises ``ValueError``
    for unsupported types so callers can return an HTTP 415 response.

    Args:
        file_bytes (bytes): Raw file content as received from an upload.
        filename (str): Original filename, used for extension fallback and metadata.
        content_type (str): MIME type reported by the upload (e.g. ``"text/plain"``).

    Returns:
        tuple[list[Chunk], DocumentMeta]: ``(chunks, meta)`` where ``meta``
            contains the auto-generated ``doc_id``, upload timestamp, file type,
            chunk count, and byte size.

    Raises:
        ValueError: When neither ``content_type`` nor the file extension maps
            to a supported processor.

    Example:
        >>> chunks, meta = dispatch(b"hello world", "notes.txt", "text/plain")
        >>> meta.file_type
        'txt'
        >>> meta.chunk_count == len(chunks)
        True

        >>> dispatch(b"", "file.xyz", "application/unknown")
        ValueError: Unsupported file type: file.xyz (application/unknown)
    """
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
