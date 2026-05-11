import pytest
from unittest.mock import patch
from backend.services.document_processor import (
    process_txt,
    dispatch,
    SUPPORTED_TYPES,
    EXTENSION_MAP,
)


class TestProcessTxt:
    def test_returns_chunks_and_size(self):
        content = b"Hello world. This is a test document."
        chunks, size = process_txt(content, "test.txt", "doc123")
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert size == len(content)

    def test_chunk_fields_are_set_correctly(self):
        content = b"Sample text content."
        chunks, _ = process_txt(content, "notes.txt", "docABC")
        chunk = chunks[0]
        assert chunk.doc_id == "docABC"
        assert chunk.filename == "notes.txt"
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0
        assert chunk.text == "Sample text content."

    def test_empty_file_returns_no_chunks(self):
        chunks, size = process_txt(b"", "empty.txt", "doc0")
        assert chunks == []
        assert size == 0

    def test_whitespace_only_returns_no_chunks(self):
        chunks, _ = process_txt(b"   \n\t  ", "blank.txt", "doc0")
        assert chunks == []

    def test_large_text_produces_multiple_chunks(self):
        # ~600 words should produce 2 chunks with default chunk_size=512
        content = (" ".join([f"word{i}" for i in range(600)])).encode()
        chunks, _ = process_txt(content, "big.txt", "docX")
        assert len(chunks) >= 2

    def test_chunk_indices_are_sequential(self):
        content = (" ".join([f"w{i}" for i in range(600)])).encode()
        chunks, _ = process_txt(content, "test.txt", "docY")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_ids_are_unique(self):
        content = (" ".join([f"w{i}" for i in range(600)])).encode()
        chunks, _ = process_txt(content, "test.txt", "docZ")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_decodes_utf8_content(self):
        content = "héllo wörld".encode("utf-8")
        chunks, _ = process_txt(content, "utf8.txt", "docU")
        assert len(chunks) > 0
        assert "héllo" in chunks[0].text or "wörld" in chunks[0].text


class TestDispatch:
    def test_dispatch_txt_by_content_type(self):
        content = b"plain text"
        chunks, meta = dispatch(content, "file.txt", "text/plain")
        assert meta.file_type == "txt"
        assert meta.filename == "file.txt"
        assert meta.size_bytes == len(content)

    def test_dispatch_txt_by_extension_when_content_type_unknown(self):
        content = b"some markdown"
        chunks, meta = dispatch(content, "notes.txt", "application/octet-stream")
        assert meta.file_type == "txt"

    def test_dispatch_markdown_by_content_type(self):
        content = b"# Title\n\nSome text."
        chunks, meta = dispatch(content, "doc.md", "text/markdown")
        assert meta.file_type == "txt"

    def test_dispatch_markdown_by_extension(self):
        content = b"# Heading"
        chunks, meta = dispatch(content, "readme.md", "application/octet-stream")
        assert meta.file_type == "txt"

    def test_dispatch_unsupported_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            dispatch(b"data", "file.xyz", "application/unknown")

    def test_meta_doc_id_is_a_uuid_string(self):
        content = b"hello"
        _, meta = dispatch(content, "a.txt", "text/plain")
        import uuid
        assert uuid.UUID(meta.doc_id)  # raises if invalid

    def test_meta_chunk_count_matches_chunks(self):
        content = b"hello world"
        chunks, meta = dispatch(content, "a.txt", "text/plain")
        assert meta.chunk_count == len(chunks)

    def test_dispatch_pdf_by_extension_with_mock(self):
        """dispatch resolves 'pdf' type from extension; actual parsing is mocked."""
        mock_chunk = type("C", (), {})()
        with patch("backend.services.document_processor.process_pdf", return_value=([mock_chunk], 100)) as mock_pdf:
            chunks, meta = dispatch(b"pdf-bytes", "report.pdf", "application/octet-stream")
        mock_pdf.assert_called_once()
        assert meta.file_type == "pdf"
