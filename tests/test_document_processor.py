import io
import pytest
from unittest.mock import patch, MagicMock
from backend.services.document_processor import (
    process_txt,
    process_docx,
    process_image,
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


class TestProcessImageVisionPipeline:
    """Tests for the two-stage vision model → Tesseract fallback in process_image."""

    def test_uses_vision_model_as_primary(self):
        with patch("backend.services.vision_service.describe_image", return_value="Vision result"):
            chunks, size = process_image(b"img", "photo.png", "doc1")
        assert len(chunks) == 1
        assert chunks[0].text == "Vision result"

    def test_vision_result_sets_correct_metadata(self):
        with patch("backend.services.vision_service.describe_image", return_value="Some text"):
            chunks, _ = process_image(b"img", "scan.jpg", "docX")
        assert chunks[0].filename == "scan.jpg"
        assert chunks[0].page_number == 1
        assert chunks[0].doc_id == "docX"

    def test_falls_back_to_tesseract_when_vision_raises(self):
        with patch("backend.services.vision_service.describe_image", side_effect=Exception("API down")), \
             patch("backend.services.document_processor.TESSERACT_AVAILABLE", True), \
             patch("pytesseract.image_to_string", return_value="OCR result"), \
             patch("PIL.Image.open"):
            chunks, _ = process_image(b"img", "scan.png", "doc2")
        # Tesseract path reached; chunk text comes from OCR
        assert any("OCR result" in c.text for c in chunks)

    def test_returns_empty_when_vision_raises_and_tesseract_unavailable(self):
        with patch("backend.services.vision_service.describe_image", side_effect=Exception("API down")), \
             patch("backend.services.document_processor.TESSERACT_AVAILABLE", False):
            chunks, size = process_image(b"img", "photo.webp", "doc3")
        assert chunks == []

    def test_returns_empty_when_vision_returns_blank_and_tesseract_unavailable(self):
        with patch("backend.services.vision_service.describe_image", return_value="   "), \
             patch("backend.services.document_processor.TESSERACT_AVAILABLE", False):
            chunks, _ = process_image(b"img", "blank.png", "doc4")
        assert chunks == []

    def test_chunk_indices_are_sequential(self):
        long_text = " ".join([f"w{i}" for i in range(600)])
        with patch("backend.services.vision_service.describe_image", return_value=long_text):
            chunks, _ = process_image(b"img", "big.png", "doc5")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestProcessPdfImageExtraction:
    """Tests for PDF image extraction and vision integration in process_pdf."""

    def _make_fitz_page(self, text="", images=None):
        """Build a minimal PyMuPDF page mock."""
        page = MagicMock()
        page.get_text.return_value = text
        page.get_images.return_value = images or []
        return page

    def _make_fitz_doc(self, pages):
        doc = MagicMock()
        doc.__iter__ = MagicMock(return_value=iter(enumerate(pages, start=1)))
        doc.__enter__ = MagicMock(return_value=doc)
        doc.__exit__ = MagicMock(return_value=False)
        return doc

    def test_text_only_pdf_produces_text_chunks(self):
        from backend.services.document_processor import process_pdf
        with patch("fitz.open") as mock_fitz, \
             patch("backend.services.vision_service.describe_image"):
            page = MagicMock()
            page.get_text.return_value = "Hello world text."
            page.get_images.return_value = []
            mock_doc = MagicMock()
            # enumerate(doc) yields (index, item) — so mock_doc must iterate plain pages
            mock_doc.__iter__ = MagicMock(return_value=iter([page]))
            mock_doc.close = MagicMock()
            mock_fitz.return_value = mock_doc

            chunks, size = process_pdf(b"pdf", "doc.pdf", "docA")
        assert any("Hello world text." in c.text for c in chunks)

    def test_image_chunk_contains_label_prefix(self):
        from backend.services.document_processor import process_pdf
        with patch("fitz.open") as mock_fitz, \
             patch("backend.services.vision_service.describe_image", return_value="A pie chart showing sales data"):
            page = MagicMock()
            page.get_text.return_value = ""
            page.get_images.return_value = [(1, 0, 0, 0, 0, 0, 0)]  # one image
            mock_doc = MagicMock()
            mock_doc.__iter__ = MagicMock(return_value=iter([page]))
            mock_doc.extract_image.return_value = {
                "image": b"img_bytes",
                "ext": "png",
                "width": 300,
                "height": 200,
            }
            mock_doc.close = MagicMock()
            mock_fitz.return_value = mock_doc

            chunks, _ = process_pdf(b"pdf", "doc.pdf", "docB")
        assert any("[Image on page 1]:" in c.text for c in chunks)

    def test_tiny_images_are_skipped(self):
        from backend.services.document_processor import process_pdf
        with patch("fitz.open") as mock_fitz, \
             patch("backend.services.vision_service.describe_image") as mock_vision:
            page = MagicMock()
            page.get_text.return_value = ""
            page.get_images.return_value = [(1, 0, 0, 0, 0, 0, 0)]
            mock_doc = MagicMock()
            mock_doc.__iter__ = MagicMock(return_value=iter([page]))
            mock_doc.extract_image.return_value = {
                "image": b"tiny",
                "ext": "png",
                "width": 50,   # below the 100px threshold
                "height": 50,
            }
            mock_doc.close = MagicMock()
            mock_fitz.return_value = mock_doc

            process_pdf(b"pdf", "doc.pdf", "docC")
        mock_vision.assert_not_called()

    def test_failed_image_does_not_abort_pdf_processing(self):
        from backend.services.document_processor import process_pdf
        with patch("fitz.open") as mock_fitz, \
             patch("backend.services.vision_service.describe_image", side_effect=Exception("API down")):
            page = MagicMock()
            page.get_text.return_value = "Some page text."
            page.get_images.return_value = [(1, 0, 0, 0, 0, 0, 0)]
            mock_doc = MagicMock()
            mock_doc.__iter__ = MagicMock(return_value=iter([page]))
            mock_doc.extract_image.return_value = {
                "image": b"img",
                "ext": "png",
                "width": 300,
                "height": 200,
            }
            mock_doc.close = MagicMock()
            mock_fitz.return_value = mock_doc

            # Should not raise
            chunks, _ = process_pdf(b"pdf", "doc.pdf", "docD")
        assert any("Some page text." in c.text for c in chunks)


class TestProcessDocx:
    """Tests for structured DOCX extraction (paragraphs + tables + heading markers)."""

    def _make_docx(self, paragraphs=None, tables=None):
        """Build a minimal python-docx Document mock."""
        doc = MagicMock()

        para_mocks = []
        for text, style_name in (paragraphs or []):
            p = MagicMock()
            p.text = text
            p.style.name = style_name
            para_mocks.append(p)
        doc.paragraphs = para_mocks

        table_mocks = []
        for rows_data in (tables or []):
            table = MagicMock()
            row_mocks = []
            for row_cells in rows_data:
                row = MagicMock()
                cell_mocks = []
                for cell_text in row_cells:
                    cell = MagicMock()
                    cell.text = cell_text
                    cell_mocks.append(cell)
                row.cells = cell_mocks
                row_mocks.append(row)
            table.rows = row_mocks
            table_mocks.append(table)
        doc.tables = table_mocks

        return doc

    def test_paragraph_text_extracted(self):
        doc = self._make_docx(paragraphs=[("Hello world.", "Normal")])
        with patch("docx.Document", return_value=doc):
            chunks, _ = process_docx(b"docx", "file.docx", "doc1")
        assert any("Hello world." in c.text for c in chunks)

    def test_heading_gets_hash_prefix(self):
        doc = self._make_docx(paragraphs=[("Introduction", "Heading 1")])
        with patch("docx.Document", return_value=doc):
            chunks, _ = process_docx(b"docx", "file.docx", "doc2")
        assert any("# Introduction" in c.text for c in chunks)

    def test_empty_paragraphs_skipped(self):
        doc = self._make_docx(paragraphs=[("  ", "Normal"), ("Real text.", "Normal")])
        with patch("docx.Document", return_value=doc):
            chunks, _ = process_docx(b"docx", "file.docx", "doc3")
        full = " ".join(c.text for c in chunks)
        assert "Real text." in full
        assert full.count("Real text.") == 1

    def test_table_rows_extracted_as_pipe_delimited(self):
        tables = [[["Name", "Age"], ["Alice", "30"]]]
        doc = self._make_docx(tables=tables)
        with patch("docx.Document", return_value=doc):
            chunks, _ = process_docx(b"docx", "file.docx", "doc4")
        full = " ".join(c.text for c in chunks)
        assert "Name | Age" in full
        assert "Alice | 30" in full

    def test_merged_cells_deduplicated(self):
        # python-docx repeats merged cell text across spanned cells
        tables = [[["Merged", "Merged", "Other"]]]
        doc = self._make_docx(tables=tables)
        with patch("docx.Document", return_value=doc):
            chunks, _ = process_docx(b"docx", "file.docx", "doc5")
        full = " ".join(c.text for c in chunks)
        # "Merged" should appear only once in the pipe-delimited row
        assert full.count("Merged") == 1

    def test_empty_table_cells_skipped(self):
        tables = [[["", "Value", ""]]]
        doc = self._make_docx(tables=tables)
        with patch("docx.Document", return_value=doc):
            chunks, _ = process_docx(b"docx", "file.docx", "doc6")
        full = " ".join(c.text for c in chunks)
        assert "Value" in full
        assert "| |" not in full

    def test_paragraphs_and_tables_combined(self):
        doc = self._make_docx(
            paragraphs=[("Summary paragraph.", "Normal")],
            tables=[[["Col1", "Col2"], ["A", "B"]]],
        )
        with patch("docx.Document", return_value=doc):
            chunks, _ = process_docx(b"docx", "file.docx", "doc7")
        full = " ".join(c.text for c in chunks)
        assert "Summary paragraph." in full
        assert "Col1 | Col2" in full

    def test_chunk_indices_are_sequential(self):
        long_text = " ".join([f"word{i}" for i in range(600)])
        doc = self._make_docx(paragraphs=[(long_text, "Normal")])
        with patch("docx.Document", return_value=doc):
            chunks, _ = process_docx(b"docx", "file.docx", "doc8")
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_size_bytes_matches_input(self):
        doc = self._make_docx(paragraphs=[("text", "Normal")])
        content = b"fake-docx-bytes"
        with patch("docx.Document", return_value=doc):
            _, size = process_docx(content, "file.docx", "doc9")
        assert size == len(content)
