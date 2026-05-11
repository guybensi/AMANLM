import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from backend.models.document import Chunk, DocumentMeta
from backend.services.vector_store import VectorStore


def make_chunk(doc_id="doc1", chunk_index=0, filename="file.txt"):
    return Chunk(
        chunk_id=f"{doc_id}-{chunk_index}",
        doc_id=doc_id,
        filename=filename,
        page_number=1,
        chunk_index=chunk_index,
        text=f"chunk text {chunk_index}",
    )


def make_meta(doc_id="doc1", filename="file.txt", chunk_count=1):
    return DocumentMeta(
        doc_id=doc_id,
        filename=filename,
        file_type="txt",
        chunk_count=chunk_count,
        upload_time=datetime.utcnow(),
        size_bytes=100,
    )


def make_embeddings(n: int, dim: int = 384) -> np.ndarray:
    return np.random.rand(n, dim).astype(np.float32)


@pytest.fixture
def fresh_store():
    """Return a VectorStore with cache loading suppressed."""
    with patch.object(VectorStore, "_try_load_cache", return_value=None):
        store = VectorStore()
    return store


class TestVectorStoreAdd:
    def test_add_increases_chunk_count(self, fresh_store):
        chunks = [make_chunk(chunk_index=i) for i in range(3)]
        embeddings = make_embeddings(3)
        meta = make_meta(chunk_count=3)
        fresh_store.add(chunks, embeddings, meta)
        assert fresh_store.stats()["total_chunks"] == 3

    def test_add_increases_doc_count(self, fresh_store):
        fresh_store.add([make_chunk()], make_embeddings(1), make_meta())
        assert fresh_store.stats()["total_docs"] == 1

    def test_add_multiple_docs(self, fresh_store):
        for doc_id in ("doc1", "doc2"):
            fresh_store.add(
                [make_chunk(doc_id=doc_id)],
                make_embeddings(1),
                make_meta(doc_id=doc_id),
            )
        assert fresh_store.stats()["total_docs"] == 2
        assert fresh_store.stats()["total_chunks"] == 2

    def test_add_stacks_embeddings_correctly(self, fresh_store):
        fresh_store.add([make_chunk(doc_id="d1")], make_embeddings(1), make_meta(doc_id="d1"))
        fresh_store.add([make_chunk(doc_id="d2")], make_embeddings(1), make_meta(doc_id="d2"))
        assert fresh_store.get_all_embeddings().shape == (2, 384)

    def test_add_empty_embeddings_does_not_crash(self, fresh_store):
        fresh_store.add([], np.empty((0, 384), dtype=np.float32), make_meta(chunk_count=0))
        assert fresh_store.stats()["total_chunks"] == 0


class TestVectorStoreRemoveDoc:
    def test_remove_doc_decreases_chunk_count(self, fresh_store):
        chunks = [make_chunk(chunk_index=i) for i in range(2)]
        fresh_store.add(chunks, make_embeddings(2), make_meta(chunk_count=2))
        removed = fresh_store.remove_doc("doc1")
        assert removed == 2
        assert fresh_store.stats()["total_chunks"] == 0

    def test_remove_doc_decreases_doc_count(self, fresh_store):
        fresh_store.add([make_chunk()], make_embeddings(1), make_meta())
        fresh_store.remove_doc("doc1")
        assert fresh_store.stats()["total_docs"] == 0

    def test_remove_nonexistent_doc_returns_zero(self, fresh_store):
        assert fresh_store.remove_doc("no-such-doc") == 0

    def test_remove_only_removes_target_doc(self, fresh_store):
        fresh_store.add([make_chunk(doc_id="d1")], make_embeddings(1), make_meta(doc_id="d1"))
        fresh_store.add([make_chunk(doc_id="d2")], make_embeddings(1), make_meta(doc_id="d2"))
        fresh_store.remove_doc("d1")
        assert fresh_store.stats()["total_docs"] == 1
        assert fresh_store.stats()["total_chunks"] == 1
        remaining = fresh_store.get_all_chunks()
        assert all(c.doc_id == "d2" for c in remaining)

    def test_remove_all_docs_resets_embeddings_shape(self, fresh_store):
        fresh_store.add([make_chunk()], make_embeddings(1), make_meta())
        fresh_store.remove_doc("doc1")
        assert fresh_store.get_all_embeddings().shape == (0, 384)


class TestVectorStoreStats:
    def test_empty_store_stats(self, fresh_store):
        s = fresh_store.stats()
        assert s["total_docs"] == 0
        assert s["total_chunks"] == 0
        assert s["memory_mb"] == 0.0

    def test_memory_mb_increases_after_add(self, fresh_store):
        # 1 row × 384 floats × 4 bytes = 1536 B < 0.01 MB after rounding.
        # Use 10 rows (15360 B = 0.01 MB rounded) to get a detectable difference.
        before = fresh_store.stats()["memory_mb"]
        n = 10
        chunks = [make_chunk(chunk_index=i) for i in range(n)]
        fresh_store.add(chunks, make_embeddings(n), make_meta(chunk_count=n))
        after = fresh_store.stats()["memory_mb"]
        assert after > before

    def test_stats_returns_all_required_keys(self, fresh_store):
        s = fresh_store.stats()
        assert "total_docs" in s
        assert "total_chunks" in s
        assert "memory_mb" in s
