import os
import pickle
import numpy as np
from threading import Lock

from backend.models.document import Chunk, DocumentMeta
from backend.config import settings


class VectorStore:
    def __init__(self):
        self._lock = Lock()
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self._doc_metas: dict[str, DocumentMeta] = {}
        self._try_load_cache()

    def add(self, chunks: list[Chunk], embeddings: np.ndarray, meta: DocumentMeta):
        """Append chunks, their embeddings, and document metadata to the store.

        Thread-safe.  When ``embeddings`` is non-empty it is vstacked onto the
        existing embedding matrix; when empty the matrix is left unchanged so
        documents with zero chunks (e.g. images without Tesseract) can still be
        registered.

        Args:
            chunks (list[Chunk]): Parsed text chunks for the document.
            embeddings (np.ndarray): Float32 array of shape ``(len(chunks), dim)``
                containing the L2-normalised embedding for each chunk.
                Pass an empty ``(0, dim)`` array when there are no chunks.
            meta (DocumentMeta): Metadata for the document being added.

        Example:
            >>> store = VectorStore()
            >>> store.add(chunks, embeddings, meta)
            >>> store.stats()["total_chunks"]
            5
        """
        with self._lock:
            self._chunks.extend(chunks)
            if len(embeddings) > 0:
                self._embeddings = np.vstack([self._embeddings, embeddings]) if len(self._embeddings) > 0 else embeddings
            self._doc_metas[meta.doc_id] = meta

    def get_all_embeddings(self) -> np.ndarray:
        return self._embeddings

    def get_chunk(self, index: int) -> Chunk:
        return self._chunks[index]

    def get_all_chunks(self) -> list[Chunk]:
        return self._chunks

    def get_all_metas(self) -> list[DocumentMeta]:
        return list(self._doc_metas.values())

    def remove_doc(self, doc_id: str) -> int:
        """Remove all chunks and embeddings for a document from the store.

        Thread-safe.  Rebuilds the chunk list and embedding matrix by keeping
        only rows not belonging to ``doc_id``.  Also removes the document's
        metadata entry.  When ``doc_id`` does not exist, returns 0 without
        raising.

        Args:
            doc_id (str): The UUID of the document to remove, as stored in
                ``DocumentMeta.doc_id``.

        Returns:
            int: Number of chunks that were removed.  Returns 0 if the document
                was not found.

        Example:
            >>> store.add(chunks, embeddings, meta)
            >>> store.remove_doc(meta.doc_id)
            5
            >>> store.stats()["total_chunks"]
            0
        """
        with self._lock:
            if doc_id not in self._doc_metas:
                return 0
            indices_to_keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
            removed = len(self._chunks) - len(indices_to_keep)
            self._chunks = [self._chunks[i] for i in indices_to_keep]
            if len(indices_to_keep) > 0:
                self._embeddings = self._embeddings[indices_to_keep]
            else:
                self._embeddings = np.empty((0, 384), dtype=np.float32)
            del self._doc_metas[doc_id]
            return removed

    def stats(self) -> dict:
        """Return a summary of the store's current memory usage and document counts.

        Returns:
            dict: Dictionary with the following keys:

                * ``"total_docs"`` (int) — number of registered documents.
                * ``"total_chunks"`` (int) — total number of stored chunks.
                * ``"memory_mb"`` (float) — embedding matrix size in MB,
                  rounded to 2 decimal places.

        Example:
            >>> store.stats()
            {'total_docs': 3, 'total_chunks': 47, 'memory_mb': 0.07}
        """
        mem_mb = self._embeddings.nbytes / (1024 * 1024)
        return {
            "total_docs": len(self._doc_metas),
            "total_chunks": len(self._chunks),
            "memory_mb": round(mem_mb, 2),
        }

    def save_cache(self):
        """Persist the current store state to disk as a pickle file.

        Creates the cache directory if it does not exist, then serialises
        chunks, embeddings, and document metadata to ``settings.cache_file``.
        Called automatically after uploads and deletions so the store survives
        process restarts.

        Example:
            >>> store.add(chunks, embeddings, meta)
            >>> store.save_cache()
            # State written to cache/docs.pkl
        """
        os.makedirs(settings.cache_dir, exist_ok=True)
        with open(settings.cache_file, "wb") as f:
            pickle.dump({
                "chunks": self._chunks,
                "embeddings": self._embeddings,
                "doc_metas": self._doc_metas,
            }, f)

    def _try_load_cache(self):
        if os.path.exists(settings.cache_file):
            try:
                with open(settings.cache_file, "rb") as f:
                    data = pickle.load(f)
                self._chunks = data["chunks"]
                self._embeddings = data["embeddings"]
                self._doc_metas = data["doc_metas"]
            except Exception:
                pass


vector_store = VectorStore()
