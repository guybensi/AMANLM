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
        mem_mb = self._embeddings.nbytes / (1024 * 1024)
        return {
            "total_docs": len(self._doc_metas),
            "total_chunks": len(self._chunks),
            "memory_mb": round(mem_mb, 2),
        }

    def save_cache(self):
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
