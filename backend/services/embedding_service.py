import numpy as np
from sentence_transformers import SentenceTransformer
from backend.config import settings


class EmbeddingService:
    def __init__(self):
        self._model: SentenceTransformer | None = None

    def _load(self):
        if self._model is None:
            self._model = SentenceTransformer(settings.embedding_model)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # L2-normalize so cosine similarity = dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (embeddings / norms).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


embedding_service = EmbeddingService()
