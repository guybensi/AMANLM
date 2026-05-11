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
        """Encode a list of strings into L2-normalised embedding vectors.

        Lazily loads the sentence-transformer model on first call.  Embeddings
        are L2-normalised so that cosine similarity equals the dot product,
        enabling fast retrieval via ``np.dot``.

        Args:
            texts (list[str]): One or more strings to embed.  Must be non-empty.

        Returns:
            np.ndarray: Float32 array of shape ``(len(texts), embedding_dim)``
                with each row normalised to unit length.

        Example:
            >>> vecs = embedding_service.encode(["hello world", "foo bar"])
            >>> vecs.shape
            (2, 384)
            >>> import numpy as np
            >>> np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
            True
        """
        self._load()
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # L2-normalize so cosine similarity = dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (embeddings / norms).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single string and return its 1-D embedding vector.

        Convenience wrapper around ``encode()`` that avoids the caller having
        to wrap the string in a list and index the result.

        Args:
            text (str): The string to embed.

        Returns:
            np.ndarray: 1-D float32 array of shape ``(embedding_dim,)``,
                L2-normalised.

        Example:
            >>> vec = embedding_service.encode_single("machine learning")
            >>> vec.shape
            (384,)
        """
        return self.encode([text])[0]


embedding_service = EmbeddingService()
