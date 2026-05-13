import pytest
from unittest.mock import patch, MagicMock
from backend.services.compression_service import compress_chunk
from backend.config import settings


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


class TestCompressChunk:
    def test_short_text_returned_unchanged_without_api_call(self):
        """Chunks below the length threshold bypass the LLM entirely."""
        with patch.object(settings, "compress_min_length", 1000), \
             patch("backend.services.compression_service._get_client") as mock_client:
            result = compress_chunk("Short text.")
        mock_client.assert_not_called()
        assert result == "Short text."

    def test_long_text_is_sent_to_llm(self):
        """Chunks at or above the threshold are compressed via the LLM."""
        long_text = "word " * 100
        with patch.object(settings, "compress_min_length", 10), \
             patch("backend.services.compression_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = _mock_response("Compressed.")
            result = compress_chunk(long_text)
        mock_client.return_value.chat.completions.create.assert_called_once()
        assert result == "Compressed."

    def test_api_failure_returns_original_text(self):
        """Any API error falls back to the original text — ingestion never fails."""
        long_text = "word " * 100
        with patch.object(settings, "compress_min_length", 10), \
             patch("backend.services.compression_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = Exception("API down")
            result = compress_chunk(long_text)
        assert result == long_text

    def test_empty_llm_response_returns_original_text(self):
        """An empty or whitespace-only response falls back to the original."""
        long_text = "word " * 100
        with patch.object(settings, "compress_min_length", 10), \
             patch("backend.services.compression_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = _mock_response("   ")
            result = compress_chunk(long_text)
        assert result == long_text

    def test_none_llm_response_returns_original_text(self):
        """A None content field in the LLM response falls back to the original."""
        long_text = "word " * 100
        with patch.object(settings, "compress_min_length", 10), \
             patch("backend.services.compression_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = _mock_response(None)
            result = compress_chunk(long_text)
        assert result == long_text

    def test_compressed_text_is_stripped(self):
        """Leading/trailing whitespace in the LLM output is stripped."""
        long_text = "word " * 100
        with patch.object(settings, "compress_min_length", 10), \
             patch("backend.services.compression_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = _mock_response("  Trimmed output.  ")
            result = compress_chunk(long_text)
        assert result == "Trimmed output."

    def test_text_exactly_at_threshold_is_compressed(self):
        """A chunk whose length exactly equals compress_min_length is compressed."""
        text = "x" * 50
        with patch.object(settings, "compress_min_length", 50), \
             patch("backend.services.compression_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = _mock_response("Dense.")
            result = compress_chunk(text)
        assert result == "Dense."

    def test_text_one_below_threshold_is_not_compressed(self):
        """A chunk one character shorter than the threshold bypasses the LLM."""
        text = "x" * 49
        with patch.object(settings, "compress_min_length", 50), \
             patch("backend.services.compression_service._get_client") as mock_client:
            result = compress_chunk(text)
        mock_client.assert_not_called()
        assert result == text

    def test_uses_groq_model_from_settings(self):
        """Compression calls are made with the configured groq_model."""
        long_text = "word " * 100
        with patch.object(settings, "compress_min_length", 10), \
             patch("backend.services.compression_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = _mock_response("ok")
            compress_chunk(long_text)
        call_kwargs = mock_client.return_value.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == settings.groq_model

    def test_low_temperature_used_for_determinism(self):
        """Compression requests use temperature ≤ 0.2 for consistent output."""
        long_text = "word " * 100
        with patch.object(settings, "compress_min_length", 10), \
             patch("backend.services.compression_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = _mock_response("ok")
            compress_chunk(long_text)
        call_kwargs = mock_client.return_value.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] <= 0.2
