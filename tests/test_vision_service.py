import pytest
from unittest.mock import patch, MagicMock
from backend.services.vision_service import describe_image, _EXT_TO_MIME


class TestDescribeImage:
    def _mock_response(self, content: str) -> MagicMock:
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_returns_model_content(self):
        with patch("backend.services.vision_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                self._mock_response("Invoice total: $1,200")
            )
            result = describe_image(b"fake-image-bytes", ext="png")
        assert result == "Invoice total: $1,200"

    def test_returns_empty_string_when_model_returns_none(self):
        with patch("backend.services.vision_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                self._mock_response(None)
            )
            result = describe_image(b"fake-image-bytes", ext="jpeg")
        assert result == ""

    def test_uses_correct_vision_model(self):
        with patch("backend.services.vision_service._get_client") as mock_client, \
             patch("backend.services.vision_service.settings") as mock_settings:
            mock_settings.groq_vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
            mock_settings.groq_api_key = "test"
            mock_settings.groq_base_url = "https://api.groq.com/openai/v1"
            mock_client.return_value.chat.completions.create.return_value = (
                self._mock_response("text")
            )
            describe_image(b"bytes", ext="jpeg")
            call_kwargs = mock_client.return_value.chat.completions.create.call_args
            assert call_kwargs.kwargs["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"

    def test_uses_correct_mime_for_png(self):
        with patch("backend.services.vision_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                self._mock_response("text")
            )
            describe_image(b"bytes", ext="png")
            messages = mock_client.return_value.chat.completions.create.call_args.kwargs["messages"]
            image_url = messages[0]["content"][0]["image_url"]["url"]
            assert image_url.startswith("data:image/png;base64,")

    def test_uses_correct_mime_for_jpeg(self):
        with patch("backend.services.vision_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                self._mock_response("text")
            )
            describe_image(b"bytes", ext="jpg")
            messages = mock_client.return_value.chat.completions.create.call_args.kwargs["messages"]
            image_url = messages[0]["content"][0]["image_url"]["url"]
            assert image_url.startswith("data:image/jpeg;base64,")

    def test_defaults_to_jpeg_mime_for_unknown_ext(self):
        with patch("backend.services.vision_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                self._mock_response("text")
            )
            describe_image(b"bytes", ext="xyz")
            messages = mock_client.return_value.chat.completions.create.call_args.kwargs["messages"]
            image_url = messages[0]["content"][0]["image_url"]["url"]
            assert image_url.startswith("data:image/jpeg;base64,")

    def test_image_bytes_are_base64_encoded(self):
        import base64
        raw = b"hello image data"
        with patch("backend.services.vision_service._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                self._mock_response("text")
            )
            describe_image(raw, ext="png")
            messages = mock_client.return_value.chat.completions.create.call_args.kwargs["messages"]
            image_url = messages[0]["content"][0]["image_url"]["url"]
            encoded_part = image_url.split("base64,")[1]
            assert base64.b64decode(encoded_part) == raw

    def test_ext_to_mime_table_covers_common_formats(self):
        assert _EXT_TO_MIME["jpeg"] == "image/jpeg"
        assert _EXT_TO_MIME["png"]  == "image/png"
        assert _EXT_TO_MIME["webp"] == "image/webp"
        assert _EXT_TO_MIME["gif"]  == "image/gif"
