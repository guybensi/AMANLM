import pytest
from unittest.mock import patch, MagicMock
from backend.services.transcription_service import transcribe, WHISPER_NATIVE_EXTS, _GROQ_MAX_BYTES
from backend.services.document_processor import process_video, dispatch
from backend.config import settings


def _mock_transcription(text: str) -> MagicMock:
    return text  # response_format="text" returns a plain str


class TestTranscribe:
    def test_returns_transcript_string(self):
        with patch("backend.services.transcription_service._get_client") as mock_client:
            mock_client.return_value.audio.transcriptions.create.return_value = "Hello world."
            result = transcribe(b"audio", "lecture.mp4")
        assert result == "Hello world."

    def test_file_over_25mb_returns_empty_without_api_call(self):
        large = b"x" * (_GROQ_MAX_BYTES + 1)
        with patch("backend.services.transcription_service._get_client") as mock_client:
            result = transcribe(large, "big.mp4")
        mock_client.assert_not_called()
        assert result == ""

    def test_file_exactly_at_limit_is_rejected(self):
        at_limit = b"x" * (_GROQ_MAX_BYTES + 1)
        with patch("backend.services.transcription_service._get_client") as mock_client:
            result = transcribe(at_limit, "edge.mp4")
        mock_client.assert_not_called()
        assert result == ""

    def test_file_one_byte_below_limit_is_accepted(self):
        just_under = b"x" * _GROQ_MAX_BYTES
        with patch("backend.services.transcription_service._get_client") as mock_client:
            mock_client.return_value.audio.transcriptions.create.return_value = "ok"
            result = transcribe(just_under, "ok.mp4")
        mock_client.return_value.audio.transcriptions.create.assert_called_once()
        assert result == "ok"

    def test_uses_configured_audio_model(self):
        with patch("backend.services.transcription_service._get_client") as mock_client:
            mock_client.return_value.audio.transcriptions.create.return_value = "text"
            transcribe(b"audio", "file.mp3")
        call_kwargs = mock_client.return_value.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == settings.groq_audio_model

    def test_response_format_is_text(self):
        with patch("backend.services.transcription_service._get_client") as mock_client:
            mock_client.return_value.audio.transcriptions.create.return_value = "text"
            transcribe(b"audio", "file.mp4")
        call_kwargs = mock_client.return_value.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["response_format"] == "text"

    def test_result_is_stripped(self):
        with patch("backend.services.transcription_service._get_client") as mock_client:
            mock_client.return_value.audio.transcriptions.create.return_value = "  Hello.  "
            result = transcribe(b"audio", "file.mp4")
        assert result == "Hello."

    def test_none_response_returns_empty(self):
        with patch("backend.services.transcription_service._get_client") as mock_client:
            mock_client.return_value.audio.transcriptions.create.return_value = None
            result = transcribe(b"audio", "file.mp4")
        assert result == ""

    def test_mp4_uses_video_mime(self):
        with patch("backend.services.transcription_service._get_client") as mock_client:
            mock_client.return_value.audio.transcriptions.create.return_value = "text"
            transcribe(b"video", "lecture.mp4")
        file_arg = mock_client.return_value.audio.transcriptions.create.call_args.kwargs["file"]
        assert file_arg[2] == "video/mp4"

    def test_mp3_uses_audio_mime(self):
        with patch("backend.services.transcription_service._get_client") as mock_client:
            mock_client.return_value.audio.transcriptions.create.return_value = "text"
            transcribe(b"audio", "podcast.mp3")
        file_arg = mock_client.return_value.audio.transcriptions.create.call_args.kwargs["file"]
        assert file_arg[2] == "audio/mpeg"


class TestWhisperNativeExts:
    def test_mp4_is_native(self):
        assert "mp4" in WHISPER_NATIVE_EXTS

    def test_webm_is_native(self):
        assert "webm" in WHISPER_NATIVE_EXTS

    def test_mp3_is_native(self):
        assert "mp3" in WHISPER_NATIVE_EXTS

    def test_mov_is_not_native(self):
        assert "mov" not in WHISPER_NATIVE_EXTS

    def test_avi_is_not_native(self):
        assert "avi" not in WHISPER_NATIVE_EXTS

    def test_mkv_is_not_native(self):
        assert "mkv" not in WHISPER_NATIVE_EXTS


class TestProcessVideo:
    def test_native_format_produces_chunks(self):
        with patch("backend.services.transcription_service.transcribe", return_value="Lecture content here."):
            chunks, size = process_video(b"video", "lecture.mp4", "doc1")
        assert len(chunks) > 0
        assert "Lecture content here." in chunks[0].text

    def test_chunk_metadata_correct(self):
        with patch("backend.services.transcription_service.transcribe", return_value="Text."):
            chunks, _ = process_video(b"video", "talk.mp4", "docX")
        assert chunks[0].filename == "talk.mp4"
        assert chunks[0].page_number == 1
        assert chunks[0].doc_id == "docX"
        assert chunks[0].chunk_index == 0

    def test_empty_transcript_returns_no_chunks(self):
        with patch("backend.services.transcription_service.transcribe", return_value=""):
            chunks, _ = process_video(b"video", "silent.mp4", "doc2")
        assert chunks == []

    def test_whitespace_only_transcript_returns_no_chunks(self):
        with patch("backend.services.transcription_service.transcribe", return_value="   "):
            chunks, _ = process_video(b"video", "silent.webm", "doc3")
        assert chunks == []

    def test_api_failure_returns_no_chunks(self):
        with patch("backend.services.transcription_service.transcribe", side_effect=Exception("API down")):
            chunks, _ = process_video(b"video", "broken.mp4", "doc4")
        assert chunks == []

    def test_size_bytes_matches_input(self):
        content = b"fake-video-bytes"
        with patch("backend.services.transcription_service.transcribe", return_value="Text."):
            _, size = process_video(content, "v.mp4", "doc5")
        assert size == len(content)

    def test_chunk_indices_sequential_for_long_transcript(self):
        long_text = " ".join([f"word{i}" for i in range(600)])
        with patch("backend.services.transcription_service.transcribe", return_value=long_text):
            chunks, _ = process_video(b"video", "long.mp4", "doc6")
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_non_native_without_moviepy_returns_no_chunks(self):
        with patch("backend.services.transcription_service.transcribe", return_value="text"), \
             patch.dict("sys.modules", {"moviepy": None, "moviepy.editor": None}):
            chunks, _ = process_video(b"video", "clip.avi", "doc7")
        assert chunks == []

    def test_dispatch_routes_mp4_to_video_processor(self):
        with patch("backend.services.document_processor.process_video", return_value=([], 100)) as mock_vid:
            dispatch(b"mp4-bytes", "lecture.mp4", "video/mp4")
        mock_vid.assert_called_once()

    def test_dispatch_routes_webm_by_extension(self):
        with patch("backend.services.document_processor.process_video", return_value=([], 50)) as mock_vid:
            dispatch(b"webm-bytes", "clip.webm", "application/octet-stream")
        mock_vid.assert_called_once()

    def test_dispatch_routes_mp3_by_content_type(self):
        with patch("backend.services.document_processor.process_video", return_value=([], 50)) as mock_vid:
            dispatch(b"mp3-bytes", "podcast.mp3", "audio/mpeg")
        mock_vid.assert_called_once()
