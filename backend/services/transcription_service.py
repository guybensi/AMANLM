import io
from openai import OpenAI
from backend.config import settings

_client: OpenAI | None = None

# Formats the Groq Whisper API accepts natively (no conversion needed)
WHISPER_NATIVE_EXTS: frozenset[str] = frozenset({
    "mp4", "webm", "m4a", "mp3", "wav", "flac", "ogg", "opus", "mpeg", "mpga",
})

_GROQ_MAX_BYTES = 25 * 1024 * 1024  # 25 MB hard limit on Groq Whisper uploads

_EXT_TO_MIME: dict[str, str] = {
    "mp4":  "video/mp4",
    "webm": "video/webm",
    "mov":  "video/quicktime",
    "avi":  "video/x-msvideo",
    "mkv":  "video/x-matroska",
    "m4v":  "video/x-m4v",
    "mp3":  "audio/mpeg",
    "wav":  "audio/wav",
    "m4a":  "audio/mp4",
    "ogg":  "audio/ogg",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    return _client


def transcribe(file_bytes: bytes, filename: str) -> str:
    """Send an audio or video file to the Groq Whisper API and return the transcript.

    Uses ``settings.groq_audio_model`` (default ``whisper-large-v3``) which is
    the highest-quality transcription model available on Groq.  Files larger
    than the 25 MB Groq limit are rejected immediately and return an empty
    string rather than raising an exception.

    Args:
        file_bytes (bytes): Raw audio or video content.
        filename (str): Original filename — used to set the correct MIME type
            in the multipart upload and preserved in the API metadata.

    Returns:
        str: Transcribed text, or an empty string when the file is too large
            or the API returns no content.

    Raises:
        openai.APIError: Propagates API-level errors (auth, rate limit, etc.)
            so callers can implement their own fallback strategy.

    Example:
        >>> with open("lecture.mp4", "rb") as f:
        ...     text = transcribe(f.read(), "lecture.mp4")
        >>> isinstance(text, str)
        True
    """
    if len(file_bytes) > _GROQ_MAX_BYTES:
        return ""

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp4"
    mime = _EXT_TO_MIME.get(ext, "video/mp4")

    client = _get_client()
    response = client.audio.transcriptions.create(
        model=settings.groq_audio_model,
        file=(filename, io.BytesIO(file_bytes), mime),
        response_format="text",
    )
    # response_format="text" returns a plain str
    return str(response).strip() if response else ""
