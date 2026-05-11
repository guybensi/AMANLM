import base64
from openai import OpenAI
from backend.config import settings

_client: OpenAI | None = None

# Maps common image file extensions to MIME types for the vision API payload
_EXT_TO_MIME: dict[str, str] = {
    "jpeg": "image/jpeg",
    "jpg":  "image/jpeg",
    "png":  "image/png",
    "webp": "image/webp",
    "gif":  "image/gif",
    "bmp":  "image/bmp",
    "tiff": "image/tiff",
    "tif":  "image/tiff",
}

_VISION_PROMPT = (
    "Extract and transcribe all text visible in this image. "
    "If there are charts, tables, diagrams, or figures, describe their content in detail."
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    return _client


def describe_image(image_bytes: bytes, ext: str = "jpeg") -> str:
    """Send an image to the Groq vision model and return extracted text or a description.

    Encodes the raw image bytes as a base64 data-URL and submits it to the
    configured vision model (``settings.groq_vision_model``) with a prompt that
    requests full text transcription plus structured descriptions of visual
    elements such as charts, tables, and diagrams.

    Args:
        image_bytes (bytes): Raw image content — any format supported by the
            vision model (JPEG, PNG, WebP, GIF, BMP, TIFF).
        ext (str): File extension (without the leading dot) used to determine
            the correct MIME type for the data-URL.  Defaults to ``"jpeg"``.

    Returns:
        str: The model's response — transcribed text and/or element descriptions.
            Returns an empty string if the model produces no content.

    Raises:
        openai.APIError: Propagates any API-level error (auth failure, rate
            limit, unsupported model, etc.) to allow callers to implement their
            own fallback strategy.

    Example:
        >>> with open("invoice.png", "rb") as f:
        ...     result = describe_image(f.read(), ext="png")
        >>> "Invoice" in result
        True

        >>> describe_image(b"", ext="png")  # empty image → empty response
        ''
    """
    mime = _EXT_TO_MIME.get(ext.lower(), "image/jpeg")
    b64 = base64.b64encode(image_bytes).decode()
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.groq_vision_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": _VISION_PROMPT},
            ],
        }],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""
