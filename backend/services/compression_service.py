from openai import OpenAI
from backend.config import settings

_client: OpenAI | None = None

_COMPRESS_PROMPT = """Compress the following document chunk into a dense, information-rich representation.

Rules:
- Preserve ALL: key facts, named entities, dates, numbers, technical terms, relationships, and conclusions.
- Remove: filler phrases, repetition, boilerplate language, and transitional words that add no meaning.
- Do NOT add any information that is not in the original text.
- Output only the compressed text, nothing else.

CHUNK:
{text}

COMPRESSED:"""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    return _client


def compress_chunk(text: str) -> str:
    """Compress a document chunk using the LLM to preserve key information with fewer tokens.

    Chunks shorter than ``settings.compress_min_length`` are returned unchanged —
    they are already concise enough that compression would provide no benefit.
    If the LLM call fails for any reason, the original text is returned so that
    ingestion never fails due to a compression error.

    Args:
        text (str): Raw chunk text extracted from a parsed document.

    Returns:
        str: LLM-compressed text, or the original ``text`` when the chunk is
            below the length threshold or the API call fails.

    Example:
        >>> long = "The quick brown fox jumped over the lazy dog. " * 20
        >>> compressed = compress_chunk(long)
        >>> len(compressed) <= len(long)
        True
    """
    if len(text) < settings.compress_min_length:
        return text

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": _COMPRESS_PROMPT.format(text=text)}],
            temperature=0.1,
            max_tokens=512,
        )
        compressed = (response.choices[0].message.content or "").strip()
        return compressed if compressed else text
    except Exception:
        return text
