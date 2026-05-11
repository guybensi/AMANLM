import re


def recursive_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping word-count chunks that approximate token windows.

    The text is first cleaned, then split into windows of ``chunk_size`` words
    that advance by ``chunk_size - overlap`` words each step, so neighbouring
    chunks share ``overlap`` words of context.

    Args:
        text (str): Raw input text to be chunked.
        chunk_size (int): Maximum number of words per chunk. Defaults to 512.
        overlap (int): Number of words shared between consecutive chunks.
            Must be less than ``chunk_size``. Defaults to 64.

    Returns:
        list[str]: Ordered list of non-empty text chunks.  Returns an empty
            list when the cleaned text contains no words.

    Example:
        >>> chunks = recursive_chunk("one two three four five", chunk_size=3, overlap=1)
        >>> chunks
        ['one two three', 'three four five']

        >>> recursive_chunk("", chunk_size=512, overlap=64)
        []
    """
    text = clean_text(text)
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def clean_text(text: str) -> str:
    """Normalize text by removing non-printable characters and collapsing whitespace.

    Strips characters outside the printable ASCII and extended Unicode ranges,
    collapses runs of three or more newlines to two, and reduces consecutive
    spaces or tabs to a single space.

    Args:
        text (str): Raw input string, e.g. extracted from a PDF page or OCR output.

    Returns:
        str: Cleaned, stripped string.  Returns an empty string when the input
            contains only whitespace or non-printable characters.

    Example:
        >>> clean_text("hello   world\\n\\n\\nfoo")
        'hello world\\n\\nfoo'

        >>> clean_text("  \\t  ")
        ''
    """
    text = re.sub(r"[^\x20-\x7E\n\t\u00A0-\uFFFF]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
