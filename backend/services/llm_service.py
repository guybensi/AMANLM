from openai import OpenAI
from backend.config import settings
from backend.services.rag_service import ScoredChunk

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    return _client


def build_prompt(query: str, scored_chunks: list[ScoredChunk], mode: str) -> list[dict]:
    """Assemble the chat message list to send to the LLM.

    Constructs a system prompt that injects the retrieved source chunks as
    numbered context blocks and enforces strict source-grounded answering.
    The ``mode`` parameter controls answer length and style.

    Args:
        query (str): The user's original question, placed as the final user message.
        scored_chunks (list[ScoredChunk]): Ranked context chunks from ``retrieve()``.
            Each chunk appears as a labelled block in the system prompt.
        mode (str): Answer style — ``"short"`` for 2-3 sentences, ``"long"`` for a
            detailed structured response.

    Returns:
        list[dict]: Two-element list of ``{"role": ..., "content": ...}`` dicts
            suitable for the OpenAI-compatible chat completions API:
            ``[system_message, user_message]``.

    Example:
        >>> chunks = retrieve("What causes inflation?", top_k=5)
        >>> messages = build_prompt("What causes inflation?", chunks, mode="short")
        >>> messages[0]["role"]
        'system'
        >>> messages[1]["content"]
        'What causes inflation?'
    """
    length_instruction = (
        "Answer in 2-3 concise sentences. Be direct and to the point."
        if mode == "short"
        else "Provide a comprehensive, well-structured answer. Use bullet points or sections if helpful. Include relevant details and examples from the sources."
    )

    context_blocks = []
    for i, sc in enumerate(scored_chunks, 1):
        context_blocks.append(
            f"[Source {i}: {sc.chunk.filename}, Page {sc.chunk.page_number}]\n{sc.chunk.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = f"""You are a research assistant. Your ONLY job is to answer questions based strictly on the provided source documents.

RULES:
1. Answer ONLY using information from the sources below. Do NOT use outside knowledge.
2. If the answer cannot be found in the sources, say: "I could not find a clear answer in the uploaded documents."
3. {length_instruction}
4. Do NOT fabricate citations or page numbers. The system will attach citations automatically.
5. Speak in first person as if you read the documents yourself.

SOURCES:
{context}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]


def chat_completion(messages: list[dict]) -> str:
    """Send a message list to the Groq LLM and return the assistant's reply.

    Calls the OpenAI-compatible chat completions endpoint configured in
    ``settings``.  Uses a low temperature (0.2) to favour factual, consistent
    responses suitable for document Q&A.

    Args:
        messages (list[dict]): Conversation history in OpenAI message format,
            e.g. ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``.

    Returns:
        str: The assistant's response text.  Returns an empty string if the
            model returns no content.

    Example:
        >>> msgs = build_prompt("Summarise chapter 3", chunks, mode="long")
        >>> answer = chat_completion(msgs)
        >>> isinstance(answer, str)
        True
    """
    client = get_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""
