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

    Constructs a NotebookLM-inspired system prompt that injects retrieved source
    chunks as numbered context blocks.  The prompt instructs the model to:

    * Cite each statement with ``[i]`` notation (or ``[i, j, k]`` for multiple
      sources), using the source index shown in the context blocks.
    * **Bold** the most important parts of the response.
    * Use bullet points when the response is long (unless the query asks for a
      different format).
    * Flag — rather than refuse — when a statement draws on knowledge outside
      the provided sources, so the user can independently verify it.
    * Note when the sources contain no relevant information for the query.
    * Ask for clarification when the query is ambiguous.
    * Never use the word "delve" or "delves".
    * Default to English unless the query requests another language.

    The ``mode`` parameter controls answer length:

    * ``"short"`` — 2-3 concise sentences, direct and to the point.
    * ``"long"`` — comprehensive, well-structured with sections or bullets.

    Args:
        query (str): The user's original question, placed as the final user message.
        scored_chunks (list[ScoredChunk]): Ranked context chunks from ``retrieve()``.
            Each chunk appears as a labelled ``[Source i]`` block in the system prompt.
        mode (str): Answer style — ``"short"`` or ``"long"``.

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

    system_prompt = f"""You are a helpful expert research assistant. Your goal is to provide insightful, well-grounded responses to the user's query by drawing on the sources below.

RESPONSE GUIDELINES:
1. Cite each statement that is supported by a source using [i] notation immediately after the statement, where i is the source number. If a statement draws on multiple sources, list all of them: [i, j, k].
2. **Bold** the most important parts of your response to make it easier to understand.
3. {length_instruction} If your response is getting long and no specific format was requested, use bullet points to improve readability.
4. If the query is ambiguous, ask the user for clarification before answering.
5. If any part of your response includes information from outside the given sources, explicitly note that it is not from the sources and that the user may want to independently verify it.
6. If the sources do not contain any relevant information for the query, state that clearly in your response.
7. Do not use the word "delve" or "delves".
8. Answer in English unless the query requests a response in a different language.

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
