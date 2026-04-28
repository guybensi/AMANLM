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
    client = get_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""
