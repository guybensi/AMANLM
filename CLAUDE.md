# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

AMANLM is a local NotebookLM-style RAG application. Users upload documents (PDF, DOCX, TXT, images, video, audio), which are parsed, embedded, and stored. They can then chat with their documents — every answer cites the source chunks it drew from.

## Commands

```bash
# Run the server (dev mode with hot reload)
python app.py

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_llm_service.py -v

# Run a single test by name
python -m pytest tests/test_llm_service.py::TestBuildPromptHistory::test_single_turn_history -v

# Build the frontend
cd frontend && npm install && npm run build && cd ..

# Start via Docker
docker compose up --build

# Stop Docker
docker compose down
```

## Architecture

### Ingestion flow

`POST /api/documents/upload` → `routers/documents.py`

1. `document_processor.dispatch()` detects file type (MIME → extension fallback) and routes to the matching parser:
   - **PDF** — `process_pdf()`: PyMuPDF text extraction + Groq Vision for embedded images
   - **TXT/MD** — `process_txt()`: UTF-8 decode
   - **DOCX** — `process_docx()`: paragraphs with heading markers + pipe-delimited table rows
   - **Image** — `process_image()`: Groq Vision primary, Tesseract OCR fallback
   - **Video/Audio** — `process_video()`: Groq Whisper direct upload for native formats (mp4, webm, mp3, wav…); moviepy audio extraction for non-native formats (mov, avi, mkv)

2. If `COMPRESS_CHUNKS=true`, each chunk passes through `compression_service.compress_chunk()` (Groq LLM call) before embedding.

3. `embedding_service.encode()` produces L2-normalised float32 vectors via `paraphrase-multilingual-MiniLM-L12-v2`.

4. `vector_store.add()` appends chunks + embeddings in memory, then `save_cache()` persists to `cache/docs.pkl`.

### Query flow

`POST /api/chat` → `routers/chat.py`

1. `rag_service.retrieve()` embeds the query and runs dot-product similarity against all stored embeddings (already L2-normalised, so dot product = cosine similarity).
2. `calculate_confidence()` combines top-score, coverage, and consistency into a 0–100 score.
3. `llm_service.build_prompt()` assembles a NotebookLM-inspired system prompt with numbered `[Source i]` blocks, then appends prior conversation turns and the current query.
4. `llm_service.chat_completion()` calls Groq via the OpenAI-compatible SDK.
5. The response includes `sources` (list of `SourceProof`) which the frontend renders as expandable cards. Citation markers `[1]`, `[2, 3]` in the answer text are converted to clickable badges by `MessageBubble.jsx` using a `wrapCitations()` pre-processor and a custom ReactMarkdown `code` component.

### Key singletons

- `vector_store` (module-level in `vector_store.py`) — the single in-memory store; shared across all requests
- `embedding_service` (module-level in `embedding_service.py`) — loads the sentence-transformer model once on startup
- Groq API clients in each service (`_client` globals) — lazily initialised on first call

### Settings

All settings live in `backend/config.py` as a `pydantic-settings` `BaseSettings` class, loaded from `.env`. The `settings` singleton is imported directly by services. Key fields:

| Setting | Purpose |
|---|---|
| `groq_model` | Chat/RAG LLM (`llama-3.3-70b-versatile`) |
| `groq_vision_model` | Vision model for images/PDF (`llama-4-scout`) |
| `groq_audio_model` | Whisper model for video/audio (`whisper-large-v3`) |
| `compress_chunks` | Enable LLM compression at ingestion (default: false) |
| `compress_min_length` | Min chars to trigger compression (default: 200) |
| `chunk_size` / `chunk_overlap` | Text chunking parameters (512 / 64) |

### Frontend

React + Vite app in `frontend/src/`. Built output is served as static files by FastAPI from `static/`. The frontend is **not** rebuilt automatically — run `npm run build` after frontend changes.

Key frontend pieces:
- `useChat.js` — sends `{ message, mode, top_k, history }` to `/api/chat`; captures `chatHistory` before adding the new message so only prior turns are sent as history
- `MessageBubble.jsx` — pre-processes answer text with `wrapCitations()` to turn `[1]` into `` `[[1]]` ``, then intercepts that sentinel in a custom ReactMarkdown `code` renderer to produce clickable citation badges
- `SourceCard.jsx` — accepts `highlighted` and `cardRef` props; auto-expands when `highlighted` becomes true

### Testing

Tests are in `tests/` and use `pytest` with `unittest.mock`. All external API calls (Groq LLM, Vision, Whisper) are mocked — no real API key is needed to run tests. The `conftest.py` sets up any shared fixtures.

To reset document state between test runs, delete `cache/docs.pkl`.
