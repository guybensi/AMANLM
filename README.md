# AMANLM — AI Research Assistant

A local NotebookLM-like platform. Upload your documents and chat with them — every answer is grounded in your sources with verbatim quotes, page numbers, and a confidence score.

---

## Features

- **RAG-powered chat** — answers come only from your uploaded documents, never from outside knowledge
- **Source proof** — every answer shows the exact quote, filename, and page number it came from
- **Confidence score** — a 0–100% bar shows how well the answer is grounded in your sources
- **Short / Long answer toggle** — switch between concise and detailed responses
- **Multi-format uploads** — PDF, TXT, DOCX, Markdown, and images (with Tesseract OCR)
- **Persistent cache** — documents survive restarts via local pickle cache
- **Dark mode** — on by default

---

## Requirements

- Python 3.10+
- Node.js 18+
- A [Groq](https://console.groq.com) API key (free)
- *(Optional)* [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) for image OCR

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/guybensi/AMANLM.git
cd AMANLM
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Install Python dependencies

```bash
# Install CPU-only PyTorch first (saves ~1.5 GB vs the full CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest
pip install -r requirements.txt
```

### 4. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 5. Run

```bash
python app.py
```

Open your browser at **http://localhost:8000**

---

## Usage

1. **Upload documents** — drag and drop files into the left sidebar (PDF, TXT, DOCX, images)
2. **Ask questions** — type in the chat box and press Enter
3. **Read the answer** — each response includes a confidence bar and expandable source cards with verbatim quotes
4. **Toggle answer length** — use the ⚡ Short / 📝 Long buttons above the input box

---

## Project Structure

```
AMANLM/
├── app.py                    # Entry point — run this to start the server
├── requirements.txt
├── .env                      # Your API key (not committed)
├── backend/
│   ├── config.py             # Settings loaded from .env
│   ├── main.py               # FastAPI app
│   ├── routers/
│   │   ├── documents.py      # Upload / list / delete endpoints
│   │   └── chat.py           # RAG chat endpoint
│   ├── services/
│   │   ├── document_processor.py  # PDF, TXT, DOCX, image parsing
│   │   ├── embedding_service.py   # sentence-transformers (all-MiniLM-L6-v2)
│   │   ├── vector_store.py        # In-memory numpy store + pickle cache
│   │   ├── rag_service.py         # Cosine similarity retrieval + confidence scoring
│   │   └── llm_service.py         # Groq API calls + prompt builder
│   └── models/               # Pydantic request/response schemas
├── frontend/                 # React + Vite source
│   └── src/
│       ├── components/       # Chat, Upload, Sources, Layout components
│       ├── hooks/            # useChat, useDocuments
│       └── stores/           # Zustand global state
├── static/                   # Built frontend (served by FastAPI)
└── cache/                    # Auto-created — persists uploaded docs across restarts
```

---

## Available Groq Models (free)

| Model | Speed | Quality |
|---|---|---|
| `llama-3.3-70b-versatile` | Fast | Best — recommended |
| `llama-3.1-8b-instant` | Very fast | Good for quick answers |
| `mixtral-8x7b-32768` | Fast | Long context |

Change the model any time in `.env` without restarting — takes effect on the next chat message.

---

## Notes

- The embedding model (`all-MiniLM-L6-v2`, ~80 MB) downloads automatically on first run
- Image OCR requires the Tesseract binary: [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki)
- Documents are stored in memory and cached to `cache/docs.pkl` — delete this file to reset
