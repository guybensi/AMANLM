import os

# Provide a dummy API key so pydantic_settings doesn't raise on import
os.environ.setdefault("GROQ_API_KEY", "test-key")
