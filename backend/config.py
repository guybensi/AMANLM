import shutil
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_audio_model: str = "whisper-large-v3"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    cache_dir: str = "cache"
    cache_file: str = "cache/docs.pkl"

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

    compress_chunks: bool = False
    compress_min_length: int = 200


settings = Settings()


def check_tesseract() -> bool:
    return shutil.which("tesseract") is not None


TESSERACT_AVAILABLE = check_tesseract()
