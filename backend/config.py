import shutil
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")
    groq_vision_model: str = Field("meta-llama/llama-4-scout-17b-16e-instruct", env="GROQ_VISION_MODEL")
    groq_base_url: str = "https://api.groq.com/openai/v1"

    cache_dir: str = "cache"
    cache_file: str = "cache/docs.pkl"

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

    compress_chunks: bool = Field(False, env="COMPRESS_CHUNKS")
    compress_min_length: int = Field(200, env="COMPRESS_MIN_LENGTH")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


def check_tesseract() -> bool:
    return shutil.which("tesseract") is not None


TESSERACT_AVAILABLE = check_tesseract()
