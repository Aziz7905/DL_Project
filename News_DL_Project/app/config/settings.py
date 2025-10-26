# app/config/settings.py
from pathlib import Path
from typing import Dict
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    # Pydantic v2 style config
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    # Paths
    DATA_DIR: str = str(ROOT / "data")
    INDEX_PATH: str = str(ROOT / "vectorstore")
    BM25_PATH: str = str(ROOT / "bm25.pkl")
    RL_LOG_PATH: str = str(ROOT / "rl_logs.jsonl")

    # API keys (cloud-only)
    GROQ_API_KEY: str = ""                 # Groq
    HUGGINGFACEHUB_API_TOKEN: str = ""     # Hugging Face Inference API
    SERPAPI_API_KEY: str = ""              # optional

    # LLMs
    GROQ_MODEL_ANSWER: str = "llama-3.3-70b-versatile"
    HF_MODEL_EXTRACT: str = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_MODEL_VERIFY:  str = "mistralai/Mistral-7B-Instruct-v0.3"


    # Embeddings (LOCAL)
    HF_MODEL_EMBED: str = "BAAI/bge-large-en-v1.5"

    # RAG chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # Scoring weights
    AGGREGATION_WEIGHTS: Dict[str, float] = {
        "evidence_support": 0.55,
        "source_credibility": 0.30,
        "cross_verification": 0.15,
    }

Config = Settings()
