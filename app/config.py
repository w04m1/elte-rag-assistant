from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embedding provider
    embedding_provider: Literal["local", "openrouter"] = "local"

    # Local embedding model (hf sentence-transformers)
    embedding_model_name: str = "all-mpnet-base-v2"

    # OpenRouter embedding model
    openrouter_embedding_model: str = "openai/text-embedding-3-large"

    # FAISS vector store path
    faiss_index_path: str = "data/vector_store"
    runtime_settings_path: str = "data/runtime/settings.json"
    scrape_manifest_path: str = "data/runtime/scrape_manifest.json"

    # Ingestion/chunking params
    max_tokens: int = 256
    raw_data_path: str = "data/raw"
    scrape_download_path: str = "data/raw"

    # Retrieval params
    retrieval_k: int = 5
    retrieval_fetch_k: int = 30
    retrieval_use_reranker: bool = True
    retrieval_hybrid: bool = True
    retrieval_hybrid_weight: float = 0.6

    # Reranker model
    reranker_model: str = "google/gemini-3-flash-preview"

    # LLM provider
    llm_provider: Literal["openrouter", "ollama"] = "openrouter"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-3-flash-preview"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_allow_origins: str = "*"


# Singleton
settings = Settings()
