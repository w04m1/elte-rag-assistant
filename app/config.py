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
    index_snapshots_root_path: str = "data/indexes"
    active_index_state_path: str = "data/runtime/active_indexes.json"
    runtime_settings_path: str = "data/runtime/settings.json"
    usage_log_path: str = "data/runtime/usage_log.jsonl"
    documents_sync_state_path: str = "data/runtime/documents_sync_state.json"
    documents_typesense_url: str = "https://typesense.elte.hu/multi_search"
    documents_typesense_api_key: str = ""
    documents_sync_per_page: int = 250
    documents_request_timeout_seconds: float = 30.0

    # Legacy scraper settings (unused, kept for env compatibility)
    scrape_manifest_path: str = "data/runtime/scrape_manifest.json"
    scrape_news_path: str = "data/scraped_news"

    # News sync settings
    news_typesense_url: str = "https://typesense.elte.hu/multi_search"
    news_typesense_api_key: str = ""
    news_records_path: str = "data/news/items"
    news_state_path: str = "data/news/state.json"
    news_faiss_index_path: str = "data/news_vector_store"
    news_bootstrap_pages: int = 4
    news_sync_pages: int = 2
    # Deprecated: periodic scheduler removed, news sync is manual-only.
    news_sync_interval_seconds: int = 21600
    news_request_timeout_seconds: float = 30.0

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
    retrieval_max_chunks_per_doc: int = 3

    # Reranker model
    reranker_model: str = "google/gemini-3-flash-preview"
    reranker_cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_cross_encoder_top_n: int = 30

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

    # Runtime profile defaults
    default_embedding_profile: Literal[
        "local_minilm",
        "local_mpnet",
        "openai_small",
        "openai_large",
    ] = "local_minilm"
    default_pipeline_mode: Literal["baseline_v1", "enhanced_v2"] = "baseline_v1"
    # Kept as plain string for backward compatibility with older env values (e.g. "llm").
    default_reranker_mode: str = "off"
    default_chunk_profile: str = "standard"
    default_parser_profile: str = "docling_v1"


# Singleton
settings = Settings()
