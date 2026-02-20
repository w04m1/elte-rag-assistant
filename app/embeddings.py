import logging

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


def get_embeddings() -> Embeddings:
    provider = settings.embedding_provider

    if provider == "local":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Using LOCAL embeddings: model='{settings.embedding_model_name}', device={device.upper()}",
        )
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": device},
        )

    if provider == "openrouter":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        logger.info(
            f"Using OPENROUTER embeddings: model='{settings.openrouter_embedding_model}'",
        )
        return OpenAIEmbeddings(
            model=settings.openrouter_embedding_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Must be 'local' or 'openrouter'."
    )
