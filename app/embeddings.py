import logging

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.profiles import EmbeddingProfile, get_embedding_profile_spec

logger = logging.getLogger(__name__)


def _resolve_provider_and_model(
    embedding_profile: EmbeddingProfile | None,
) -> tuple[str, str]:
    if embedding_profile is not None:
        spec = get_embedding_profile_spec(embedding_profile)
        return spec.provider, spec.model

    provider = settings.embedding_provider
    model_name = (
        settings.embedding_model_name
        if provider == "local"
        else settings.openrouter_embedding_model
    )
    return provider, model_name


def _resolve_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps"):
        try:
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    if hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                return "xpu"
        except Exception:
            pass
    return "cpu"


def get_embeddings(
    embedding_profile: EmbeddingProfile | None = None,
) -> Embeddings:
    provider, model_name = _resolve_provider_and_model(embedding_profile)

    if provider == "local":
        device = _resolve_torch_device()
        logger.info(
            "Using LOCAL embeddings: model='%s', device=%s",
            model_name,
            device.upper(),
        )
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )

    if provider == "openrouter":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        logger.info(
            "Using OPENROUTER embeddings: model='%s'",
            model_name,
        )
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Must be 'local' or 'openrouter'."
    )
