import json
from pathlib import Path

from pydantic import BaseModel

from app.config import settings
from app.profiles import (
    EmbeddingProfile,
    PipelineMode,
    RerankerMode,
    get_embedding_profile_spec,
)
from app.rag_chain import DEFAULT_SYSTEM_PROMPT

LOCKED_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT.strip()
ADDITIONAL_SYSTEM_PROMPT_HEADER = "Additional runtime instructions for this deployment:"


class RuntimeSettings(BaseModel):
    generator_model: str
    reranker_model: str
    system_prompt: str
    embedding_profile: EmbeddingProfile
    pipeline_mode: PipelineMode
    reranker_mode: RerankerMode
    chunk_profile: str
    parser_profile: str
    max_chunks_per_doc: int
    embedding_provider: str
    embedding_model: str


def compose_system_prompt(editable_prompt: str | None) -> str:
    extra = (editable_prompt or "").strip()
    if not extra:
        return LOCKED_SYSTEM_PROMPT
    return (
        f"{LOCKED_SYSTEM_PROMPT}\n\n"
        f"{ADDITIONAL_SYSTEM_PROMPT_HEADER}\n{extra}"
    )


def _extract_editable_prompt(raw_prompt: str | None) -> str:
    prompt = (raw_prompt or "").strip()
    if not prompt or prompt == LOCKED_SYSTEM_PROMPT:
        return ""

    formatted_prefix = (
        f"{LOCKED_SYSTEM_PROMPT}\n\n{ADDITIONAL_SYSTEM_PROMPT_HEADER}\n"
    )
    if prompt.startswith(formatted_prefix):
        return prompt[len(formatted_prefix) :].strip()

    if prompt.startswith(LOCKED_SYSTEM_PROMPT):
        return prompt[len(LOCKED_SYSTEM_PROMPT) :].strip()

    return prompt


def _infer_profile_from_legacy_values(
    provider: str,
    model: str,
) -> EmbeddingProfile:
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip().lower()

    if normalized_provider == "local":
        if normalized_model == "all-minilm-l6-v2":
            return "local_minilm"
        return "local_mpnet"

    if "text-embedding-3-small" in normalized_model:
        return "openai_small"

    return "openai_large"


def _normalize_embedding_fields(payload: dict, *, prefer_legacy: bool = False) -> dict:
    embedding_profile = payload.get("embedding_profile")
    if prefer_legacy or embedding_profile not in {
        "local_minilm",
        "local_mpnet",
        "openai_small",
        "openai_large",
    }:
        embedding_profile = _infer_profile_from_legacy_values(
            str(payload.get("embedding_provider", settings.embedding_provider)),
            str(
                payload.get(
                    "embedding_model",
                    settings.embedding_model_name
                    if settings.embedding_provider == "local"
                    else settings.openrouter_embedding_model,
                )
            ),
        )

    spec = get_embedding_profile_spec(embedding_profile)  # type: ignore[arg-type]
    normalized = dict(payload)
    normalized["embedding_profile"] = embedding_profile
    normalized["embedding_provider"] = spec.provider
    normalized["embedding_model"] = spec.model
    return normalized


def _normalize_reranker_mode(value: str | None) -> RerankerMode:
    normalized = (value or "").strip().lower()
    if normalized == "cross_encoder":
        return "cross_encoder"
    # Legacy "llm" mode is retired and now maps to "off".
    return "off"


def build_default_runtime_settings() -> RuntimeSettings:
    default_profile = getattr(settings, "default_embedding_profile", "local_minilm")
    if default_profile not in {"local_minilm", "local_mpnet", "openai_small", "openai_large"}:
        default_profile = _infer_profile_from_legacy_values(
            str(getattr(settings, "embedding_provider", "local")),
            str(
                getattr(settings, "embedding_model_name", "all-mpnet-base-v2")
                if str(getattr(settings, "embedding_provider", "local")).lower() == "local"
                else getattr(
                    settings,
                    "openrouter_embedding_model",
                    "openai/text-embedding-3-large",
                )
            ),
        )
    default_spec = get_embedding_profile_spec(default_profile)
    default_pipeline_mode = getattr(settings, "default_pipeline_mode", "baseline_v1")
    if default_pipeline_mode not in {"baseline_v1", "enhanced_v2"}:
        default_pipeline_mode = "baseline_v1"
    default_reranker_mode = _normalize_reranker_mode(
        str(getattr(settings, "default_reranker_mode", "off"))
    )

    return RuntimeSettings(
        generator_model=(
            settings.openrouter_model
            if settings.llm_provider == "openrouter"
            else settings.ollama_model
        ),
        reranker_model=settings.reranker_model,
        system_prompt="",
        embedding_profile=default_profile,
        pipeline_mode=default_pipeline_mode,
        reranker_mode=default_reranker_mode,
        chunk_profile=getattr(settings, "default_chunk_profile", "standard"),
        parser_profile=getattr(settings, "default_parser_profile", "docling_v1"),
        max_chunks_per_doc=max(
            1, int(getattr(settings, "retrieval_max_chunks_per_doc", 3))
        ),
        embedding_provider=default_spec.provider,
        embedding_model=default_spec.model,
    )


class RuntimeSettingsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._settings = self._load()

    def _load(self) -> RuntimeSettings:
        if not self.path.exists():
            default_settings = build_default_runtime_settings()
            self._write(default_settings)
            return default_settings
        data = json.loads(self.path.read_text(encoding="utf-8"))
        raw_has_embedding_profile = isinstance(data, dict) and "embedding_profile" in data
        merged_payload = {
            **build_default_runtime_settings().model_dump(),
            **data,
        }
        merged_payload["system_prompt"] = _extract_editable_prompt(
            str(merged_payload.get("system_prompt", ""))
        )
        merged_payload["reranker_mode"] = _normalize_reranker_mode(
            str(merged_payload.get("reranker_mode", "off"))
        )
        merged = _normalize_embedding_fields(
            merged_payload,
            prefer_legacy=not raw_has_embedding_profile,
        )
        return RuntimeSettings.model_validate(merged)

    def _write(self, runtime_settings: RuntimeSettings) -> None:
        self.path.write_text(
            json.dumps(runtime_settings.model_dump(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def get(self) -> RuntimeSettings:
        return self._settings

    def update(
        self,
        *,
        generator_model: str | None = None,
        reranker_model: str | None = None,
        system_prompt: str | None = None,
        embedding_profile: EmbeddingProfile | None = None,
        pipeline_mode: PipelineMode | None = None,
        reranker_mode: RerankerMode | None = None,
        chunk_profile: str | None = None,
        parser_profile: str | None = None,
        max_chunks_per_doc: int | None = None,
    ) -> RuntimeSettings:
        next_profile = self._settings.embedding_profile if embedding_profile is None else embedding_profile
        profile_spec = get_embedding_profile_spec(next_profile)
        next_settings = self._settings.model_copy(
            update={
                "generator_model": (
                    self._settings.generator_model
                    if generator_model is None
                    else generator_model
                ),
                "reranker_model": (
                    self._settings.reranker_model
                    if reranker_model is None
                    else reranker_model
                ),
                "system_prompt": (
                    self._settings.system_prompt
                    if system_prompt is None
                    else system_prompt.strip()
                ),
                "embedding_profile": next_profile,
                "pipeline_mode": (
                    self._settings.pipeline_mode if pipeline_mode is None else pipeline_mode
                ),
                "reranker_mode": (
                    self._settings.reranker_mode if reranker_mode is None else reranker_mode
                ),
                "chunk_profile": (
                    self._settings.chunk_profile if chunk_profile is None else chunk_profile.strip()
                ),
                "parser_profile": (
                    self._settings.parser_profile if parser_profile is None else parser_profile.strip()
                ),
                "max_chunks_per_doc": (
                    self._settings.max_chunks_per_doc
                    if max_chunks_per_doc is None
                    else max(1, int(max_chunks_per_doc))
                ),
                "embedding_provider": profile_spec.provider,
                "embedding_model": profile_spec.model,
            }
        )
        self._settings = next_settings
        self._write(next_settings)
        return next_settings
