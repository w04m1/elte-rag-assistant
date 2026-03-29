import json
from pathlib import Path

from pydantic import BaseModel

from app.config import settings
from app.rag_chain import DEFAULT_SYSTEM_PROMPT

LOCKED_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT.strip()
ADDITIONAL_SYSTEM_PROMPT_HEADER = "Additional runtime instructions for this deployment:"


class RuntimeSettings(BaseModel):
    generator_model: str
    reranker_model: str
    system_prompt: str
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


def build_default_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        generator_model=(
            settings.openrouter_model
            if settings.llm_provider == "openrouter"
            else settings.ollama_model
        ),
        reranker_model=settings.reranker_model,
        system_prompt="",
        embedding_provider=settings.embedding_provider,
        embedding_model=(
            settings.embedding_model_name
            if settings.embedding_provider == "local"
            else settings.openrouter_embedding_model
        ),
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
        merged = {
            **build_default_runtime_settings().model_dump(),
            **data,
        }
        merged["system_prompt"] = _extract_editable_prompt(
            str(merged.get("system_prompt", ""))
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
    ) -> RuntimeSettings:
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
            }
        )
        self._settings = next_settings
        self._write(next_settings)
        return next_settings
