import json

from app.runtime_settings import RuntimeSettingsStore


def test_runtime_settings_migrates_legacy_embedding_fields(tmp_path):
    path = tmp_path / "runtime-settings.json"
    path.write_text(
        json.dumps(
            {
                "generator_model": "g",
                "reranker_model": "r",
                "system_prompt": "",
                "embedding_provider": "local",
                "embedding_model": "all-MiniLM-L6-v2",
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    store = RuntimeSettingsStore(path)
    settings = store.get()
    assert settings.embedding_profile == "local_minilm"
    assert settings.embedding_provider == "local"
    assert settings.embedding_model == "all-MiniLM-L6-v2"


def test_runtime_settings_updates_profile_and_model_consistently(tmp_path):
    store = RuntimeSettingsStore(tmp_path / "runtime-settings.json")
    updated = store.update(embedding_profile="openai_small")
    assert updated.embedding_profile == "openai_small"
    assert updated.embedding_provider == "openrouter"
    assert updated.embedding_model == "openai/text-embedding-3-small"
