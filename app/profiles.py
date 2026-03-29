from dataclasses import dataclass
from typing import Literal


EmbeddingProfile = Literal[
    "local_minilm",
    "local_mpnet",
    "openai_small",
    "openai_large",
]

PipelineMode = Literal["baseline_v1", "enhanced_v2"]
RerankerMode = Literal["cross_encoder", "off"]


@dataclass(frozen=True)
class EmbeddingProfileSpec:
    provider: Literal["local", "openrouter"]
    model: str


EMBEDDING_PROFILE_SPECS: dict[EmbeddingProfile, EmbeddingProfileSpec] = {
    "local_minilm": EmbeddingProfileSpec(
        provider="local",
        model="all-MiniLM-L6-v2",
    ),
    "local_mpnet": EmbeddingProfileSpec(
        provider="local",
        model="all-mpnet-base-v2",
    ),
    "openai_small": EmbeddingProfileSpec(
        provider="openrouter",
        model="openai/text-embedding-3-small",
    ),
    "openai_large": EmbeddingProfileSpec(
        provider="openrouter",
        model="openai/text-embedding-3-large",
    ),
}


def get_embedding_profile_spec(profile: EmbeddingProfile) -> EmbeddingProfileSpec:
    spec = EMBEDDING_PROFILE_SPECS.get(profile)
    if spec is None:
        raise ValueError(f"Unknown embedding profile: {profile}")
    return spec
