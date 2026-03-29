import hashlib
import json
from pathlib import Path
from typing import Any

from app.profiles import EmbeddingProfile

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx"}


def _iter_source_files(source_dir: str | Path) -> list[Path]:
    root = Path(source_dir)
    files: list[Path] = []
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def compute_corpus_hash(source_dir: str | Path) -> str:
    hasher = hashlib.sha256()
    for path in _iter_source_files(source_dir):
        stat = path.stat()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
    return hasher.hexdigest()[:16]


def build_snapshot_id(
    *,
    corpus_hash: str,
    embedding_profile: EmbeddingProfile,
    chunk_profile: str,
    parser_profile: str,
) -> str:
    return (
        f"{corpus_hash}__{embedding_profile}__"
        f"{chunk_profile.strip().lower()}__{parser_profile.strip().lower()}"
    )


def snapshot_dir(root_dir: str | Path, snapshot_id: str) -> Path:
    return Path(root_dir) / snapshot_id


def manifest_path(root_dir: str | Path, snapshot_id: str) -> Path:
    return snapshot_dir(root_dir, snapshot_id) / "manifest.json"


def write_snapshot_manifest(
    *,
    root_dir: str | Path,
    snapshot_id: str,
    payload: dict[str, Any],
) -> Path:
    target_dir = snapshot_dir(root_dir, snapshot_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(root_dir, snapshot_id)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def read_snapshot_manifest(
    *,
    root_dir: str | Path,
    snapshot_id: str,
) -> dict[str, Any] | None:
    path = manifest_path(root_dir, snapshot_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_active_map(state_path: str | Path) -> dict[str, str]:
    path = Path(state_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    snapshots = payload.get("active_snapshots")
    if not isinstance(snapshots, dict):
        return {}
    return {str(k): str(v) for k, v in snapshots.items() if str(v).strip()}


def _save_active_map(state_path: str | Path, active_map: dict[str, str]) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active_snapshots": active_map,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def get_active_snapshot_id(
    *,
    state_path: str | Path,
    embedding_profile: EmbeddingProfile,
) -> str | None:
    active_map = _load_active_map(state_path)
    snapshot_id = active_map.get(embedding_profile)
    return snapshot_id if snapshot_id else None


def set_active_snapshot_id(
    *,
    state_path: str | Path,
    embedding_profile: EmbeddingProfile,
    snapshot_id: str,
) -> None:
    active_map = _load_active_map(state_path)
    active_map[embedding_profile] = snapshot_id
    _save_active_map(state_path, active_map)


def resolve_active_index_path(
    *,
    root_dir: str | Path,
    state_path: str | Path,
    embedding_profile: EmbeddingProfile,
) -> Path | None:
    snapshot_id = get_active_snapshot_id(
        state_path=state_path,
        embedding_profile=embedding_profile,
    )
    if not snapshot_id:
        return None
    candidate = snapshot_dir(root_dir, snapshot_id)
    if not candidate.exists():
        return None
    return candidate

