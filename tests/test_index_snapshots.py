from pathlib import Path

from app.index_snapshots import (
    build_snapshot_id,
    compute_corpus_hash,
    get_active_snapshot_id,
    resolve_active_index_path,
    set_active_snapshot_id,
    snapshot_dir,
)


def test_compute_corpus_hash_changes_with_file_metadata(tmp_path):
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    file_path = source_dir / "demo.pdf"
    file_path.write_bytes(b"%PDF-1.4 demo")

    first_hash = compute_corpus_hash(source_dir)
    file_path.write_bytes(b"%PDF-1.4 changed")
    second_hash = compute_corpus_hash(source_dir)

    assert first_hash != second_hash


def test_active_snapshot_pointers_are_profile_specific(tmp_path):
    state_path = tmp_path / "active.json"
    set_active_snapshot_id(
        state_path=state_path,
        embedding_profile="local_minilm",
        snapshot_id="snap-local",
    )
    set_active_snapshot_id(
        state_path=state_path,
        embedding_profile="openai_large",
        snapshot_id="snap-openai",
    )

    assert (
        get_active_snapshot_id(
            state_path=state_path,
            embedding_profile="local_minilm",
        )
        == "snap-local"
    )
    assert (
        get_active_snapshot_id(
            state_path=state_path,
            embedding_profile="openai_large",
        )
        == "snap-openai"
    )


def test_resolve_active_index_path_returns_snapshot_dir(tmp_path):
    root_dir = tmp_path / "indexes"
    state_path = tmp_path / "active.json"

    snapshot_id = build_snapshot_id(
        corpus_hash="abc123",
        embedding_profile="local_minilm",
        chunk_profile="standard",
        parser_profile="docling_v1",
    )
    target_dir = snapshot_dir(root_dir, snapshot_id)
    target_dir.mkdir(parents=True)

    set_active_snapshot_id(
        state_path=state_path,
        embedding_profile="local_minilm",
        snapshot_id=snapshot_id,
    )
    resolved = resolve_active_index_path(
        root_dir=root_dir,
        state_path=state_path,
        embedding_profile="local_minilm",
    )
    assert resolved == Path(target_dir)
