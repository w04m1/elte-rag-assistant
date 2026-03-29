import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

from app.config import settings
from app.document_sync import run_documents_sync
from app.embeddings import get_embeddings
from app.ingest import create_vector_db
from app.index_snapshots import (
    build_snapshot_id,
    compute_corpus_hash,
    resolve_active_index_path,
    set_active_snapshot_id,
    write_snapshot_manifest,
)
from app.news_ingest import run_news_pipeline
from app.profiles import EmbeddingProfile, PipelineMode, RerankerMode
from app.rag_chain import ask as rag_ask
from app.runtime_settings import (
    RuntimeSettings,
    RuntimeSettingsStore,
    compose_system_prompt,
)
from app.usage_log import (
    append_usage_entry,
    compute_usage_stats,
    read_recent_usage_entries,
    set_usage_feedback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

resources: dict = {}
_news_job_lock = threading.Lock()


def _build_bm25(db: FAISS | None):
    if db is None or not settings.retrieval_hybrid:
        return None
    all_docs = list(db.docstore._dict.values())
    if not all_docs:
        return None
    return BM25Retriever.from_documents(all_docs, k=settings.retrieval_fetch_k)


def _load_vector_db(embeddings, index_path: str, label: str) -> FAISS | None:
    if not os.path.exists(index_path):
        logger.warning("%s vector store not found at %s.", label.capitalize(), index_path)
        return None
    logger.info("Loading %s vector store from %s...", label, index_path)
    db = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("%s vector store loaded successfully.", label.capitalize())
    return db


def _resolve_document_index_path(runtime_settings: RuntimeSettings) -> str:
    profile: EmbeddingProfile = runtime_settings.embedding_profile
    snapshot_path = resolve_active_index_path(
        root_dir=settings.index_snapshots_root_path,
        state_path=settings.active_index_state_path,
        embedding_profile=profile,
    )
    if snapshot_path is not None:
        return str(snapshot_path)
    return settings.faiss_index_path


def _refresh_retrievers() -> None:
    db = resources.get("db")
    resources["bm25_retriever"] = _build_bm25(db)


def _reload_resources_from_disk(runtime_settings: RuntimeSettings | None = None) -> None:
    if runtime_settings is None:
        runtime_settings = _get_runtime_settings_store().get()

    try:
        embeddings = get_embeddings(embedding_profile=runtime_settings.embedding_profile)
    except Exception:
        logger.exception(
            "Failed to initialize embeddings for profile '%s'",
            runtime_settings.embedding_profile,
        )
        resources["embeddings"] = None
        resources["db"] = None
        resources["news_db"] = None
        _refresh_retrievers()
        return

    resources["embeddings"] = embeddings

    document_index_path = _resolve_document_index_path(runtime_settings)
    resources["db"] = _load_vector_db(embeddings, document_index_path, "document")
    resources["news_db"] = _load_vector_db(embeddings, settings.news_faiss_index_path, "news")
    _refresh_retrievers()


def _set_job_status(job_name: str, **values) -> dict:
    current = resources.setdefault(job_name, {})
    current.update(values)
    resources[job_name] = current
    return current


def _default_news_status(status: str = "idle") -> dict:
    return {
        "status": status,
        "error": None,
        "mode": None,
        "pages": None,
        "processed_count": 0,
        "added_count": 0,
        "updated_count": 0,
        "unchanged_count": 0,
        "embedded_count": 0,
        "last_run_at": None,
    }


def _run_reindex_job() -> None:
    _set_job_status("reindex_status", status="running", error=None)
    try:
        runtime_settings = _get_runtime_settings_store().get()
        corpus_hash = compute_corpus_hash(settings.raw_data_path)
        snapshot_id = build_snapshot_id(
            corpus_hash=corpus_hash,
            embedding_profile=runtime_settings.embedding_profile,
            chunk_profile=runtime_settings.chunk_profile,
            parser_profile=runtime_settings.parser_profile,
        )
        snapshot_dir = (
            Path(settings.index_snapshots_root_path) / snapshot_id
        )
        ingest_summary = create_vector_db(
            source_dir=settings.raw_data_path,
            output_dir=str(snapshot_dir),
            embedding_profile=runtime_settings.embedding_profile,
            chunk_profile=runtime_settings.chunk_profile,
            parser_profile=runtime_settings.parser_profile,
        )

        manifest_payload = {
            "snapshot_id": snapshot_id,
            "built_at_utc": datetime.now(UTC).isoformat(),
            "corpus_hash": corpus_hash,
            "embedding_profile": runtime_settings.embedding_profile,
            "embedding_provider": runtime_settings.embedding_provider,
            "embedding_model": runtime_settings.embedding_model,
            "chunk_profile": runtime_settings.chunk_profile,
            "parser_profile": runtime_settings.parser_profile,
            "max_tokens": ingest_summary.get("max_tokens"),
            "source_count": ingest_summary.get("source_count", 0),
            "chunk_count": ingest_summary.get("chunk_count", 0),
            "output_dir": str(snapshot_dir),
        }
        write_snapshot_manifest(
            root_dir=settings.index_snapshots_root_path,
            snapshot_id=snapshot_id,
            payload=manifest_payload,
        )
        set_active_snapshot_id(
            state_path=settings.active_index_state_path,
            embedding_profile=runtime_settings.embedding_profile,
            snapshot_id=snapshot_id,
        )

        _reload_resources_from_disk(runtime_settings=runtime_settings)
        db = resources.get("db")
        chunk_count = len(db.docstore._dict) if db is not None else 0
        _set_job_status(
            "reindex_status",
            status="completed",
            error=None,
            vector_count=chunk_count,
            result={
                "snapshot_id": snapshot_id,
                "embedding_profile": runtime_settings.embedding_profile,
                "chunk_count": chunk_count,
            },
        )
    except Exception as exc:
        logger.exception("Reindex failed")
        _set_job_status("reindex_status", status="failed", error=str(exc))


def _run_documents_sync_job() -> None:
    _set_job_status("documents_sync_status", status="running", error=None, result=None)
    try:
        result = run_documents_sync(
            download_dir=settings.raw_data_path,
            state_path=settings.documents_sync_state_path,
        )
        _set_job_status(
            "documents_sync_status",
            status="completed",
            error=None,
            result=result,
        )
    except Exception as exc:
        logger.exception("Documents sync failed")
        _set_job_status("documents_sync_status", status="failed", error=str(exc))


def _run_news_job(mode: Literal["bootstrap", "sync"]) -> None:
    if not _news_job_lock.acquire(blocking=False):
        logger.info("News job already running; skipping %s request", mode)
        return

    _set_job_status(
        "news_status",
        status="running",
        error=None,
        mode=mode,
    )

    try:
        result = run_news_pipeline(mode=mode)
        _reload_resources_from_disk()
        _set_job_status(
            "news_status",
            status="completed",
            error=None,
            mode=mode,
            pages=result.get("pages"),
            processed_count=result.get("processed_count", 0),
            added_count=result.get("added_count", 0),
            updated_count=result.get("updated_count", 0),
            unchanged_count=result.get("unchanged_count", 0),
            embedded_count=result.get("embedded_count", 0),
            last_run_at=datetime.now(UTC).isoformat(),
        )
    except Exception as exc:
        logger.exception("News %s failed", mode)
        _set_job_status(
            "news_status",
            status="failed",
            error=str(exc),
            mode=mode,
            last_run_at=datetime.now(UTC).isoformat(),
        )
    finally:
        _news_job_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load runtime settings and retrieval resources on startup."""
    resources["runtime_settings_store"] = RuntimeSettingsStore(
        settings.runtime_settings_path
    )
    resources["reindex_status"] = {
        "status": "idle",
        "error": None,
        "vector_count": 0,
    }
    resources["documents_sync_status"] = {
        "status": "idle",
        "error": None,
        "result": None,
    }
    resources["news_status"] = _default_news_status()

    _reload_resources_from_disk()

    try:
        yield
    finally:
        resources.clear()


app = FastAPI(
    title="ELTE RAG Assistant API",
    version="0.5.0",
    lifespan=lifespan,
)

allow_origins = (
    ["*"]
    if settings.cors_allow_origins.strip() == "*"
    else [origin.strip() for origin in settings.cors_allow_origins.split(",")]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatHistoryCitedSource(BaseModel):
    citation_id: str
    source: str
    document: str
    page: int | None = None
    relevant_snippet: str
    source_type: Literal["pdf", "news"] = "pdf"
    published_at: str | None = None


class ChatHistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    cited_sources: list[ChatHistoryCitedSource] | None = None


class QueryRequest(BaseModel):
    query: str
    history: list[ChatHistoryTurn] = Field(default_factory=list)


class SourceItem(BaseModel):
    content: str
    document: str
    page: int | None = None


class CitedSourceItem(BaseModel):
    citation_id: str
    source: str
    document: str
    page: int | None = None
    relevant_snippet: str
    source_type: Literal["pdf", "news"] = "pdf"
    published_at: str | None = None


class AskResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[SourceItem]
    model_used: str
    reasoning: str = ""
    confidence: str = ""
    cited_sources: list[CitedSourceItem] = Field(default_factory=list)
    answer_mode: Literal["llm", "deterministic"] = "llm"
    verification_passed: bool | None = None
    guardrails_triggered: list[str] = Field(default_factory=list)


class DocumentListItem(BaseModel):
    source: str
    title: str
    chunk_count: int = 0


class DocumentsResponse(BaseModel):
    documents: list[DocumentListItem]
    count: int


class RuntimeSettingsUpdateRequest(BaseModel):
    generator_model: str | None = None
    reranker_model: str | None = None
    system_prompt: str | None = None
    embedding_profile: EmbeddingProfile | None = None
    pipeline_mode: PipelineMode | None = None
    reranker_mode: RerankerMode | None = None
    chunk_profile: str | None = None
    parser_profile: str | None = None
    max_chunks_per_doc: int | None = None


class JobStatusResponse(BaseModel):
    status: str
    error: str | None = None
    vector_count: int | None = None
    result: dict | None = None


class ActiveIndexResponse(BaseModel):
    embedding_profile: EmbeddingProfile
    index_path: str
    from_snapshot: bool


class NewsJobStatusResponse(BaseModel):
    status: str
    error: str | None = None
    mode: Literal["bootstrap", "sync"] | None = None
    pages: int | None = None
    processed_count: int | None = None
    added_count: int | None = None
    updated_count: int | None = None
    unchanged_count: int | None = None
    embedded_count: int | None = None
    last_run_at: str | None = None


class UsageSourceTypes(BaseModel):
    pdf: int = 0
    news: int = 0


class UsageLogEntry(BaseModel):
    request_id: str = ""
    timestamp_utc: str
    query_text: str
    answer_length_chars: int = 0
    confidence: str = "unknown"
    model_used: str = ""
    reranker_model: str = ""
    latency_ms: float = 0.0
    cited_sources_count: int = 0
    source_types: UsageSourceTypes = Field(default_factory=UsageSourceTypes)
    feedback: bool | None = None
    feedback_timestamp_utc: str | None = None
    status: Literal["ok", "error"] = "ok"


class UsageLogResponse(BaseModel):
    entries: list[UsageLogEntry]
    count: int
    limit: int


class UsageStatsResponse(BaseModel):
    window_days: int
    generated_at_utc: str
    total_queries: int
    avg_latency_ms: float
    citation_presence_rate: float
    non_empty_answer_rate: float
    confidence_distribution: dict[str, int]
    source_mix_pdf_vs_news: UsageSourceTypes
    helpful_feedback_count: int
    unhelpful_feedback_count: int
    helpful_feedback_rate: float
    feedback_coverage_rate: float


class FeedbackRequest(BaseModel):
    request_id: str
    helpful: bool


class FeedbackResponse(BaseModel):
    status: Literal["updated"]
    request_id: str
    helpful: bool


def _require_db():
    db = resources.get("db")
    if db is None:
        raise HTTPException(status_code=503, detail="Vector database is not available.")
    return db


def _get_runtime_settings_store() -> RuntimeSettingsStore:
    store = resources.get("runtime_settings_store")
    if store is None:
        store = RuntimeSettingsStore(settings.runtime_settings_path)
        resources["runtime_settings_store"] = store
    return store


def _collect_documents(db: FAISS) -> list[DocumentListItem]:
    grouped: dict[str, DocumentListItem] = {}
    for doc in db.docstore._dict.values():
        source = doc.metadata.get("source", "unknown")
        if source not in grouped:
            grouped[source] = DocumentListItem(
                source=source,
                title=doc.metadata.get("title", source),
                chunk_count=0,
            )
        grouped[source].chunk_count += 1
    return sorted(grouped.values(), key=lambda item: item.title.lower())


def _safe_upload_destination(filename: str) -> Path:
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    target_dir = Path(settings.raw_data_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / safe_name


def _safe_source_destination(filename: str) -> Path:
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".pdf", ".doc", ".docx"}:
        raise HTTPException(status_code=400, detail="Unsupported source file type.")
    target_dir = Path(settings.raw_data_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / safe_name


def _count_source_types(cited_sources: list[dict]) -> dict[str, int]:
    counts = {"pdf": 0, "news": 0}
    for source in cited_sources:
        source_type = str(source.get("source_type", "pdf")).strip().lower()
        if source_type == "news":
            counts["news"] += 1
        else:
            counts["pdf"] += 1
    return counts


def _safe_append_usage_entry(payload: dict) -> None:
    try:
        append_usage_entry(settings.usage_log_path, payload)
    except Exception:
        logger.exception("Failed to write usage log entry")


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: QueryRequest):
    started_at = perf_counter()
    request_id = uuid4().hex
    runtime_settings = _get_runtime_settings_store().get()
    chat_history_payload: list[dict] = []
    for turn in request.history:
        payload_turn: dict = {"role": turn.role, "text": turn.text}
        if turn.role == "assistant" and turn.cited_sources:
            payload_turn["cited_sources"] = [
                cited_source.model_dump() for cited_source in turn.cited_sources
            ]
        chat_history_payload.append(payload_turn)

    try:
        db = _require_db()
        bm25 = resources.get("bm25_retriever")
        news_db = resources.get("news_db")
        result = await rag_ask(
            query=request.query,
            chat_history=chat_history_payload,
            db=db,
            bm25_retriever=bm25,
            news_db=news_db,
            system_prompt=compose_system_prompt(runtime_settings.system_prompt),
            generator_model=runtime_settings.generator_model,
            reranker_model=runtime_settings.reranker_model,
            pipeline_mode=runtime_settings.pipeline_mode,
            reranker_mode=runtime_settings.reranker_mode,
            max_chunks_per_doc=runtime_settings.max_chunks_per_doc,
        )
    except HTTPException:
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        _safe_append_usage_entry(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "query_text": request.query,
                "answer_length_chars": 0,
                "confidence": "unknown",
                "model_used": runtime_settings.generator_model,
                "reranker_model": runtime_settings.reranker_model,
                "latency_ms": latency_ms,
                "cited_sources_count": 0,
                "source_types": {"pdf": 0, "news": 0},
                "status": "error",
            }
        )
        raise
    except Exception as exc:
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        _safe_append_usage_entry(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "query_text": request.query,
                "answer_length_chars": 0,
                "confidence": "unknown",
                "model_used": runtime_settings.generator_model,
                "reranker_model": runtime_settings.reranker_model,
                "latency_ms": latency_ms,
                "cited_sources_count": 0,
                "source_types": {"pdf": 0, "news": 0},
                "status": "error",
            }
        )
        logger.exception("RAG chain error")
        raise HTTPException(
            status_code=502, detail=f"LLM generation failed: {exc}"
        ) from exc

    response = AskResponse(
        request_id=request_id,
        answer=result.answer,
        sources=[
            SourceItem(content=s["content"], document=s["document"], page=s.get("page"))
            for s in result.sources
        ],
        model_used=result.model_used,
        reasoning=result.reasoning,
        confidence=result.confidence,
        cited_sources=[CitedSourceItem(**cs) for cs in result.cited_sources],
        answer_mode=result.answer_mode,
        verification_passed=result.verification_passed,
        guardrails_triggered=result.guardrails_triggered,
    )
    latency_ms = round((perf_counter() - started_at) * 1000, 2)
    _safe_append_usage_entry(
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "query_text": request.query,
            "answer_length_chars": len(result.answer or ""),
            "confidence": result.confidence or "unknown",
            "model_used": result.model_used or runtime_settings.generator_model,
            "reranker_model": runtime_settings.reranker_model,
            "latency_ms": latency_ms,
            "cited_sources_count": len(result.cited_sources),
            "source_types": _count_source_types(result.cited_sources),
            "status": "ok",
        }
    )
    return response


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    updated = set_usage_feedback(
        settings.usage_log_path,
        request_id=request.request_id,
        helpful=request.helpful,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Request ID not found.")
    return FeedbackResponse(
        status="updated",
        request_id=request.request_id,
        helpful=request.helpful,
    )


@app.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    db = _require_db()
    documents = _collect_documents(db)
    return DocumentsResponse(documents=documents, count=len(documents))


@app.get("/admin/settings", response_model=RuntimeSettings)
async def get_admin_settings():
    return _get_runtime_settings_store().get()


@app.put("/admin/settings", response_model=RuntimeSettings)
async def update_admin_settings(request: RuntimeSettingsUpdateRequest):
    store = _get_runtime_settings_store()
    previous = store.get()
    updated = store.update(
        generator_model=request.generator_model,
        reranker_model=request.reranker_model,
        system_prompt=request.system_prompt,
        embedding_profile=request.embedding_profile,
        pipeline_mode=request.pipeline_mode,
        reranker_mode=request.reranker_mode,
        chunk_profile=request.chunk_profile,
        parser_profile=request.parser_profile,
        max_chunks_per_doc=request.max_chunks_per_doc,
    )
    if updated.embedding_profile != previous.embedding_profile:
        _reload_resources_from_disk(runtime_settings=updated)
    return updated


@app.get("/admin/indexes/active", response_model=ActiveIndexResponse)
async def get_active_index():
    runtime_settings = _get_runtime_settings_store().get()
    snapshot_path = resolve_active_index_path(
        root_dir=settings.index_snapshots_root_path,
        state_path=settings.active_index_state_path,
        embedding_profile=runtime_settings.embedding_profile,
    )
    if snapshot_path is not None:
        return ActiveIndexResponse(
            embedding_profile=runtime_settings.embedding_profile,
            index_path=str(snapshot_path),
            from_snapshot=True,
        )
    return ActiveIndexResponse(
        embedding_profile=runtime_settings.embedding_profile,
        index_path=settings.faiss_index_path,
        from_snapshot=False,
    )


@app.get("/admin/documents", response_model=DocumentsResponse)
async def list_admin_documents():
    return await list_documents()


@app.post("/admin/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    destination = _safe_upload_destination(file.filename or "document.pdf")
    with destination.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)
    return {"status": "uploaded", "file_name": destination.name}


@app.delete("/admin/documents/{file_name}")
async def delete_document(file_name: str):
    destination = _safe_upload_destination(file_name)
    if not destination.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    destination.unlink()
    return {"status": "deleted", "file_name": destination.name}


@app.get("/files/{file_name}")
async def get_source_file(file_name: str):
    destination = _safe_source_destination(file_name)
    if not destination.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    media_type_by_ext = {
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    media_type = media_type_by_ext.get(destination.suffix.lower(), "application/octet-stream")
    return FileResponse(
        destination,
        media_type=media_type,
        filename=destination.name,
        content_disposition_type="inline",
    )


@app.post("/admin/reindex", response_model=JobStatusResponse)
async def trigger_reindex(background_tasks: BackgroundTasks):
    status = resources.get("reindex_status", {})
    if status.get("status") == "running":
        return JobStatusResponse(**status)
    _set_job_status("reindex_status", status="queued", error=None)
    background_tasks.add_task(_run_reindex_job)
    return JobStatusResponse(**resources["reindex_status"])


@app.get("/admin/reindex", response_model=JobStatusResponse)
async def get_reindex_status():
    return JobStatusResponse(**resources.get("reindex_status", {"status": "idle"}))


@app.post("/admin/documents/sync", response_model=JobStatusResponse)
async def trigger_documents_sync(background_tasks: BackgroundTasks):
    status = resources.get("documents_sync_status", {})
    if status.get("status") == "running":
        return JobStatusResponse(**status)
    _set_job_status("documents_sync_status", status="queued", error=None, result=None)
    background_tasks.add_task(_run_documents_sync_job)
    return JobStatusResponse(**resources["documents_sync_status"])


@app.get("/admin/documents/sync", response_model=JobStatusResponse)
async def get_documents_sync_status():
    return JobStatusResponse(
        **resources.get("documents_sync_status", {"status": "idle"})
    )


@app.post("/admin/news/bootstrap", response_model=NewsJobStatusResponse)
async def trigger_news_bootstrap(background_tasks: BackgroundTasks):
    status = resources.get("news_status", _default_news_status())
    if status.get("status") == "running":
        return NewsJobStatusResponse(**status)
    next_status = _default_news_status(status="queued")
    next_status["mode"] = "bootstrap"
    _set_job_status(
        "news_status",
        **next_status,
    )
    background_tasks.add_task(_run_news_job, "bootstrap")
    return NewsJobStatusResponse(**resources["news_status"])


@app.post("/admin/news/sync", response_model=NewsJobStatusResponse)
async def trigger_news_sync(background_tasks: BackgroundTasks):
    status = resources.get("news_status", _default_news_status())
    if status.get("status") == "running":
        return NewsJobStatusResponse(**status)
    next_status = _default_news_status(status="queued")
    next_status["mode"] = "sync"
    _set_job_status(
        "news_status",
        **next_status,
    )
    background_tasks.add_task(_run_news_job, "sync")
    return NewsJobStatusResponse(**resources["news_status"])


@app.get("/admin/news", response_model=NewsJobStatusResponse)
async def get_news_status():
    return NewsJobStatusResponse(**resources.get("news_status", _default_news_status()))


@app.get("/admin/usage", response_model=UsageLogResponse)
async def get_usage_entries(limit: int = 200):
    bounded_limit = min(max(1, int(limit)), 1000)
    entries = read_recent_usage_entries(settings.usage_log_path, limit=bounded_limit)
    return UsageLogResponse(
        entries=[UsageLogEntry(**entry) for entry in entries],
        count=len(entries),
        limit=bounded_limit,
    )


@app.get("/admin/usage/stats", response_model=UsageStatsResponse)
async def get_usage_statistics(window_days: int = 7):
    bounded_window_days = min(max(1, int(window_days)), 365)
    stats = compute_usage_stats(settings.usage_log_path, window_days=bounded_window_days)
    return UsageStatsResponse(**stats)


@app.get("/health")
async def health():
    db = resources.get("db")
    news_db = resources.get("news_db")
    doc_count = len(db.docstore._dict) if db is not None else 0
    news_count = len(news_db.docstore._dict) if news_db is not None else 0
    runtime_settings = _get_runtime_settings_store().get()
    return {
        "status": "ok",
        "embedding_profile": runtime_settings.embedding_profile,
        "embedding_provider": runtime_settings.embedding_provider,
        "embedding_model": runtime_settings.embedding_model,
        "pipeline_mode": runtime_settings.pipeline_mode,
        "reranker_mode": runtime_settings.reranker_mode,
        "llm_provider": settings.llm_provider,
        "llm_model": runtime_settings.generator_model,
        "reranker_model": runtime_settings.reranker_model,
        "vector_store_loaded": db is not None,
        "vector_count": doc_count,
        "news_vector_store_loaded": news_db is not None,
        "news_vector_count": news_count,
    }


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "ELTE RAG Assistant API. /docs for Swagger",
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
