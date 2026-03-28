import asyncio
import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

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
from app.news_ingest import run_news_pipeline
from app.rag_chain import ask as rag_ask
from app.runtime_settings import RuntimeSettings, RuntimeSettingsStore

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


def _refresh_retrievers() -> None:
    db = resources.get("db")
    resources["bm25_retriever"] = _build_bm25(db)


def _reload_resources_from_disk() -> None:
    embeddings = resources.get("embeddings")
    if embeddings is None:
        embeddings = get_embeddings()
        resources["embeddings"] = embeddings
    resources["db"] = _load_vector_db(embeddings, settings.faiss_index_path, "document")
    resources["news_db"] = _load_vector_db(
        embeddings, settings.news_faiss_index_path, "news"
    )
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
        create_vector_db()
        _reload_resources_from_disk()
        db = resources.get("db")
        chunk_count = len(db.docstore._dict) if db is not None else 0
        _set_job_status(
            "reindex_status",
            status="completed",
            error=None,
            vector_count=chunk_count,
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


async def _periodic_news_sync() -> None:
    interval_seconds = max(60, int(settings.news_sync_interval_seconds))
    while True:
        await asyncio.sleep(interval_seconds)
        await asyncio.to_thread(_run_news_job, "sync")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load runtime settings and retrieval resources on startup."""
    embeddings = get_embeddings()
    resources["embeddings"] = embeddings
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

    resources["db"] = _load_vector_db(embeddings, settings.faiss_index_path, "document")
    resources["news_db"] = _load_vector_db(
        embeddings, settings.news_faiss_index_path, "news"
    )
    _refresh_retrievers()

    sync_task = asyncio.create_task(_periodic_news_sync())
    resources["news_sync_task"] = sync_task

    try:
        yield
    finally:
        sync_task.cancel()
        with suppress(asyncio.CancelledError):
            await sync_task
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


class QueryRequest(BaseModel):
    query: str


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
    answer: str
    sources: list[SourceItem]
    model_used: str
    reasoning: str = ""
    confidence: str = ""
    cited_sources: list[CitedSourceItem] = Field(default_factory=list)


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


class JobStatusResponse(BaseModel):
    status: str
    error: str | None = None
    vector_count: int | None = None
    result: dict | None = None


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


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: QueryRequest):
    db = _require_db()
    bm25 = resources.get("bm25_retriever")
    news_db = resources.get("news_db")
    runtime_settings = _get_runtime_settings_store().get()
    try:
        result = await rag_ask(
            query=request.query,
            db=db,
            bm25_retriever=bm25,
            news_db=news_db,
            system_prompt=runtime_settings.system_prompt,
            generator_model=runtime_settings.generator_model,
            reranker_model=runtime_settings.reranker_model,
        )
    except Exception as exc:
        logger.exception("RAG chain error")
        raise HTTPException(
            status_code=502, detail=f"LLM generation failed: {exc}"
        ) from exc

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceItem(content=s["content"], document=s["document"], page=s.get("page"))
            for s in result.sources
        ],
        model_used=result.model_used,
        reasoning=result.reasoning,
        confidence=result.confidence,
        cited_sources=[CitedSourceItem(**cs) for cs in result.cited_sources],
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
    return store.update(
        generator_model=request.generator_model,
        reranker_model=request.reranker_model,
        system_prompt=request.system_prompt,
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


@app.get("/health")
async def health():
    db = resources.get("db")
    news_db = resources.get("news_db")
    doc_count = len(db.docstore._dict) if db is not None else 0
    news_count = len(news_db.docstore._dict) if news_db is not None else 0
    runtime_settings = _get_runtime_settings_store().get()
    return {
        "status": "ok",
        "embedding_provider": runtime_settings.embedding_provider,
        "embedding_model": runtime_settings.embedding_model,
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
