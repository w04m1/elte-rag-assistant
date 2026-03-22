import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.embeddings import get_embeddings
from app.ingest import create_vector_db
from app.rag_chain import ask as rag_ask
from app.runtime_settings import RuntimeSettings, RuntimeSettingsStore
from app.scraper import run_targeted_scrape

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

resources: dict = {}


def _build_bm25(db: FAISS | None):
    if db is None or not settings.retrieval_hybrid:
        return None
    all_docs = list(db.docstore._dict.values())
    if not all_docs:
        return None
    return BM25Retriever.from_documents(all_docs, k=settings.retrieval_fetch_k)


def _load_vector_db(embeddings) -> FAISS | None:
    if not os.path.exists(settings.faiss_index_path):
        logger.warning("Vector store not found at %s.", settings.faiss_index_path)
        return None
    logger.info("Loading vector store from %s...", settings.faiss_index_path)
    db = FAISS.load_local(
        settings.faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("Vector store loaded successfully.")
    return db


def _refresh_retrievers() -> None:
    db = resources.get("db")
    resources["bm25_retriever"] = _build_bm25(db)


def _reload_resources_from_disk() -> None:
    embeddings = resources.get("embeddings")
    if embeddings is None:
        embeddings = get_embeddings()
        resources["embeddings"] = embeddings
    resources["db"] = _load_vector_db(embeddings)
    _refresh_retrievers()


def _set_job_status(job_name: str, **values) -> dict:
    current = resources.setdefault(job_name, {})
    current.update(values)
    resources[job_name] = current
    return current


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


def _run_scrape_job() -> None:
    _set_job_status("scrape_status", status="running", error=None)
    try:
        result = run_targeted_scrape(
            download_dir=settings.scrape_download_path,
            manifest_path=settings.scrape_manifest_path,
        )
        _set_job_status(
            "scrape_status",
            status="completed",
            error=None,
            result=result,
        )
    except Exception as exc:
        logger.exception("Scrape failed")
        _set_job_status("scrape_status", status="failed", error=str(exc))


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
    resources["scrape_status"] = {
        "status": "idle",
        "error": None,
        "result": None,
    }

    resources["db"] = _load_vector_db(embeddings)
    _refresh_retrievers()
    yield
    resources.clear()


app = FastAPI(
    title="ELTE RAG Assistant API",
    version="0.4.0",
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
    document: str
    page: int | None = None
    relevant_snippet: str


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


def _safe_destination(filename: str) -> Path:
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    target_dir = Path(settings.raw_data_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / safe_name


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: QueryRequest):
    db = _require_db()
    bm25 = resources.get("bm25_retriever")
    runtime_settings = _get_runtime_settings_store().get()
    try:
        result = await rag_ask(
            query=request.query,
            db=db,
            bm25_retriever=bm25,
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
    destination = _safe_destination(file.filename or "document.pdf")
    with destination.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)
    return {"status": "uploaded", "file_name": destination.name}


@app.delete("/admin/documents/{file_name}")
async def delete_document(file_name: str):
    destination = _safe_destination(file_name)
    if not destination.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    destination.unlink()
    return {"status": "deleted", "file_name": destination.name}


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


@app.post("/admin/scrape", response_model=JobStatusResponse)
async def trigger_scrape(background_tasks: BackgroundTasks):
    status = resources.get("scrape_status", {})
    if status.get("status") == "running":
        return JobStatusResponse(**status)
    _set_job_status("scrape_status", status="queued", error=None, result=None)
    background_tasks.add_task(_run_scrape_job)
    return JobStatusResponse(**resources["scrape_status"])


@app.get("/admin/scrape", response_model=JobStatusResponse)
async def get_scrape_status():
    return JobStatusResponse(**resources.get("scrape_status", {"status": "idle"}))


@app.get("/health")
async def health():
    db = resources.get("db")
    doc_count = len(db.docstore._dict) if db is not None else 0
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
