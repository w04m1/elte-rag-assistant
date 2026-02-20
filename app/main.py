import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.embeddings import get_embeddings
from app.rag_chain import ask as rag_ask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store, embedding model, and BM25 retriever on startup."""
    embeddings = get_embeddings()
    resources["embeddings"] = embeddings

    if os.path.exists(settings.faiss_index_path):
        logger.info(
            f"Loading vector store from {settings.faiss_index_path}...",
        )
        db = FAISS.load_local(
            settings.faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        resources["db"] = db
        logger.info("Vector store loaded successfully.")

        # Build BM25 retriever from documents in the FAISS store
        if settings.retrieval_hybrid:
            try:
                from langchain_community.retrievers import BM25Retriever

                all_docs = list(db.docstore._dict.values())
                bm25 = BM25Retriever.from_documents(
                    all_docs, k=settings.retrieval_fetch_k
                )
                resources["bm25_retriever"] = bm25
                logger.info(f"BM25 retriever built from {len(all_docs)} documents.")
            except Exception as exc:
                logger.warning(f"Failed to build BM25 retriever: {exc}")
                resources["bm25_retriever"] = None
    else:
        logger.warning(f"Vector store not found at {settings.faiss_index_path}.")
        resources["db"] = None

    yield
    resources.clear()


app = FastAPI(
    title="ELTE RAG Assistant API",
    version="0.3.0",
    lifespan=lifespan,
)

# TODO: restrict CORS after heavy development phase
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    cited_sources: list[CitedSourceItem] = []


class SearchResultItem(BaseModel):
    content: str
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


def _require_db():
    db = resources.get("db")
    if db is None:
        raise HTTPException(status_code=503, detail="Vector database is not available.")
    return db


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: QueryRequest):
    """query -> retrieve -> rerank -> generate -> answer with cited sources(kind of)"""
    db = _require_db()
    bm25 = resources.get("bm25_retriever")
    try:
        result = await rag_ask(query=request.query, db=db, bm25_retriever=bm25)
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


@app.get("/documents")
async def list_documents():
    """List unique documents present in the FAISS vector store."""
    db = _require_db()
    docs_meta = db.docstore._dict.values()
    unique: dict[str, dict] = {}
    for doc in docs_meta:
        source = doc.metadata.get("source", "unknown")
        if source not in unique:
            unique[source] = {
                "source": source,
                "title": doc.metadata.get("title", source),
            }
    return {"documents": list(unique.values()), "count": len(unique)}


@app.get("/health")
async def health():
    """Health-check returning model info, LLM provider, vector store status."""
    db = resources.get("db")
    doc_count = 0
    if db is not None:
        doc_count = len(db.docstore._dict)
    return {
        "status": "ok",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": (
            settings.embedding_model_name
            if settings.embedding_provider == "local"
            else settings.openrouter_embedding_model
        ),
        "llm_provider": settings.llm_provider,
        "llm_model": (
            settings.openrouter_model
            if settings.llm_provider == "openrouter"
            else settings.ollama_model
        ),
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
