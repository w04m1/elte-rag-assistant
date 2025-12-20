import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from contextlib import asynccontextmanager

DB_FAISS_PATH = "data/vector_store"
MODEL_NAME = "all-MiniLM-L6-v2"

resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store and embedding model on startup."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device.upper()}...")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME, model_kwargs={"device": device}
    )

    if os.path.exists(DB_FAISS_PATH):
        print(f"Loading vector store from {DB_FAISS_PATH}...")
        resources["db"] = FAISS.load_local(
            DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
    else:
        print(
            f"Warning: Vector store not found at {DB_FAISS_PATH}."
        )
        resources["db"] = None

    yield
    resources.clear()


app = FastAPI(title="RAG Assistant API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    k: int = 5


@app.post("/search")
async def search(request: QueryRequest):
    """
    Search the vector database for relevant documents.
    """
    if not resources.get("db"):
        raise HTTPException(
            status_code=503,
            detail="Vector database is not available.",
        )

    results = resources["db"].similarity_search(request.query, k=request.k)

    return {
        "query": request.query,
        "results": [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in results
        ],
    }


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Use /search endpoint to query.",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
