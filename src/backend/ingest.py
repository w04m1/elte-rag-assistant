import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/raw"
DB_FAISS_PATH = "data/vector_store"
MODEL_NAME = "all-MiniLM-L6-v2"


def normalize_text(text):
    """Normalize whitespace while preserving structure."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines)


def create_vector_db():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file
                doc.page_content = normalize_text(doc.page_content)

            documents.extend(docs)
            print(f"Loaded {len(docs)} pages from {file}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device.upper()}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME, model_kwargs={"device": device}
    )

    print("Creating vector store...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Success! Vector store saved to {DB_FAISS_PATH}")


if __name__ == "__main__":
    create_vector_db()
