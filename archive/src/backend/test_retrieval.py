import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "data/vector_store"
MODEL_NAME = "all-MiniLM-L6-v2"


def test_vector_store():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading vector store... (using {device.upper()})")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME, model_kwargs={"device": device}
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )

    print(f"Vector Store Statistics:")
    print(f"Total vectors: {db.index.ntotal}")

    test_queries = [
        "What are the requirements for doctoral studies?",
        "How do I enroll in courses?",
        "What is the thesis submission process?",
        "What are the language requirements?",
        "Physical education requirements",
    ]

    print("Testing Retrieval Quality:")

    for query in test_queries:
        print(f"Query: '{query}'")

        results = db.similarity_search_with_score(query, k=3)

        for i, (doc, score) in enumerate(results, 1):
            print(f"Result {i} (Distance: {score:.4f}):")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Content preview: {doc.page_content[:200]}...")

    print("Retrieval Depth Analysis:")
    test_query = "doctoral thesis requirements"

    for k in [1, 3, 5, 10]:
        results = db.similarity_search_with_score(test_query, k=k)
        print(f"Top-{k} results for '{test_query}':")
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            print(f"{i}. {source[:50]:<50} | Distance: {score:.4f}")


if __name__ == "__main__":
    test_vector_store()
