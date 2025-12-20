import os
import json
import torch
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from experiment_config import (
    DATA_PATH,
    EXPERIMENT_BASE_PATH,
    EMBEDDING_MODELS,
    CLEANING_STRATEGIES,
    CHUNK_CONFIGS,
)


def create_experiment_vector_db(
    name, model_key="minilm", cleaning_key="basic", chunk_key="default"
):
    experiment_path = os.path.join(EXPERIMENT_BASE_PATH, name)
    os.makedirs(experiment_path, exist_ok=True)

    model_name = EMBEDDING_MODELS[model_key]
    cleaning_fn = CLEANING_STRATEGIES[cleaning_key]
    chunk_config = CHUNK_CONFIGS[chunk_key]

    metadata = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "model_key": model_key,
        "cleaning_strategy": cleaning_key,
        "chunk_config": chunk_config,
        "chunk_key": chunk_key,
    }

    print(f"Experiment: {name}")
    print(f"Model: {model_name}")
    print(f"Cleaning: {cleaning_key}")
    print(
        f"Chunking: {chunk_key} (size={chunk_config['chunk_size']}, overlap={chunk_config['chunk_overlap']})"
    )

    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file
                doc.page_content = cleaning_fn(doc.page_content)
                doc.metadata["experiment"] = name
                doc.metadata["cleaning"] = cleaning_key

            documents.extend(docs)
            print(f"Loaded {len(docs)} pages from {file}")

    print(f"Total documents loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_config["chunk_size"],
        chunk_overlap=chunk_config["chunk_overlap"],
        separators=chunk_config["separators"],
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")

    metadata["total_documents"] = len(documents)
    metadata["total_chunks"] = len(texts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model: {model_name} on {device.upper()}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": device}
    )

    print("Creating vector store...")
    db = FAISS.from_documents(texts, embeddings)

    db.save_local(experiment_path)

    with open(os.path.join(experiment_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Vector store saved to {experiment_path}")
    print(f"Experiment complete!")

    return experiment_path, metadata


def run_experiments(experiments):
    results = []

    for i, exp in enumerate(experiments, 1):
        print(f"Running Experiment {i}/{len(experiments)}")

        try:
            path, metadata = create_experiment_vector_db(**exp)
            results.append({"success": True, "path": path, "metadata": metadata})
        except Exception as e:
            print(f"Error in experiment {exp['name']}: {e}")
            results.append({"success": False, "error": str(e), "experiment": exp})

    return results


if __name__ == "__main__":
    experiments = [
        {
            "name": "exp1_minilm_basic",
            "model_key": "minilm",
            "cleaning_key": "basic",
            "chunk_key": "default",
        },
        {
            "name": "exp2_minilm_normalized",
            "model_key": "minilm",
            "cleaning_key": "normalized",
            "chunk_key": "default",
        },
        {
            "name": "exp3_minilm_minimal",
            "model_key": "minilm",
            "cleaning_key": "minimal",
            "chunk_key": "default",
        },
        {
            "name": "exp4_mpnet_normalized",
            "model_key": "mpnet",
            "cleaning_key": "normalized",
            "chunk_key": "default",
        },
        {
            "name": "exp5_minilm_normalized_small",
            "model_key": "minilm",
            "cleaning_key": "normalized",
            "chunk_key": "small",
        },
        {
            "name": "exp6_minilm_normalized_large",
            "model_key": "minilm",
            "cleaning_key": "normalized",
            "chunk_key": "large",
        },
    ]

    print("Starting experimental pipeline...")
    print(f"Total experiments: {len(experiments)}")

    results = run_experiments(experiments)

    print("EXPERIMENT SUMMARY")
    successful = sum(1 for r in results if r["success"])
    print(f"Successful: {successful}/{len(experiments)}")
    print(f"Failed: {len(experiments) - successful}/{len(experiments)}")
