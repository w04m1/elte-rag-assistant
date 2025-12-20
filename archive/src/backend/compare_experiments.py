import os
import json
import torch
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from experiment_config import EXPERIMENT_BASE_PATH, TEST_QUERIES, EMBEDDING_MODELS


def load_experiment(experiment_name):
    experiment_path = os.path.join(EXPERIMENT_BASE_PATH, experiment_name)

    with open(os.path.join(experiment_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    model_name = metadata["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": device}
    )
    db = FAISS.load_local(
        experiment_path, embeddings, allow_dangerous_deserialization=True
    )

    return db, metadata


def evaluate_experiment(experiment_name, queries=None, k=3):
    if queries is None:
        queries = TEST_QUERIES

    db, metadata = load_experiment(experiment_name)

    results = {"experiment": experiment_name, "metadata": metadata, "queries": []}

    for query in queries:
        search_results = db.similarity_search_with_score(query, k=k)

        query_result = {"query": query, "top_k": k, "results": []}

        for doc, score in search_results:
            query_result["results"].append(
                {
                    "score": float(score),
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "content_preview": doc.page_content[:150],
                }
            )

        avg_score = sum(r["score"] for r in query_result["results"]) / len(
            query_result["results"]
        )
        query_result["avg_distance"] = avg_score
        query_result["min_distance"] = min(r["score"] for r in query_result["results"])

        results["queries"].append(query_result)

    all_avg_scores = [q["avg_distance"] for q in results["queries"]]
    results["overall_avg_distance"] = sum(all_avg_scores) / len(all_avg_scores)
    results["overall_min_distance"] = min(q["min_distance"] for q in results["queries"])

    return results


def compare_experiments(experiment_names, queries=None, k=3):
    if queries is None:
        queries = TEST_QUERIES

    all_results = []

    print(f"Comparing {len(experiment_names)} experiments")

    for exp_name in experiment_names:
        print(f"Evaluating: {exp_name}...")
        try:
            result = evaluate_experiment(exp_name, queries, k)
            all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {exp_name}: {e}")

    print("Results Comparison")
    print(f"{'Experiment':<35} {'Avg Dist':<12} {'Min Dist':<12} {'Chunks':<10}")

    for result in all_results:
        exp_name = result["experiment"]
        avg_dist = result["overall_avg_distance"]
        min_dist = result["overall_min_distance"]
        chunks = result["metadata"]["total_chunks"]
        print(f"{exp_name:<35} {avg_dist:<12.4f} {min_dist:<12.4f} {chunks:<10}")

    best_exp = min(all_results, key=lambda x: x["overall_avg_distance"])
    print(f"Best Performer: {best_exp['experiment']}")
    print(f"Average Distance: {best_exp['overall_avg_distance']:.4f}")
    print(f"Configuration:")
    print(f"  - Model: {best_exp['metadata']['model_key']}")
    print(f"  - Cleaning: {best_exp['metadata']['cleaning_strategy']}")
    print(f"  - Chunking: {best_exp['metadata']['chunk_key']}")

    print("Query-by-Query Comparison")

    for i, query in enumerate(queries):
        print(f"Query {i + 1}: '{query}'")
        print(
            f"{'Experiment':<35} {'Avg Dist':<12} {'Top-1 Dist':<12} {'Top-1 Source':<30}"
        )

        for result in all_results:
            query_result = result["queries"][i]
            exp_name = result["experiment"]
            avg = query_result["avg_distance"]
            top1_dist = query_result["results"][0]["score"]
            top1_source = query_result["results"][0]["source"][:30]
            print(f"{exp_name:<35} {avg:<12.4f} {top1_dist:<12.4f} {top1_source:<30}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(EXPERIMENT_BASE_PATH, f"comparison_{timestamp}.json")

    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Detailed report saved to: {report_path}")

    return all_results


if __name__ == "__main__":
    experiments = [
        d
        for d in os.listdir(EXPERIMENT_BASE_PATH)
        if os.path.isdir(os.path.join(EXPERIMENT_BASE_PATH, d)) and d.startswith("exp")
    ]

    if not experiments:
        print("No experiments found. Run ingest_experiment.py first!")
    else:
        print(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  - {exp}")

        compare_experiments(experiments, k=3)
