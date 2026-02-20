from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.rag_chain import RAGResult


@pytest.fixture()
def client(test_faiss_db):
    from app.main import app, resources

    resources["db"] = test_faiss_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    resources.clear()


@pytest.fixture()
def client_no_db():
    from app.main import app, resources

    with (
        patch("app.main.settings") as mock_settings,
        patch("app.main.get_embeddings") as mock_get_embeddings,
    ):
        mock_settings.embedding_provider = "local"
        mock_settings.embedding_model_name = "all-MiniLM-L6-v2"
        mock_settings.faiss_index_path = "/nonexistent/path"
        mock_settings.retrieval_hybrid = False
        mock_get_embeddings.return_value = MagicMock()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    resources.clear()


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["vector_store_loaded"] is True
        assert data["vector_count"] > 0

    def test_health_no_db(self, client_no_db):
        resp = client_no_db.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vector_store_loaded"] is False


class TestRootEndpoint:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "ok" in resp.json()["status"]


# class TestSearchEndpoint:
#     def test_search_returns_results(self, client):
#         resp = client.post("/search", json={"query": "thesis submission"})
#         assert resp.status_code == 200
#         data = resp.json()
#         assert data["query"] == "thesis submission"
#         assert len(data["results"]) > 0
#
#     def test_search_no_db(self, client_no_db):
#         resp = client_no_db.post("/search", json={"query": "test"})
#         assert resp.status_code == 503
#
#     def test_search_ignores_k_in_body(self, client):
#         """k should not be accepted in the request body (server-side config only)."""
#         resp = client.post("/search", json={"query": "thesis submission", "k": 2})
#         assert resp.status_code == 200


class TestAskEndpoint:
    def test_ask_with_mock_llm(self, client):
        mock_result = RAGResult(
            answer="Students must submit by April 15th. [Thesis Rules, p. 3]",
            sources=[{"content": "...", "document": "Thesis Rules", "page": 3}],
            model_used="mock-model",
            reasoning="The context from Thesis Rules p.3 states the deadline.",
            confidence="high",
            cited_sources=[
                {
                    "document": "Thesis Rules",
                    "page": 3,
                    "relevant_snippet": "Students must submit...",
                }
            ],
        )
        with patch(
            "app.main.rag_ask", new_callable=AsyncMock, return_value=mock_result
        ):
            resp = client.post("/ask", json={"query": "When is the thesis deadline?"})
            assert resp.status_code == 200
            data = resp.json()
            assert "April 15th" in data["answer"]
            assert data["model_used"] == "mock-model"
            assert len(data["sources"]) == 1
            assert (
                data["reasoning"]
                == "The context from Thesis Rules p.3 states the deadline."
            )
            assert data["confidence"] == "high"
            assert len(data["cited_sources"]) == 1

    def test_ask_no_db(self, client_no_db):
        resp = client_no_db.post("/ask", json={"query": "test"})
        assert resp.status_code == 503

    def test_ask_without_k(self, client):
        """Verify k is no longer required or used in request body."""
        mock_result = RAGResult(
            answer="Test answer",
            sources=[],
            model_used="mock-model",
        )
        with patch(
            "app.main.rag_ask", new_callable=AsyncMock, return_value=mock_result
        ):
            resp = client.post("/ask", json={"query": "test question"})
            assert resp.status_code == 200

    def test_ask_response_has_structured_fields(self, client):
        """Verify response includes reasoning, confidence, and cited_sources."""
        mock_result = RAGResult(
            answer="Answer text",
            sources=[],
            model_used="model",
            reasoning="",
            confidence="",
            cited_sources=[],
        )
        with patch(
            "app.main.rag_ask", new_callable=AsyncMock, return_value=mock_result
        ):
            resp = client.post("/ask", json={"query": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert "reasoning" in data
            assert "confidence" in data
            assert "cited_sources" in data


class TestDocumentsEndpoint:
    def test_list_documents(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert isinstance(data["documents"], list)

    def test_documents_no_db(self, client_no_db):
        resp = client_no_db.get("/documents")
        assert resp.status_code == 503
