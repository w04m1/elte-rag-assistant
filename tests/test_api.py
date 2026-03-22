from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.rag_chain import RAGResult
from app.runtime_settings import RuntimeSettingsStore


@pytest.fixture()
def client(test_faiss_db, tmp_path):
    from app.main import app, resources

    runtime_store = RuntimeSettingsStore(tmp_path / "runtime-settings.json")
    resources["db"] = test_faiss_db
    resources["runtime_settings_store"] = runtime_store
    resources["reindex_status"] = {"status": "idle", "error": None, "vector_count": 0}
    resources["scrape_status"] = {"status": "idle", "error": None, "result": None}
    with patch("app.main.settings.runtime_settings_path", str(tmp_path / "runtime-settings.json")):
        with TestClient(app, raise_server_exceptions=False) as c:
            resources["db"] = test_faiss_db
            resources["runtime_settings_store"] = runtime_store
            resources["reindex_status"] = {"status": "idle", "error": None, "vector_count": 0}
            resources["scrape_status"] = {"status": "idle", "error": None, "result": None}
            yield c
    resources.clear()


@pytest.fixture()
def client_no_db(tmp_path):
    from app.main import app, resources

    with (
        patch("app.main.settings") as mock_settings,
        patch("app.main.get_embeddings") as mock_get_embeddings,
    ):
        mock_settings.embedding_provider = "local"
        mock_settings.embedding_model_name = "all-MiniLM-L6-v2"
        mock_settings.faiss_index_path = "/nonexistent/path"
        mock_settings.runtime_settings_path = str(tmp_path / "runtime-settings.json")
        mock_settings.retrieval_hybrid = False
        mock_settings.cors_allow_origins = "*"
        mock_settings.raw_data_path = str(tmp_path / "raw")
        mock_settings.scrape_download_path = str(tmp_path / "raw")
        mock_settings.scrape_manifest_path = str(tmp_path / "scrape.json")
        mock_settings.llm_provider = "openrouter"
        mock_settings.openrouter_model = "generator"
        mock_settings.reranker_model = "reranker"
        mock_settings.openrouter_embedding_model = "embedding"
        mock_settings.ollama_model = "ollama"
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


class TestAdminSettingsEndpoint:
    def test_get_settings(self, client):
        resp = client.get("/admin/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "generator_model" in data
        assert "reranker_model" in data

    def test_update_settings(self, client):
        resp = client.put(
            "/admin/settings",
            json={"generator_model": "demo-generator", "system_prompt": "Demo prompt"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["generator_model"] == "demo-generator"
        assert data["system_prompt"] == "Demo prompt"


class TestAdminDocumentsEndpoint:
    def test_upload_document(self, client, tmp_path):
        upload_dir = tmp_path / "raw"
        upload_dir.mkdir()
        with patch("app.main.settings.raw_data_path", str(upload_dir)):
            resp = client.post(
                "/admin/documents/upload",
                files={"file": ("demo.pdf", b"%PDF-1.4 demo", "application/pdf")},
            )
        assert resp.status_code == 200
        assert (upload_dir / "demo.pdf").exists()

    def test_delete_document(self, client, tmp_path):
        upload_dir = tmp_path / "raw"
        upload_dir.mkdir()
        target = upload_dir / "demo.pdf"
        target.write_bytes(b"%PDF-1.4 demo")
        with patch("app.main.settings.raw_data_path", str(upload_dir)):
            resp = client.delete("/admin/documents/demo.pdf")
        assert resp.status_code == 200
        assert not target.exists()


class TestBackgroundJobs:
    def test_trigger_reindex(self, client):
        with patch("app.main._run_reindex_job") as mock_job:
            resp = client.post("/admin/reindex")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"
        mock_job.assert_called_once()

    def test_trigger_scrape(self, client):
        with patch("app.main._run_scrape_job") as mock_job:
            resp = client.post("/admin/scrape")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"
        mock_job.assert_called_once()


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
