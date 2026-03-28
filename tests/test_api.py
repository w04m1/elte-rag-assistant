import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
    resources["documents_sync_status"] = {
        "status": "idle",
        "error": None,
        "result": None,
    }
    resources["news_status"] = {
        "status": "idle",
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
    resources["news_db"] = None
    with (
        patch("app.main.settings.runtime_settings_path", str(tmp_path / "runtime-settings.json")),
        patch("app.main.settings.usage_log_path", str(tmp_path / "usage-log.jsonl")),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            resources["db"] = test_faiss_db
            resources["runtime_settings_store"] = runtime_store
            resources["reindex_status"] = {"status": "idle", "error": None, "vector_count": 0}
            resources["documents_sync_status"] = {
                "status": "idle",
                "error": None,
                "result": None,
            }
            resources["news_status"] = {
                "status": "idle",
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
            resources["news_db"] = None
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
        mock_settings.usage_log_path = str(tmp_path / "usage-log.jsonl")
        mock_settings.retrieval_hybrid = False
        mock_settings.cors_allow_origins = "*"
        mock_settings.raw_data_path = str(tmp_path / "raw")
        mock_settings.documents_sync_state_path = str(tmp_path / "documents-sync-state.json")
        mock_settings.news_faiss_index_path = str(tmp_path / "news-index")
        mock_settings.news_sync_interval_seconds = 21600
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
                    "citation_id": "C1",
                    "source": "thesis_rules.pdf",
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
            assert data["request_id"]
            assert data["model_used"] == "mock-model"
            assert len(data["sources"]) == 1
            assert (
                data["reasoning"]
                == "The context from Thesis Rules p.3 states the deadline."
            )
            assert data["confidence"] == "high"
            assert len(data["cited_sources"]) == 1
            assert data["cited_sources"][0]["citation_id"] == "C1"
            assert data["cited_sources"][0]["source"] == "thesis_rules.pdf"
            from app.main import settings as app_settings

            usage_path = Path(app_settings.usage_log_path)
            assert usage_path.exists()
            lines = usage_path.read_text(encoding="utf-8").strip().splitlines()
            assert lines
            entry = json.loads(lines[-1])
            assert entry["request_id"] == data["request_id"]
            assert entry["status"] == "ok"
            assert entry["query_text"] == "When is the thesis deadline?"
            assert entry["cited_sources_count"] == 1
            assert entry["source_types"]["pdf"] == 1
            assert entry["source_types"]["news"] == 0

    def test_ask_no_db(self, client_no_db):
        resp = client_no_db.post("/ask", json={"query": "test"})
        assert resp.status_code == 503

    def test_ask_logs_error_on_generation_failure(self, client):
        with patch("app.main.rag_ask", new_callable=AsyncMock, side_effect=Exception("boom")):
            resp = client.post("/ask", json={"query": "Will fail?"})
            assert resp.status_code == 502

        from app.main import settings as app_settings

        usage_path = Path(app_settings.usage_log_path)
        assert usage_path.exists()
        lines = usage_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        entry = json.loads(lines[-1])
        assert entry["status"] == "error"
        assert entry["query_text"] == "Will fail?"
        assert entry["answer_length_chars"] == 0
        assert entry["cited_sources_count"] == 0

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
            assert "request_id" in data
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

    def test_get_source_file(self, client, tmp_path):
        upload_dir = tmp_path / "raw"
        upload_dir.mkdir()
        target = upload_dir / "demo.pdf"
        target.write_bytes(b"%PDF-1.4 demo")
        with patch("app.main.settings.raw_data_path", str(upload_dir)):
            resp = client.get("/files/demo.pdf")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert "inline" in resp.headers.get("content-disposition", "").lower()

    def test_get_source_file_docx(self, client, tmp_path):
        upload_dir = tmp_path / "raw"
        upload_dir.mkdir()
        target = upload_dir / "demo.docx"
        target.write_bytes(b"PK\x03\x04 demo")
        with patch("app.main.settings.raw_data_path", str(upload_dir)):
            resp = client.get("/files/demo.docx")
        assert resp.status_code == 200
        assert (
            resp.headers["content-type"]
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert "inline" in resp.headers.get("content-disposition", "").lower()

    def test_get_source_file_not_found(self, client, tmp_path):
        upload_dir = tmp_path / "raw"
        upload_dir.mkdir()
        with patch("app.main.settings.raw_data_path", str(upload_dir)):
            resp = client.get("/files/missing.pdf")
        assert resp.status_code == 404


class TestBackgroundJobs:
    def test_trigger_reindex(self, client):
        with patch("app.main._run_reindex_job") as mock_job:
            resp = client.post("/admin/reindex")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"
        mock_job.assert_called_once()

    def test_trigger_documents_sync(self, client):
        with patch("app.main._run_documents_sync_job") as mock_job:
            resp = client.post("/admin/documents/sync")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"
        mock_job.assert_called_once()

    def test_old_scrape_route_removed(self, client):
        resp = client.post("/admin/scrape")
        assert resp.status_code == 404

    def test_get_documents_sync_status(self, client):
        resp = client.get("/admin/documents/sync")
        assert resp.status_code == 200
        assert "status" in resp.json()

    def test_trigger_news_bootstrap(self, client):
        with patch("app.main._run_news_job") as mock_job:
            resp = client.post("/admin/news/bootstrap")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "queued"
        assert payload["mode"] == "bootstrap"
        mock_job.assert_called_once_with("bootstrap")

    def test_trigger_news_sync(self, client):
        with patch("app.main._run_news_job") as mock_job:
            resp = client.post("/admin/news/sync")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "queued"
        assert payload["mode"] == "sync"
        mock_job.assert_called_once_with("sync")

    def test_get_news_status(self, client):
        resp = client.get("/admin/news")
        assert resp.status_code == 200
        payload = resp.json()
        assert "status" in payload
        assert "processed_count" in payload
        assert "embedded_count" in payload


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


class TestUsageEndpoints:
    def test_get_usage_entries_returns_newest_first_and_skips_bad_lines(self, client):
        from app.main import settings as app_settings

        usage_path = Path(app_settings.usage_log_path)
        usage_path.parent.mkdir(parents=True, exist_ok=True)

        first = {
            "timestamp_utc": "2026-03-20T10:00:00+00:00",
            "query_text": "First question",
            "answer_length_chars": 10,
            "confidence": "low",
            "model_used": "m1",
            "reranker_model": "r1",
            "latency_ms": 101.5,
            "cited_sources_count": 0,
            "source_types": {"pdf": 0, "news": 0},
            "status": "ok",
        }
        second = {
            "timestamp_utc": "2026-03-21T10:00:00+00:00",
            "query_text": "Second question",
            "answer_length_chars": 20,
            "confidence": "high",
            "model_used": "m2",
            "reranker_model": "r2",
            "latency_ms": 88.4,
            "cited_sources_count": 1,
            "source_types": {"pdf": 1, "news": 0},
            "status": "ok",
        }

        usage_path.write_text(
            "\n".join(
                [
                    json.dumps(first, ensure_ascii=True),
                    "{ this is malformed json",
                    json.dumps(second, ensure_ascii=True),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        resp = client.get("/admin/usage?limit=2")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["count"] == 2
        assert payload["entries"][0]["query_text"] == "Second question"
        assert payload["entries"][1]["query_text"] == "First question"

    def test_get_usage_stats_respects_window(self, client):
        from app.main import settings as app_settings

        usage_path = Path(app_settings.usage_log_path)
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        inside_window = (now - timedelta(days=2)).isoformat()
        outside_window = (now - timedelta(days=20)).isoformat()

        usage_entries = [
            {
                "timestamp_utc": inside_window,
                "query_text": "Q1",
                "answer_length_chars": 100,
                "confidence": "high",
                "model_used": "m",
                "reranker_model": "r",
                "latency_ms": 100.0,
                "cited_sources_count": 1,
                "source_types": {"pdf": 1, "news": 0},
                "status": "ok",
            },
            {
                "timestamp_utc": inside_window,
                "query_text": "Q2",
                "answer_length_chars": 0,
                "confidence": "medium",
                "model_used": "m",
                "reranker_model": "r",
                "latency_ms": 200.0,
                "cited_sources_count": 0,
                "source_types": {"pdf": 0, "news": 1},
                "status": "error",
            },
            {
                "timestamp_utc": outside_window,
                "query_text": "Old",
                "answer_length_chars": 100,
                "confidence": "high",
                "model_used": "m",
                "reranker_model": "r",
                "latency_ms": 999.0,
                "cited_sources_count": 1,
                "source_types": {"pdf": 1, "news": 1},
                "status": "ok",
            },
        ]

        usage_path.write_text(
            "\n".join(json.dumps(entry, ensure_ascii=True) for entry in usage_entries) + "\n",
            encoding="utf-8",
        )

        resp = client.get("/admin/usage/stats?window_days=7")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["total_queries"] == 2
        assert payload["avg_latency_ms"] == 150.0
        assert payload["citation_presence_rate"] == 0.5
        assert payload["non_empty_answer_rate"] == 0.5
        assert payload["confidence_distribution"]["high"] == 1
        assert payload["confidence_distribution"]["medium"] == 1
        assert payload["source_mix_pdf_vs_news"]["pdf"] == 1
        assert payload["source_mix_pdf_vs_news"]["news"] == 1


class TestFeedbackEndpoint:
    def test_feedback_updates_usage_history(self, client):
        mock_result = RAGResult(
            answer="Answer text",
            sources=[],
            model_used="mock-model",
            reasoning="reason",
            confidence="high",
            cited_sources=[],
        )
        with patch("app.main.rag_ask", new_callable=AsyncMock, return_value=mock_result):
            ask_resp = client.post("/ask", json={"query": "Feedback me"})
        assert ask_resp.status_code == 200
        request_id = ask_resp.json()["request_id"]

        feedback_resp = client.post(
            "/feedback",
            json={"request_id": request_id, "helpful": True},
        )
        assert feedback_resp.status_code == 200
        payload = feedback_resp.json()
        assert payload["status"] == "updated"
        assert payload["request_id"] == request_id
        assert payload["helpful"] is True

        from app.main import settings as app_settings

        usage_path = Path(app_settings.usage_log_path)
        lines = usage_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        entry = json.loads(lines[-1])
        assert entry["request_id"] == request_id
        assert entry["feedback"] is True
        assert entry["feedback_timestamp_utc"]

    def test_feedback_returns_404_when_request_id_is_unknown(self, client):
        resp = client.post(
            "/feedback",
            json={"request_id": "missing-id", "helpful": False},
        )
        assert resp.status_code == 404
