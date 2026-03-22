from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.rag_chain import (
    _format_docs,
    _build_cited_sources,
    _build_context_items,
    _replace_inline_chunk_citations,
    _rerank,
    RAG_PROMPT,
    get_llm,
    RAGResult,
    RAGOutput,
)


class TestFormatDocs:
    def test_formats_with_metadata(self):
        docs = [
            Document(
                page_content="Thesis must be submitted by April.",
                metadata={"title": "Thesis Rules", "page": 3},
            ),
            Document(
                page_content="Internship is mandatory.",
                metadata={"source": "internship.pdf", "page": 1},
            ),
        ]
        formatted = _format_docs(docs)
        assert "[C1 | Thesis Rules, p. 3]" in formatted
        assert "Thesis must be submitted" in formatted
        assert "Internship is mandatory" in formatted

    def test_empty_docs(self):
        assert _format_docs([]) == ""


class TestPromptTemplate:
    def test_prompt_has_expected_variables(self):
        assert "context" in RAG_PROMPT.input_variables or any(
            "context" in str(m) for m in RAG_PROMPT.messages
        )
        assert "question" in RAG_PROMPT.input_variables or any(
            "question" in str(m) for m in RAG_PROMPT.messages
        )

    def test_prompt_has_few_shot_examples(self):
        """The prompt should contain few-shot examples."""
        messages = RAG_PROMPT.messages
        assert len(messages) >= 5


class TestGetLLM:
    @patch("app.rag_chain.settings")
    def test_openrouter_provider(self, mock_settings):
        mock_settings.llm_provider = "openrouter"
        mock_settings.openrouter_model = "openai/gpt-4o"
        mock_settings.openrouter_api_key = "test-key"
        llm = get_llm()
        assert llm is not None

    @patch("app.rag_chain.settings")
    def test_openrouter_with_model_override(self, mock_settings):
        mock_settings.llm_provider = "openrouter"
        mock_settings.openrouter_model = "openai/gpt-4o"
        mock_settings.openrouter_api_key = "test-key"
        llm = get_llm(model_override="openai/gpt-4o-mini")
        assert llm.model_name == "openai/gpt-4o-mini"

    @patch("app.rag_chain.settings")
    def test_ollama_with_model_override(self, mock_settings):
        mock_settings.llm_provider = "ollama"
        mock_settings.ollama_model = "llama3.1"
        mock_settings.ollama_base_url = "http://localhost:11434"
        llm = get_llm(model_override="mistral")
        assert llm.model == "mistral"

    @patch("app.rag_chain.settings")
    def test_invalid_provider_raises(self, mock_settings):
        mock_settings.llm_provider = "invalid"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm()


class TestRAGResult:
    def test_defaults(self):
        r = RAGResult(answer="Hello")
        assert r.answer == "Hello"
        assert r.sources == []
        assert r.model_used == ""
        assert r.reasoning == ""
        assert r.confidence == ""
        assert r.cited_sources == []

    def test_with_structured_fields(self):
        r = RAGResult(
            answer="Test",
            sources=[],
            model_used="gpt-4o",
            reasoning="The context says...",
            confidence="high",
            cited_sources=[
                {"document": "Rules", "page": 1, "relevant_snippet": "snippet"}
            ],
        )
        assert r.confidence == "high"
        assert r.reasoning == "The context says..."
        assert len(r.cited_sources) == 1


class TestRAGOutput:
    def test_rag_output_model(self):
        output = RAGOutput(
            reasoning="I found the answer in context.",
            answer="The deadline is April 15th. [C1]",
            cited_chunk_ids=["C1"],
            confidence="high",
        )
        assert output.confidence == "high"
        assert output.cited_chunk_ids == ["C1"]

    def test_rag_output_serialization(self):
        output = RAGOutput(
            reasoning="analysis",
            answer="answer text",
            cited_chunk_ids=[],
            confidence="low",
        )
        data = output.model_dump()
        assert data["confidence"] == "low"
        assert data["cited_chunk_ids"] == []


class TestCitationGrounding:
    def test_replace_inline_chunk_citations(self):
        docs = [
            Document(
                page_content="Thesis must be submitted by April.",
                metadata={"title": "Thesis Rules", "page": 3},
            )
        ]
        context_items = _build_context_items(docs)
        citation_map = {item["citation_id"]: item for item in context_items}
        answer = _replace_inline_chunk_citations("Deadline is April 15th. [C1]", citation_map)
        assert "[Thesis Rules, p. 3]" in answer

    def test_build_cited_sources_falls_back_to_context(self):
        docs = [
            Document(
                page_content="Students must pass the final examination.",
                metadata={"title": "Exam Rules", "page": 2},
            )
        ]
        context_items = _build_context_items(docs)
        cited_sources = _build_cited_sources([], context_items)
        assert len(cited_sources) == 1
        assert cited_sources[0]["document"] == "Exam Rules"
        assert cited_sources[0]["page"] == 2


class TestRerank:
    @pytest.mark.asyncio
    @patch("app.rag_chain.settings")
    @patch("app.rag_chain.get_llm")
    async def test_rerank_sorts_by_score(self, mock_get_llm, mock_settings):
        """Verify _rerank parses scores and reorders documents."""
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        mock_settings.retrieval_use_reranker = True
        mock_settings.reranker_model = "openai/gpt-4o-mini"

        async def fake_reranker(_input):
            return AIMessage(content="[0.3, 0.9, 0.1]")

        mock_get_llm.return_value = RunnableLambda(fake_reranker)

        docs = [
            Document(page_content="Doc A", metadata={"title": "A"}),
            Document(page_content="Doc B", metadata={"title": "B"}),
            Document(page_content="Doc C", metadata={"title": "C"}),
        ]

        result = await _rerank("test query", docs, top_k=2)
        assert len(result) == 2
        # Doc B should be first (score 0.9), then Doc A (score 0.3)
        assert result[0].page_content == "Doc B"
        assert result[1].page_content == "Doc A"

    @pytest.mark.asyncio
    @patch("app.rag_chain.settings")
    async def test_rerank_disabled(self, mock_settings):
        """When reranker is disabled, documents are returned as-is."""
        mock_settings.retrieval_use_reranker = False

        docs = [
            Document(page_content="Doc A", metadata={}),
            Document(page_content="Doc B", metadata={}),
            Document(page_content="Doc C", metadata={}),
        ]

        result = await _rerank("query", docs, top_k=2)
        assert len(result) == 2
        assert result[0].page_content == "Doc A"

    @pytest.mark.asyncio
    @patch("app.rag_chain.settings")
    async def test_rerank_empty_docs(self, mock_settings):
        """Reranking empty list should return empty list."""
        mock_settings.retrieval_use_reranker = True
        result = await _rerank("query", [], top_k=5)
        assert result == []

    @pytest.mark.asyncio
    @patch("app.rag_chain.settings")
    @patch("app.rag_chain.get_llm")
    async def test_rerank_fallback_on_error(self, mock_get_llm, mock_settings):
        """If the reranker LLM fails, docs are returned as-is (truncated to top_k)."""
        from langchain_core.runnables import RunnableLambda

        mock_settings.retrieval_use_reranker = True
        mock_settings.reranker_model = "openai/gpt-4o-mini"

        async def failing_reranker(_input):
            raise Exception("LLM error")

        mock_get_llm.return_value = RunnableLambda(failing_reranker)

        docs = [Document(page_content=f"Doc {i}", metadata={}) for i in range(5)]
        result = await _rerank("query", docs, top_k=3)
        assert len(result) == 3
