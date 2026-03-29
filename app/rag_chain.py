import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field as PydanticField, field_validator

from app.config import settings

logger = logging.getLogger(__name__)


class CitedSource(BaseModel):
    """WIP: A single cited source from the context."""

    source: str
    document: str
    page: int | None = None
    relevant_snippet: str
    source_type: Literal["pdf", "news"] = "pdf"
    published_at: str | None = None

    @field_validator("page", mode="before")
    @classmethod
    def _coerce_page(cls, v: Any) -> int | None:
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None


class RAGOutput(BaseModel):
    reasoning: str = PydanticField(
        description="Internal chain-of-thought explaining how the context was interpreted",
    )
    answer: str = PydanticField(
        description="User-facing answer with inline citations in [Document Title, p. X] format",
    )
    cited_chunk_ids: list[str] = PydanticField(
        default_factory=list,
        description="Structured list of cited chunk IDs like C1, C2 derived from the provided context",
    )
    confidence: Literal["high", "medium", "low"] = PydanticField(
        description="Confidence level based on how well the context answers the question",
    )


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful FAQ assistant for ELTE University.
Answer the student's question **based only on the provided context**.
If the context does not contain enough information, say "I don't have enough information to answer that question."

Rules:
- Be concise and accurate.
- Each context block has a stable citation ID such as [C1].
- When you cite claims inline, cite the chunk IDs like [C1] or [C1][C2].
- Do not make up information that is not in the context.
- Before answering, reason step-by-step about which parts of the context are relevant.
- Assess your confidence level (high, medium, or low) based on how well the context answers the question.
- Return cited_chunk_ids using only the chunk IDs that support the answer.
"""

def build_rag_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Context:\n[C1 | Thesis Rules, p. 3]\nStudents must submit their thesis by "
                "April 15th of the final semester. Late submissions will not be accepted "
                "without prior approval from the department head.\n\n---\n"
                "Question: When is the thesis submission deadline?",
            ),
            (
                "assistant",
                "The thesis submission deadline is April 15th of the final semester. "
                "Late submissions are not accepted without prior approval from the "
                "department head. [C1]",
            ),
            (
                "human",
                "Context:\n[C1 | Enrolment Guide, p. 5]\nStudents can enroll in elective "
                "courses through the Neptun system during the registration period."
                "\n\n---\n"
                "Question: What is the tuition fee for international students?",
            ),
            (
                "assistant",
                "I don't have enough information to answer that question. The available "
                "context covers course enrollment procedures but does not mention tuition "
                "fees for international students.",
            ),
            (
                "human",
                "Conversation history:\n{chat_history}\n\n"
                "Context:\n{context}\n\n---\nQuestion: {question}",
            ),
        ]
    )


RAG_PROMPT = build_rag_prompt(DEFAULT_SYSTEM_PROMPT)

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a relevance scoring system. Given a query and a list of text "
            "chunks, score each chunk's relevance to the query on a scale of 0.0 to "
            "1.0. Return ONLY a JSON array of numbers representing the scores in the "
            "same order as the chunks. Example: [0.9, 0.3, 0.7]",
        ),
        ("human", "Query: {query}\n\nChunks:\n{chunks}"),
    ]
)

FOLLOW_UP_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite follow-up questions into standalone retrieval queries. "
            "Use the recent conversation only to resolve references. "
            "Do not answer the question. Return only the rewritten query text.",
        ),
        (
            "human",
            "Conversation:\n{chat_history}\n\n"
            "Follow-up question: {question}\n\n"
            "Standalone retrieval query:",
        ),
    ]
)


def _build_context_items(docs: list[Document]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        source = str(meta.get("source", "unknown"))
        raw_source_type = str(meta.get("source_type", "")).strip().lower()
        source_type: Literal["pdf", "news"]
        if raw_source_type in {"pdf", "news"}:
            source_type = raw_source_type  # type: ignore[assignment]
        elif source.startswith("http://") or source.startswith("https://"):
            source_type = "news"
        else:
            source_type = "pdf"
        source = meta.get("title", meta.get("source", "unknown"))
        page = meta.get("page", "?")
        items.append(
            {
                "citation_id": f"C{index}",
                "document": source,
                "page": page if page != "?" else None,
                "content": doc.page_content,
                "snippet": doc.page_content[:300],
                "source": str(meta.get("source", "unknown")),
                "source_type": source_type,
                "published_at": (
                    str(meta.get("published_at"))
                    if meta.get("published_at") is not None
                    else None
                ),
            }
        )
    return items


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string with stable IDs."""
    parts: list[str] = []
    for item in _build_context_items(docs):
        page = item["page"] if item["page"] is not None else "?"
        parts.append(
            f"[{item['citation_id']} | {item['document']}, p. {page}]\n{item['content']}"
        )
    return "\n\n".join(parts)


def _extract_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Build the sources list from retrieved documents."""
    return _build_context_items(docs)


def _normalize_citation_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _coerce_page_to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _doc_dedupe_key(doc: Document) -> tuple[str, str, str]:
    source = _normalize_citation_text(str(doc.metadata.get("source", "unknown")))
    page = _coerce_page_to_int(doc.metadata.get("page"))
    snippet = str(
        doc.metadata.get("relevant_snippet")
        or doc.metadata.get("snippet")
        or doc.page_content[:300]
    )
    return (
        source,
        str(page) if page is not None else "",
        _normalize_citation_text(snippet[:300]),
    )


def _dedupe_docs_by_source_page_snippet(docs: list[Document]) -> list[Document]:
    deduped: list[Document] = []
    seen: set[tuple[str, str, str]] = set()

    for doc in docs:
        key = _doc_dedupe_key(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    return deduped


def _citation_marker(citation_id: str) -> str:
    return citation_id[1:] if citation_id.startswith("C") else citation_id


def _replace_inline_chunk_citations(answer: str, citation_map: dict[str, dict[str, Any]]) -> str:
    id_map: dict[str, dict[str, Any]] = {
        citation_id.upper(): item for citation_id, item in citation_map.items()
    }
    doc_page_map: dict[tuple[str, int], str] = {}
    doc_only_map: dict[str, set[str]] = {}

    for citation_id, item in id_map.items():
        document = str(item.get("document", "")).strip()
        if not document:
            continue
        doc_norm = _normalize_citation_text(document)
        doc_only_map.setdefault(doc_norm, set()).add(citation_id)

        page = item.get("page")
        try:
            if page is not None:
                doc_page_map[(doc_norm, int(page))] = citation_id
        except (TypeError, ValueError):
            continue

    def replace_chunk_id(match: re.Match[str]) -> str:
        citation_id = match.group(1).upper()
        if id_map.get(citation_id) is None:
            return match.group(0)
        return f"[{_citation_marker(citation_id)}](cite:{citation_id})"

    with_chunk_ids_replaced = re.sub(r"\[([Cc]\d+)\]", replace_chunk_id, answer)

    def replace_document_reference(match: re.Match[str]) -> str:
        raw_reference = match.group(1).strip()
        parsed = re.match(
            r"^(?P<document>.+?),\s*p\.?\s*(?P<page>\d+)\s*$",
            raw_reference,
            flags=re.IGNORECASE,
        )
        if parsed:
            document_norm = _normalize_citation_text(parsed.group("document"))
            page = int(parsed.group("page"))
            citation_id = doc_page_map.get((document_norm, page))
            if citation_id:
                return f"[{_citation_marker(citation_id)}](cite:{citation_id})"

        doc_only_ids = doc_only_map.get(_normalize_citation_text(raw_reference))
        if doc_only_ids and len(doc_only_ids) == 1:
            citation_id = next(iter(doc_only_ids))
            return f"[{_citation_marker(citation_id)}](cite:{citation_id})"

        return match.group(0)

    # Replace bare bracketed references like [ELTE Rules, p. 83] but keep existing markdown links intact.
    return re.sub(
        r"\[([^\[\]\n]+)\](?!\()",
        replace_document_reference,
        with_chunk_ids_replaced,
    )


def _build_cited_sources(
    cited_chunk_ids: list[str],
    context_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {item["citation_id"]: item for item in context_items}
    cited_sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_id in cited_chunk_ids:
        citation_id = raw_id.upper()
        item = by_id.get(citation_id)
        if item is None or citation_id in seen:
            continue
        seen.add(citation_id)
        cited_sources.append(
            {
                "citation_id": citation_id,
                "source": item["source"],
                "document": item["document"],
                "page": item["page"],
                "relevant_snippet": item["snippet"],
                "source_type": item["source_type"],
                "published_at": item["published_at"],
            }
        )

    if cited_sources:
        return cited_sources

    return [
        {
            "citation_id": item["citation_id"],
            "source": item["source"],
            "document": item["document"],
            "page": item["page"],
            "relevant_snippet": item["snippet"],
            "source_type": item["source_type"],
            "published_at": item["published_at"],
        }
        for item in context_items[: min(3, len(context_items))]
    ]


def _is_response_format_compatibility_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "response_format" not in message:
        return False
    return any(
        marker in message
        for marker in (
            "must be text or json_object",
            "invalid schema for response_format",
            "not supported",
            "json_schema",
            "response_format is invalid",
        )
    )


@dataclass
class RAGResult:
    """Structured response returned by the RAG chain."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    model_used: str = ""
    reasoning: str = ""
    confidence: str = ""
    cited_sources: list[dict[str, Any]] = field(default_factory=list)


def get_llm(model_override: str | None = None):
    """Return a LangChain chat model based on the configured provider."""
    if settings.llm_provider == "openrouter":
        from langchain_openai import ChatOpenAI

        model = model_override or settings.openrouter_model
        return ChatOpenAI(
            model=model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.2,
        )
    elif settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        model = model_override or settings.ollama_model
        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


async def _rerank(
    query: str,
    docs: list[Document],
    top_k: int,
    reranker_model: str | None = None,
) -> list[Document]:
    """Rerank docs using an LLM-based relevance scorer."""
    if not settings.retrieval_use_reranker or not docs:
        return docs[:top_k]

    logger.info(
        "Reranking %d documents to top %d using %s",
        len(docs),
        top_k,
        reranker_model or settings.reranker_model,
    )

    reranker_llm = get_llm(model_override=reranker_model or settings.reranker_model)

    chunks_text = "\n\n".join(
        f"[Chunk {i + 1}]: {doc.page_content[:500]}" for i, doc in enumerate(docs)
    )

    try:
        chain = RERANK_PROMPT | reranker_llm | StrOutputParser()
        result = await chain.ainvoke({"query": query, "chunks": chunks_text})

        # Strip markdown code fences if present
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        scores: list[float] = json.loads(result)

        if not isinstance(scores, list) or len(scores) != len(docs):
            logger.warning(
                "Reranker returned invalid scores (expected %d, got %s), "
                "skipping reranking",
                len(docs),
                len(scores) if isinstance(scores, list) else "non-list",
            )
            return docs[:top_k]

        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, _score in scored_docs[:top_k]]

        logger.info(
            "Reranking complete. Top scores: %s",
            [f"{s:.2f}" for _, s in scored_docs[:top_k]],
        )
        return reranked

    except Exception as exc:
        logger.warning(
            f"Reranking failed ({exc}), returning documents as-is",
        )
        return docs[:top_k]


def _reciprocal_rank_fusion(
    results_lists: list[list[Document]],
    weights: list[float],
    k_rrf: int = 60,
) -> list[Document]:
    """Merge multiple ranked lists using weighted Reciprocal Rank Fusion.

    Each document gets a score of weight / (k_rrf + rank) from each list,
    summed across lists. Returns documents sorted by fused score descending.
    """
    fused_scores: dict[int, float] = {}  # id(doc) → score
    doc_map: dict[int, Document] = {}

    for doc_list, weight in zip(results_lists, weights):
        for rank, doc in enumerate(doc_list):
            doc_id = id(doc)
            doc_map[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + weight / (
                k_rrf + rank + 1
            )

    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [doc_map[did] for did in sorted_ids]


def _normalize_chat_history(
    chat_history: list[dict[str, Any]] | None,
    *,
    max_turns: int = 12,
    max_chars_per_turn: int = 500,
) -> list[dict[str, Any]]:
    if not chat_history:
        return []

    normalized: list[dict[str, Any]] = []
    for raw_turn in chat_history[-max_turns:]:
        role = str(raw_turn.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        text = str(raw_turn.get("text", "")).strip()
        if not text:
            continue
        normalized_turn: dict[str, Any] = {
            "role": role,
            "text": text[:max_chars_per_turn],
        }
        if role == "assistant":
            raw_cited_sources = raw_turn.get("cited_sources")
            if isinstance(raw_cited_sources, list):
                cited_sources: list[dict[str, Any]] = []
                for raw_source in raw_cited_sources:
                    if not isinstance(raw_source, dict):
                        continue
                    source = str(raw_source.get("source", "")).strip()
                    document = str(raw_source.get("document", "")).strip()
                    snippet = str(raw_source.get("relevant_snippet", "")).strip()
                    if not source or not document or not snippet:
                        continue
                    source_type = (
                        "news"
                        if str(raw_source.get("source_type", "pdf")).strip().lower()
                        == "news"
                        else "pdf"
                    )
                    cited_sources.append(
                        {
                            "citation_id": str(raw_source.get("citation_id", ""))
                            .strip()
                            .upper(),
                            "source": source,
                            "document": document,
                            "page": _coerce_page_to_int(raw_source.get("page")),
                            "relevant_snippet": snippet[:300],
                            "source_type": source_type,
                            "published_at": (
                                str(raw_source.get("published_at"))
                                if raw_source.get("published_at") is not None
                                else None
                            ),
                        }
                    )
                if cited_sources:
                    normalized_turn["cited_sources"] = cited_sources

        normalized.append(normalized_turn)
    return normalized


def _format_chat_history_for_prompt(chat_history: list[dict[str, Any]]) -> str:
    if not chat_history:
        return "(none)"

    lines: list[str] = []
    for turn in chat_history:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def _is_likely_follow_up(query: str, chat_history: list[dict[str, Any]]) -> bool:
    if not chat_history:
        return False

    normalized_query = query.strip().lower()
    if not normalized_query:
        return False

    query_word_count = len(re.findall(r"\b\w+\b", normalized_query))
    is_short_query = query_word_count <= 8

    referential_terms = {"it", "this", "that", "they", "those", "these"}
    has_referential_pronoun = any(
        re.search(rf"\b{term}\b", normalized_query) for term in referential_terms
    )
    has_referential_phrase = any(
        phrase in normalized_query
        for phrase in ("what about", "is it", "is this", "is that")
    )
    has_referential_and = (
        normalized_query == "and"
        or normalized_query.startswith("and ")
        or normalized_query.startswith("and,")
    )

    return (
        is_short_query
        or has_referential_pronoun
        or has_referential_phrase
        or has_referential_and
    )


def _normalize_rewritten_query(candidate: str) -> str:
    normalized = candidate.strip()
    if normalized.startswith("```"):
        normalized = normalized.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    normalized = normalized.strip().strip('"').strip("'").strip()
    lowered = normalized.lower()
    for prefix in (
        "standalone retrieval query:",
        "standalone query:",
        "rewritten query:",
        "query:",
    ):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            lowered = normalized.lower()
    return normalized


async def _rewrite_follow_up_query(
    query: str,
    chat_history: list[dict[str, Any]],
    llm,
) -> str:
    recent_history = chat_history[-6:]
    chain = FOLLOW_UP_REWRITE_PROMPT | llm | StrOutputParser()
    try:
        rewritten = await chain.ainvoke(
            {
                "chat_history": _format_chat_history_for_prompt(recent_history),
                "question": query,
            }
        )
    except Exception as exc:
        logger.warning("Follow-up rewrite failed (%s), using original query", exc)
        return query

    normalized = _normalize_rewritten_query(str(rewritten))
    if not normalized or normalized.lower() in {"none", "n/a", "null"}:
        return query
    return normalized


def _extract_carry_over_docs(
    chat_history: list[dict[str, Any]],
    *,
    max_sources: int = 2,
) -> list[Document]:
    for turn in reversed(chat_history):
        if turn.get("role") != "assistant":
            continue
        raw_cited_sources = turn.get("cited_sources")
        if not isinstance(raw_cited_sources, list) or not raw_cited_sources:
            continue

        docs: list[Document] = []
        for raw_source in raw_cited_sources[:max_sources]:
            if not isinstance(raw_source, dict):
                continue
            source = str(raw_source.get("source", "")).strip()
            snippet = str(raw_source.get("relevant_snippet", "")).strip()
            if not source or not snippet:
                continue
            document = str(raw_source.get("document", "")).strip() or source
            metadata = {
                "source": source,
                "title": document,
                "page": _coerce_page_to_int(raw_source.get("page")),
                "source_type": (
                    "news"
                    if str(raw_source.get("source_type", "pdf")).strip().lower()
                    == "news"
                    else "pdf"
                ),
                "published_at": (
                    str(raw_source.get("published_at"))
                    if raw_source.get("published_at") is not None
                    else None
                ),
                "relevant_snippet": snippet[:300],
                "snippet": snippet[:300],
                "carry_over": True,
            }
            docs.append(Document(page_content=snippet, metadata=metadata))
        return docs

    return []


async def ask(
    query: str,
    db: FAISS,
    bm25_retriever=None,
    news_db: FAISS | None = None,
    *,
    chat_history: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    generator_model: str | None = None,
    reranker_model: str | None = None,
) -> RAGResult:
    """Run the full RAG pipeline and return a structured result."""
    k = settings.retrieval_k
    fetch_k = settings.retrieval_fetch_k
    normalized_chat_history = _normalize_chat_history(chat_history)
    llm = get_llm(model_override=generator_model)
    retrieval_query = query
    rewrite_applied = False

    if _is_likely_follow_up(query, normalized_chat_history):
        rewritten_query = await _rewrite_follow_up_query(
            query, normalized_chat_history, llm
        )
        if rewritten_query != query:
            rewrite_applied = True
            retrieval_query = rewritten_query
            logger.info("Follow-up rewrite applied for retrieval: %s", retrieval_query)

    # Retrieval from the primary PDF/document store
    if settings.retrieval_hybrid and bm25_retriever is not None:
        faiss_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k, "fetch_k": fetch_k * 2},
        )
        weight = settings.retrieval_hybrid_weight
        faiss_docs = faiss_retriever.invoke(retrieval_query)
        bm25_docs = bm25_retriever.invoke(retrieval_query)
        docs: list[Document] = _reciprocal_rank_fusion(
            [faiss_docs, bm25_docs],
            weights=[weight, 1 - weight],
        )
        docs = docs[:fetch_k]
    else:
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k, "fetch_k": fetch_k * 2},
        )
        docs = retriever.invoke(retrieval_query)

    # Retrieval from the separate news store and fusion with document candidates.
    news_docs: list[Document] = []
    if news_db is not None:
        news_retriever = news_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k, "fetch_k": fetch_k * 2},
        )
        news_docs = news_retriever.invoke(retrieval_query)

    if news_docs and docs:
        docs = _reciprocal_rank_fusion(
            [docs, news_docs],
            weights=[1.0, 1.0],
        )[:fetch_k]
    elif news_docs:
        docs = news_docs[:fetch_k]

    carry_over_docs = _extract_carry_over_docs(normalized_chat_history, max_sources=2)
    if carry_over_docs:
        docs = _dedupe_docs_by_source_page_snippet([*docs, *carry_over_docs])
    else:
        docs = _dedupe_docs_by_source_page_snippet(docs)

    # Reranking
    rerank_query = retrieval_query if rewrite_applied else query
    docs = await _rerank(
        rerank_query,
        docs,
        top_k=k,
        reranker_model=reranker_model,
    )

    # Generation
    model_name = str(
        getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    )
    context_items = _build_context_items(docs)
    context = _format_docs(docs)
    sources = _extract_sources(docs)
    prompt = build_rag_prompt(system_prompt or DEFAULT_SYSTEM_PROMPT)
    citation_map = {item["citation_id"]: item for item in context_items}

    # Try structured output , fall back to plain string
    try:
        import warnings

        def _build_structured_llm(method: str):
            if method == "json_schema":
                return llm.with_structured_output(RAGOutput)
            return llm.with_structured_output(RAGOutput, method=method)

        async def _invoke_structured(method: str) -> RAGOutput:
            structured_llm = _build_structured_llm(method)
            chain = prompt | structured_llm
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Pydantic serializer warnings",
                    category=UserWarning,
                )
                return await chain.ainvoke(
                    {
                        "chat_history": _format_chat_history_for_prompt(
                            normalized_chat_history
                        ),
                        "context": context,
                        "question": query,
                    }
                )

        try:
            output: RAGOutput = await _invoke_structured("json_schema")
        except Exception as exc:
            if not _is_response_format_compatibility_error(exc):
                raise
            logger.warning(
                "Structured output json_schema unsupported by provider/model (%s); "
                "retrying with function_calling/json_mode",
                exc,
            )
            try:
                output = await _invoke_structured("function_calling")
            except Exception as function_exc:
                logger.warning(
                    "Structured output function_calling failed (%s); "
                    "retrying with json_mode",
                    function_exc,
                )
                output = await _invoke_structured("json_mode")

        cited_sources = _build_cited_sources(output.cited_chunk_ids, context_items)
        return RAGResult(
            answer=_replace_inline_chunk_citations(output.answer, citation_map),
            sources=sources,
            model_used=model_name,
            reasoning=output.reasoning,
            confidence=output.confidence,
            cited_sources=cited_sources,
        )
    except Exception as exc:
        logger.warning(
            f"Structured output failed ({exc}), falling back to plain string",
        )
        plain_chain = prompt | llm | StrOutputParser()
        answer: str = await plain_chain.ainvoke(
            {
                "chat_history": _format_chat_history_for_prompt(normalized_chat_history),
                "context": context,
                "question": query,
            }
        )
        fallback_citations = _build_cited_sources([], context_items)
        return RAGResult(
            answer=_replace_inline_chunk_citations(answer, citation_map),
            sources=sources,
            model_used=model_name,
            cited_sources=fallback_citations,
        )
