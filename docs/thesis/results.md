# Results Draft

## Milestone 3 Implementation Outcomes
Implemented components:
- Unified frontend (Vite + React) with `/chat` and `/admin` routes.
- Chat UI for text queries with displayed confidence, reasoning, and citations.
- Admin panel for model switching, system prompt updates, document management, and separate scrape/reindex controls.
- Targeted scraper extension for normalized news ingestion.
- End-to-end citation metadata path improvements in ingestion.
- Frontend containerization and Compose integration.

## Functional Verification
Verification performed via automated tests and manual checks:
- Backend tests for scraping, ingestion metadata handling, and API routes.
- Frontend tests for chat rendering and admin control flows.
- Manual smoke target: `docker compose up --build` and interaction checks in browser.

## Observed Strengths
- Hybrid retrieval + reranking improves context relevance.
- Separation of scrape and reindex gives controllable operations for admins.
- Structured responses improve observability for evaluation and debugging.

## Known Limitations
- Admin authentication is intentionally deferred.
- Faculty-specific vector-store partitioning is deferred.
- Citation page numbers depend on document provenance availability and full reindexing.

## Next Steps
- Run full reindex after ingestion changes on the production document set.
- Execute evaluation question set and record quantitative metrics.
- Refine prompt and retrieval parameters based on measured errors.
