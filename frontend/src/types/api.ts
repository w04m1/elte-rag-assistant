export type ChatRole = "user" | "assistant";

export interface SourceItem {
  content: string;
  document: string;
  page: number | null;
}

export interface CitedSourceItem {
  citation_id: string;
  source: string;
  document: string;
  page: number | null;
  relevant_snippet: string;
}

export interface AskResponse {
  answer: string;
  sources: SourceItem[];
  model_used: string;
  reasoning: string;
  confidence: string;
  cited_sources: CitedSourceItem[];
}

export interface RuntimeSettings {
  generator_model: string;
  reranker_model: string;
  system_prompt: string;
  embedding_provider: string;
  embedding_model: string;
}

export interface RuntimeSettingsUpdate {
  generator_model?: string;
  reranker_model?: string;
  system_prompt?: string;
}

export interface DocumentListItem {
  source: string;
  title: string;
  chunk_count: number;
}

export interface DocumentsResponse {
  documents: DocumentListItem[];
  count: number;
}

export interface JobStatusResponse {
  status: string;
  error?: string | null;
  vector_count?: number | null;
  result?: Record<string, unknown> | null;
}
