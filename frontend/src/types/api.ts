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
  source_type?: "pdf" | "news";
  published_at?: string | null;
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

export interface NewsJobStatusResponse {
  status: string;
  error?: string | null;
  mode?: "bootstrap" | "sync" | null;
  pages?: number | null;
  processed_count?: number | null;
  added_count?: number | null;
  updated_count?: number | null;
  unchanged_count?: number | null;
  embedded_count?: number | null;
  last_run_at?: string | null;
}
