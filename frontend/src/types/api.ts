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
  request_id?: string | null;
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

export interface UsageSourceTypes {
  pdf: number;
  news: number;
}

export interface UsageLogEntry {
  request_id?: string;
  timestamp_utc: string;
  query_text: string;
  answer_length_chars: number;
  confidence: string;
  model_used: string;
  reranker_model: string;
  latency_ms: number;
  cited_sources_count: number;
  source_types: UsageSourceTypes;
  feedback?: boolean | null;
  feedback_timestamp_utc?: string | null;
  status: "ok" | "error";
}

export interface UsageLogResponse {
  entries: UsageLogEntry[];
  count: number;
  limit: number;
}

export interface UsageStatsResponse {
  window_days: number;
  generated_at_utc: string;
  total_queries: number;
  avg_latency_ms: number;
  citation_presence_rate: number;
  non_empty_answer_rate: number;
  confidence_distribution: Record<string, number>;
  source_mix_pdf_vs_news: UsageSourceTypes;
  helpful_feedback_count?: number;
  unhelpful_feedback_count?: number;
  helpful_feedback_rate?: number;
  feedback_coverage_rate?: number;
}

export interface FeedbackResponse {
  status: "updated";
  request_id: string;
  helpful: boolean;
}
