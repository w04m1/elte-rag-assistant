import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { AdminPage } from "@/pages/AdminPage";

const getAdminSettingsMock = vi.fn();
const listAdminDocumentsMock = vi.fn();
const getJobStatusMock = vi.fn();
const getNewsJobStatusMock = vi.fn();
const updateAdminSettingsMock = vi.fn();
const triggerJobMock = vi.fn();
const triggerNewsJobMock = vi.fn();
const uploadDocumentMock = vi.fn();
const deleteDocumentMock = vi.fn();
const getUsageLogsMock = vi.fn();
const getUsageStatsMock = vi.fn();

vi.mock("@/lib/api", () => ({
  getAdminSettings: (...args: unknown[]) => getAdminSettingsMock(...args),
  listAdminDocuments: (...args: unknown[]) => listAdminDocumentsMock(...args),
  getJobStatus: (...args: unknown[]) => getJobStatusMock(...args),
  getNewsJobStatus: (...args: unknown[]) => getNewsJobStatusMock(...args),
  updateAdminSettings: (...args: unknown[]) => updateAdminSettingsMock(...args),
  triggerJob: (...args: unknown[]) => triggerJobMock(...args),
  triggerNewsJob: (...args: unknown[]) => triggerNewsJobMock(...args),
  uploadDocument: (...args: unknown[]) => uploadDocumentMock(...args),
  deleteDocument: (...args: unknown[]) => deleteDocumentMock(...args),
  getUsageLogs: (...args: unknown[]) => getUsageLogsMock(...args),
  getUsageStats: (...args: unknown[]) => getUsageStatsMock(...args),
}));

describe("AdminPage", () => {
  beforeEach(() => {
    getAdminSettingsMock.mockReset();
    listAdminDocumentsMock.mockReset();
    getJobStatusMock.mockReset();
    getNewsJobStatusMock.mockReset();
    updateAdminSettingsMock.mockReset();
    triggerJobMock.mockReset();
    triggerNewsJobMock.mockReset();
    uploadDocumentMock.mockReset();
    deleteDocumentMock.mockReset();
    getUsageLogsMock.mockReset();
    getUsageStatsMock.mockReset();

    getAdminSettingsMock.mockResolvedValue({
      generator_model: "google/gemini-3-flash-preview",
      reranker_model: "google/gemini-3-flash-preview",
      system_prompt: "Current prompt",
      embedding_profile: "local_minilm",
      pipeline_mode: "baseline_v1",
      reranker_mode: "off",
      chunk_profile: "standard",
      parser_profile: "docling_v1",
      max_chunks_per_doc: 3,
      embedding_provider: "openrouter",
      embedding_model: "openai/text-embedding-3-large",
    });
    listAdminDocumentsMock.mockResolvedValue({ documents: [], count: 0 });
    getJobStatusMock.mockResolvedValue({ status: "idle" });
    getNewsJobStatusMock.mockResolvedValue({
      status: "idle",
      mode: null,
      pages: 0,
      processed_count: 0,
      added_count: 0,
      updated_count: 0,
      unchanged_count: 0,
      embedded_count: 0,
      last_run_at: null,
    });
    updateAdminSettingsMock.mockResolvedValue({
      generator_model: "google/gemini-3-flash-preview",
      reranker_model: "google/gemini-3-flash-preview",
      system_prompt: "Updated prompt",
      embedding_profile: "local_minilm",
      pipeline_mode: "baseline_v1",
      reranker_mode: "off",
      chunk_profile: "standard",
      parser_profile: "docling_v1",
      max_chunks_per_doc: 3,
      embedding_provider: "openrouter",
      embedding_model: "openai/text-embedding-3-large",
    });
    triggerJobMock.mockResolvedValue({ status: "queued" });
    triggerNewsJobMock.mockResolvedValue({ status: "queued", mode: "sync" });
    getUsageStatsMock.mockResolvedValue({
      window_days: 7,
      generated_at_utc: "2026-03-28T10:30:00+00:00",
      total_queries: 12,
      avg_latency_ms: 154.2,
      citation_presence_rate: 0.75,
      non_empty_answer_rate: 1,
      confidence_distribution: { high: 6, medium: 4, low: 2, unknown: 0 },
      source_mix_pdf_vs_news: { pdf: 18, news: 5 },
    });
    getUsageLogsMock.mockResolvedValue({
      entries: [
        {
          timestamp_utc: "2026-03-28T10:29:00+00:00",
          query_text: "When is the thesis submission deadline?",
          answer_length_chars: 120,
          confidence: "high",
          model_used: "google/gemini-3-flash-preview",
          reranker_model: "google/gemini-3-flash-preview",
          latency_ms: 132.5,
          cited_sources_count: 2,
          source_types: { pdf: 2, news: 0 },
          status: "ok",
        },
      ],
      count: 1,
      limit: 50,
    });
  });

  it("submits updated runtime settings payload", async () => {
    render(<AdminPage />);

    await waitFor(() => {
      expect(screen.getByText("Admin Panel")).toBeInTheDocument();
    });

    const user = userEvent.setup();
    const promptInput = screen.getByLabelText("Additional instructions (optional)");
    await user.clear(promptInput);
    await user.type(promptInput, "Updated prompt");
    await user.click(screen.getByRole("button", { name: /save settings/i }));

    await waitFor(() => {
      expect(updateAdminSettingsMock).toHaveBeenCalledWith({
        generator_model: "google/gemini-3-flash-preview",
        system_prompt: "Updated prompt",
        embedding_profile: "local_minilm",
        pipeline_mode: "baseline_v1",
        reranker_mode: "off",
      });
    });
  });

  it("keeps document sync and reindex as separate triggers", async () => {
    render(<AdminPage />);

    await waitFor(() => {
      expect(screen.getByText("Admin Panel")).toBeInTheDocument();
    });

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /start documents sync/i }));
    await user.click(screen.getByRole("button", { name: /start reindex/i }));

    expect(triggerJobMock).toHaveBeenCalledWith("documents/sync");
    expect(triggerJobMock).toHaveBeenCalledWith("reindex");
  });

  it("triggers news bootstrap and sync independently", async () => {
    render(<AdminPage />);

    await waitFor(() => {
      expect(screen.getByText("News Index")).toBeInTheDocument();
    });

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /bootstrap \(4 pages\)/i }));
    await user.click(screen.getByRole("button", { name: /sync \(2 pages\)/i }));

    expect(triggerNewsJobMock).toHaveBeenCalledWith("bootstrap");
    expect(triggerNewsJobMock).toHaveBeenCalledWith("sync");
  });

  it("renders usage overview and recent queries", async () => {
    render(<AdminPage />);

    await waitFor(() => {
      expect(screen.getByText("Usage Overview")).toBeInTheDocument();
    });

    expect(screen.getByText(/Total queries \(last 7 days\):/i)).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByText(/75.0%/i)).toBeInTheDocument();
    expect(screen.getByText("Recent Queries")).toBeInTheDocument();
    expect(screen.getByText(/When is the thesis submission deadline\?/i)).toBeInTheDocument();
  });

  it("shows empty usage state when there are no records", async () => {
    getUsageLogsMock.mockResolvedValueOnce({ entries: [], count: 0, limit: 50 });

    render(<AdminPage />);

    await waitFor(() => {
      expect(screen.getByText("Recent Queries")).toBeInTheDocument();
    });

    expect(screen.getByText(/No usage records yet/i)).toBeInTheDocument();
  });
});
