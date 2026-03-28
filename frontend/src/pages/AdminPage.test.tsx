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

    getAdminSettingsMock.mockResolvedValue({
      generator_model: "google/gemini-3-flash-preview",
      reranker_model: "google/gemini-3-flash-preview",
      system_prompt: "Current prompt",
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
      embedding_provider: "openrouter",
      embedding_model: "openai/text-embedding-3-large",
    });
    triggerJobMock.mockResolvedValue({ status: "queued" });
    triggerNewsJobMock.mockResolvedValue({ status: "queued", mode: "sync" });
  });

  it("submits updated runtime settings payload", async () => {
    render(<AdminPage />);

    await waitFor(() => {
      expect(screen.getByText("Admin Panel")).toBeInTheDocument();
    });

    const user = userEvent.setup();
    const promptInput = screen.getByLabelText("System prompt");
    await user.clear(promptInput);
    await user.type(promptInput, "Updated prompt");
    await user.click(screen.getByRole("button", { name: /save settings/i }));

    await waitFor(() => {
      expect(updateAdminSettingsMock).toHaveBeenCalledWith({
        generator_model: "google/gemini-3-flash-preview",
        reranker_model: "google/gemini-3-flash-preview",
        system_prompt: "Updated prompt",
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
});
