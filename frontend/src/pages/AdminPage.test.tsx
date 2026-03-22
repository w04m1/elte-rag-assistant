import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { AdminPage } from "@/pages/AdminPage";

const getAdminSettingsMock = vi.fn();
const listAdminDocumentsMock = vi.fn();
const getJobStatusMock = vi.fn();
const updateAdminSettingsMock = vi.fn();
const triggerJobMock = vi.fn();
const uploadDocumentMock = vi.fn();
const deleteDocumentMock = vi.fn();

vi.mock("@/lib/api", () => ({
  getAdminSettings: (...args: unknown[]) => getAdminSettingsMock(...args),
  listAdminDocuments: (...args: unknown[]) => listAdminDocumentsMock(...args),
  getJobStatus: (...args: unknown[]) => getJobStatusMock(...args),
  updateAdminSettings: (...args: unknown[]) => updateAdminSettingsMock(...args),
  triggerJob: (...args: unknown[]) => triggerJobMock(...args),
  uploadDocument: (...args: unknown[]) => uploadDocumentMock(...args),
  deleteDocument: (...args: unknown[]) => deleteDocumentMock(...args),
}));

describe("AdminPage", () => {
  beforeEach(() => {
    getAdminSettingsMock.mockReset();
    listAdminDocumentsMock.mockReset();
    getJobStatusMock.mockReset();
    updateAdminSettingsMock.mockReset();
    triggerJobMock.mockReset();
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
    updateAdminSettingsMock.mockResolvedValue({
      generator_model: "google/gemini-3-flash-preview",
      reranker_model: "google/gemini-3-flash-preview",
      system_prompt: "Updated prompt",
      embedding_provider: "openrouter",
      embedding_model: "openai/text-embedding-3-large",
    });
    triggerJobMock.mockResolvedValue({ status: "queued" });
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

  it("keeps scrape and reindex as separate triggers", async () => {
    render(<AdminPage />);

    await waitFor(() => {
      expect(screen.getByText("Admin Panel")).toBeInTheDocument();
    });

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /start scrape/i }));
    await user.click(screen.getByRole("button", { name: /start reindex/i }));

    expect(triggerJobMock).toHaveBeenCalledWith("scrape");
    expect(triggerJobMock).toHaveBeenCalledWith("reindex");
  });
});
