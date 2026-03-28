import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { ChatPage } from "@/pages/ChatPage";

const askQuestionMock = vi.fn();

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    askQuestion: (...args: unknown[]) => askQuestionMock(...args),
  };
});

describe("ChatPage", () => {
  beforeEach(() => {
    askQuestionMock.mockReset();
    window.localStorage.clear();
  });

  it("renders GFM markdown and connects inline citation refs with citation cards", async () => {
    askQuestionMock.mockResolvedValue({
      answer: "The deadline is [Thesis Rules, p. 3].\n\n- Bring required forms",
      sources: [],
      model_used: "demo-model",
      reasoning: "Matched thesis rule chunk.",
      confidence: "high",
      cited_sources: [
        {
          citation_id: "C1",
          source: "thesis_rules.pdf",
          document: "Thesis Rules",
          page: 3,
          relevant_snippet: "Students must submit by April 15th.",
          source_type: "pdf",
        },
      ],
    });

    const user = userEvent.setup();
    render(<ChatPage />);

    await user.type(screen.getByLabelText("Question"), "When is thesis deadline?");
    await user.click(screen.getByRole("button", { name: /^send$/i }));

    await waitFor(() => {
      expect(screen.getByText(/deadline is/i)).toBeInTheDocument();
    });

    expect(screen.getByRole("button", { name: /new chat/i })).toBeInTheDocument();
    expect(screen.getByText(/Bring required forms/)).toBeInTheDocument();

    const inlineCitation = screen.getByRole("button", { name: /citation c1/i });
    await user.click(inlineCitation);

    const citationHeading = screen.getByText(/C1 • Thesis Rules, p. 3/i);
    const citationCard = citationHeading.closest("[id^='citation-']");
    expect(citationCard).toHaveClass("ring-2");

    const sourceLink = screen.getByRole("link", { name: /open source/i });
    expect(sourceLink).toHaveAttribute("href", "/api/files/thesis_rules.pdf#page=3");
    expect(screen.getByText(/^PDF$/i)).toBeInTheDocument();
  });

  it("renders news citations with external links and source type", async () => {
    askQuestionMock.mockResolvedValue({
      answer: "Latest update is available. [C1]",
      sources: [],
      model_used: "demo-model",
      reasoning: "",
      confidence: "medium",
      cited_sources: [
        {
          citation_id: "C1",
          source: "https://inf.elte.hu/en/node/326060",
          document: "Double professional success",
          page: null,
          relevant_snippet: "The Faculty won two Marketing Diamond Awards.",
          source_type: "news",
          published_at: "2026-03-08T11:45:37+00:00",
        },
      ],
    });

    const user = userEvent.setup();
    render(<ChatPage />);

    await user.type(screen.getByLabelText("Question"), "Any recent awards?");
    await user.click(screen.getByRole("button", { name: /^send$/i }));

    await waitFor(() => {
      expect(screen.getByText(/latest update/i)).toBeInTheDocument();
    });

    const sourceLink = screen.getByRole("link", { name: /open source/i });
    expect(sourceLink).toHaveAttribute("href", "https://inf.elte.hu/en/node/326060");
    expect(screen.getByText(/^News/i)).toBeInTheDocument();
  });

  it("renders local DOCX citations through backend file endpoint", async () => {
    askQuestionMock.mockResolvedValue({
      answer: "Please fill the form. [C1]",
      sources: [],
      model_used: "demo-model",
      reasoning: "",
      confidence: "medium",
      cited_sources: [
        {
          citation_id: "C1",
          source: "credittransfer_2024-25-2.docx",
          document: "Credit Transfer Form",
          page: null,
          relevant_snippet: "Use the latest DOCX form for submission.",
          source_type: "pdf",
        },
      ],
    });

    const user = userEvent.setup();
    render(<ChatPage />);

    await user.type(screen.getByLabelText("Question"), "Where is the credit transfer form?");
    await user.click(screen.getByRole("button", { name: /^send$/i }));

    await waitFor(() => {
      expect(screen.getByText(/please fill the form/i)).toBeInTheDocument();
    });

    const sourceLink = screen.getByRole("link", { name: /open source/i });
    expect(sourceLink).toHaveAttribute("href", "/api/files/credittransfer_2024-25-2.docx");
  });

  it("persists current chat and clears it with New chat", async () => {
    askQuestionMock.mockResolvedValue({
      answer: "Saved answer",
      sources: [],
      model_used: "demo-model",
      reasoning: "",
      confidence: "",
      cited_sources: [],
    });

    const user = userEvent.setup();
    const firstRender = render(<ChatPage />);

    await user.type(screen.getByLabelText("Question"), "Persist me");
    await user.click(screen.getByRole("button", { name: /^send$/i }));

    await waitFor(() => {
      expect(screen.getByText("Persist me")).toBeInTheDocument();
    });

    firstRender.unmount();
    render(<ChatPage />);

    expect(screen.getByText("Persist me")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /new chat/i }));

    expect(screen.queryByText("Persist me")).not.toBeInTheDocument();
    expect(screen.getByText(/Hi! Ask me about ELTE regulations/i)).toBeInTheDocument();
  });
});
