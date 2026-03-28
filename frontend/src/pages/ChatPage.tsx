import { FormEvent, useEffect, useMemo, useState } from "react";
import { Bot, ExternalLink, LoaderCircle, Send, User } from "lucide-react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { API_BASE_URL, askQuestion } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { CitedSourceItem } from "@/types/api";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  confidence?: string;
  reasoning?: string;
  citedSources?: CitedSourceItem[];
};

const welcomeMessage: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text: "Hi! Ask me about ELTE regulations, deadlines, and procedures. I will answer from indexed university sources.",
};

const CHAT_STORAGE_KEY = "elte-rag-assistant/chat-v1";

function markdownUrlTransform(url: string): string {
  if (url.startsWith("cite:")) {
    return url;
  }
  return defaultUrlTransform(url);
}

function isExternalUrl(source: string): boolean {
  return /^https?:\/\//i.test(source);
}

function localSourceExtension(source: string): ".pdf" | ".doc" | ".docx" | null {
  const lowerSource = source.toLowerCase();
  if (lowerSource.endsWith(".pdf")) {
    return ".pdf";
  }
  if (lowerSource.endsWith(".docx")) {
    return ".docx";
  }
  if (lowerSource.endsWith(".doc")) {
    return ".doc";
  }
  return null;
}

function buildCitationSourceUrl(citation: CitedSourceItem): string | null {
  const source = citation.source?.trim();
  if (!source) {
    return null;
  }
  const sourceType = citation.source_type ?? "pdf";

  const pageFragment = citation.page ? `#page=${citation.page}` : "";

  if (isExternalUrl(source)) {
    if (citation.page && source.toLowerCase().endsWith(".pdf") && !source.includes("#")) {
      return `${source}${pageFragment}`;
    }
    return source;
  }

  const extension = localSourceExtension(source);
  if (extension) {
    const pageSuffix = extension === ".pdf" ? pageFragment : "";
    return `${API_BASE_URL}/files/${encodeURIComponent(source)}${pageSuffix}`;
  }

  if (sourceType === "news" && source.startsWith("/")) {
    return `https://www.inf.elte.hu${source}`;
  }

  return null;
}

function formatPublishedAt(value: string | null | undefined): string | null {
  if (!value) {
    return null;
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleDateString();
}

function normalizeCitationText(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function citationMarker(citationId: string): string {
  return citationId.toUpperCase().startsWith("C") ? citationId.slice(1) : citationId;
}

function normalizeInlineCitations(
  text: string,
  citedSources: CitedSourceItem[] | undefined,
): string {
  if (!citedSources?.length) {
    return text;
  }

  const citationsById = new Map<string, CitedSourceItem>();
  const docPageMap = new Map<string, string>();
  const docOnlyMap = new Map<string, Set<string>>();

  for (const source of citedSources) {
    const citationId = source.citation_id?.toUpperCase();
    if (!citationId) {
      continue;
    }
    citationsById.set(citationId, source);

    const document = source.document?.trim();
    if (!document) {
      continue;
    }

    const docNormalized = normalizeCitationText(document);
    if (!docOnlyMap.has(docNormalized)) {
      docOnlyMap.set(docNormalized, new Set<string>());
    }
    docOnlyMap.get(docNormalized)?.add(citationId);

    if (source.page) {
      docPageMap.set(`${docNormalized}::${source.page}`, citationId);
    }
  }

  const withChunkIds = text.replace(/\[([Cc]\d+)\]/g, (match, rawCitationId: string) => {
    const citationId = rawCitationId.toUpperCase();
    if (!citationsById.has(citationId)) {
      return match;
    }
    return `[${citationMarker(citationId)}](cite:${citationId})`;
  });

  return withChunkIds.replace(/\[([^\[\]\n]+)\](?!\()/g, (match, rawReference: string) => {
    const reference = rawReference.trim();
    const parsed = reference.match(/^(.*?),\s*p\.?\s*(\d+)\s*$/i);
    if (parsed) {
      const document = normalizeCitationText(parsed[1] ?? "");
      const page = Number.parseInt(parsed[2] ?? "", 10);
      const citationId = docPageMap.get(`${document}::${page}`);
      if (citationId) {
        return `[${citationMarker(citationId)}](cite:${citationId})`;
      }
    }

    const docOnlyCandidates = docOnlyMap.get(normalizeCitationText(reference));
    if (docOnlyCandidates && docOnlyCandidates.size === 1) {
      const citationId = Array.from(docOnlyCandidates)[0];
      return `[${citationMarker(citationId)}](cite:${citationId})`;
    }

    return match;
  });
}

export function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    if (typeof window === "undefined") {
      return [welcomeMessage];
    }

    try {
      const raw = window.localStorage.getItem(CHAT_STORAGE_KEY);
      if (!raw) {
        return [welcomeMessage];
      }

      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed as ChatMessage[];
      }
    } catch {
      return [welcomeMessage];
    }
    return [welcomeMessage];
  });
  const [query, setQuery] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [highlightedCitationKey, setHighlightedCitationKey] = useState<string | null>(null);

  const canSend = useMemo(() => query.trim().length > 0 && !isSending, [isSending, query]);

  useEffect(() => {
    window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    if (!highlightedCitationKey) {
      return;
    }
    const timeout = window.setTimeout(() => {
      setHighlightedCitationKey((current) =>
        current === highlightedCitationKey ? null : current,
      );
    }, 1400);
    return () => window.clearTimeout(timeout);
  }, [highlightedCitationKey]);

  const onNewChat = () => {
    setMessages([welcomeMessage]);
    setQuery("");
    setError(null);
    setHighlightedCitationKey(null);
    window.localStorage.removeItem(CHAT_STORAGE_KEY);
  };

  const onInlineCitationClick = (messageId: string, citationId: string) => {
    const key = `${messageId}:${citationId}`;
    const element = document.getElementById(`citation-${messageId}-${citationId}`);
    if (element && typeof element.scrollIntoView === "function") {
      element.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    setHighlightedCitationKey(key);
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const prompt = query.trim();
    if (!prompt || isSending) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      text: prompt,
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setError(null);
    setIsSending(true);

    try {
      const response = await askQuestion(prompt);
      const assistantMessage: ChatMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        text: response.answer,
        confidence: response.confidence,
        reasoning: response.reasoning,
        citedSources: response.cited_sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Failed to query assistant.";
      setError(message);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2">
          <CardTitle>Assistant Chat</CardTitle>
          <Button variant="outline" onClick={onNewChat}>
            New chat
          </Button>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="max-h-[60vh] space-y-3 overflow-y-auto pr-1">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "rounded-lg border p-3",
                  message.role === "assistant"
                    ? "border-border bg-card"
                    : "ml-auto border-primary/40 bg-primary/10",
                )}
              >
                <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {message.role === "assistant" ? <Bot className="h-3.5 w-3.5" /> : <User className="h-3.5 w-3.5" />}
                  {message.role}
                </div>
                {message.role === "assistant" ? (
                  <div className="text-sm leading-6">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      urlTransform={markdownUrlTransform}
                      components={{
                        a: ({ href, children }) => {
                          if (href?.startsWith("cite:")) {
                            const citationId = href.slice("cite:".length).toUpperCase();
                            return (
                              <sup className="mx-px align-super">
                                <button
                                  type="button"
                                  aria-label={`Citation ${citationId}`}
                                  title={`Go to citation ${citationId}`}
                                  onClick={() => onInlineCitationClick(message.id, citationId)}
                                  className="cursor-pointer rounded px-px text-[1em] font-semibold leading-none text-primary transition-colors hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                                >
                                  {children}
                                </button>
                              </sup>
                            );
                          }

                          return (
                            <a
                              href={href}
                              target="_blank"
                              rel="noreferrer noopener"
                              className="text-primary underline decoration-primary/50 underline-offset-2 hover:decoration-primary"
                            >
                              {children}
                            </a>
                          );
                        },
                      }}
                    >
                      {normalizeInlineCitations(message.text, message.citedSources)}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap text-sm leading-6">{message.text}</p>
                )}

                {message.role === "assistant" && message.confidence ? (
                  <p className="mt-3 text-xs text-muted-foreground">
                    Confidence: <span className="font-medium">{message.confidence}</span>
                  </p>
                ) : null}

                {message.role === "assistant" && message.reasoning ? (
                  <p className="mt-1 text-xs text-muted-foreground">Reasoning: {message.reasoning}</p>
                ) : null}

                {message.role === "assistant" && message.citedSources?.length ? (
                  <div className="mt-3 space-y-2 rounded-md border border-border/70 bg-muted/40 p-2">
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Citations</p>
                    {message.citedSources.map((source, index) => (
                      (() => {
                        const citationId = source.citation_id || `C${index + 1}`;
                        const citationKey = `${message.id}:${citationId}`;
                        const sourceUrl = buildCitationSourceUrl(source);
                        const isHighlighted = highlightedCitationKey === citationKey;
                        const sourceType = source.source_type ?? "pdf";
                        const publishedAt = formatPublishedAt(source.published_at);
                        const sourceLabel = sourceType === "news" ? "News" : "PDF";

                        return (
                          <div
                            key={`${message.id}-${citationId}`}
                            id={`citation-${message.id}-${citationId}`}
                            className={cn(
                              "rounded-md bg-card p-2 text-xs transition-all duration-300",
                              isHighlighted ? "ring-2 ring-primary/70 shadow-md" : "ring-0",
                            )}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <p className="font-medium">
                                {citationId} • {source.document}
                                {sourceType === "pdf" && source.page ? `, p. ${source.page}` : ""}
                              </p>
                              {sourceUrl ? (
                                <a
                                  href={sourceUrl}
                                  target="_blank"
                                  rel="noreferrer noopener"
                                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-[11px] hover:bg-muted"
                                >
                                  Open source
                                  <ExternalLink className="h-3 w-3" />
                                </a>
                              ) : null}
                            </div>
                            <p className="mt-1 text-[11px] uppercase tracking-wide text-muted-foreground">
                              {sourceLabel}
                              {publishedAt ? ` • ${publishedAt}` : ""}
                            </p>
                            <p className="mt-1 text-muted-foreground">{source.relevant_snippet}</p>
                          </div>
                        );
                      })()
                    ))}
                  </div>
                ) : null}
              </div>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="flex items-center gap-2">
            <Input
              placeholder="Ask a question about ELTE policies..."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              aria-label="Question"
            />
            <Button type="submit" disabled={!canSend} className="gap-1.5">
              {isSending ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              Send
            </Button>
          </form>

          {error ? <p className="text-sm text-red-700">{error}</p> : null}
        </CardContent>
      </Card>
    </div>
  );
}
