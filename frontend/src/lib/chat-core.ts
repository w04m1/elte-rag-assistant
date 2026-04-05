import type { ChatHistoryTurn, CitedSourceItem } from "../types/api";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  confidence?: string;
  reasoning?: string;
  citedSources?: CitedSourceItem[];
  requestId?: string;
  feedback?: boolean | null;
};

export const CHAT_WELCOME_MESSAGE_ID = "welcome";
export const CHAT_WELCOME_MESSAGE_TEXT =
  "Hi! Ask me about ELTE regulations, deadlines, and procedures. I will answer from indexed university sources.";

export function createWelcomeMessage(): ChatMessage {
  return {
    id: CHAT_WELCOME_MESSAGE_ID,
    role: "assistant",
    text: CHAT_WELCOME_MESSAGE_TEXT,
  };
}

export function normalizeApiBaseUrl(rawValue: string): string {
  return rawValue.trim().replace(/\/+$/, "");
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

export function buildCitationSourceUrl(
  citation: CitedSourceItem,
  apiBaseUrl: string,
): string | null {
  const source = citation.source?.trim();
  if (!source) {
    return null;
  }

  const normalizedApiBaseUrl = normalizeApiBaseUrl(apiBaseUrl);
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
    const filesBase = normalizedApiBaseUrl ? `${normalizedApiBaseUrl}/files` : "/files";
    return `${filesBase}/${encodeURIComponent(source)}${pageSuffix}`;
  }

  if (sourceType === "news" && source.startsWith("/")) {
    return `https://www.inf.elte.hu${source}`;
  }

  return null;
}

export function formatPublishedAt(value: string | null | undefined): string | null {
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

export function normalizeInlineCitations(
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

export function buildHistoryForRequest(messages: ChatMessage[]): ChatHistoryTurn[] {
  return messages
    .filter((message) => message.id !== CHAT_WELCOME_MESSAGE_ID)
    .map((message) => {
      const historyTurn: ChatHistoryTurn = {
        role: message.role,
        text: message.text,
      };

      if (message.role === "assistant" && message.citedSources?.length) {
        historyTurn.cited_sources = message.citedSources;
      }

      return historyTurn;
    });
}
