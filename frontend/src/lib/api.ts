import type {
  AskResponse,
  DocumentsResponse,
  FeedbackResponse,
  JobStatusResponse,
  NewsJobStatusResponse,
  RuntimeSettings,
  RuntimeSettingsUpdate,
  UsageLogResponse,
  UsageStatsResponse,
} from "@/types/api";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api").replace(
  /\/$/,
  "",
);

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.headers || {}),
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed (${response.status})`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function askQuestion(query: string): Promise<AskResponse> {
  return apiFetch<AskResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
}

export async function submitFeedback(
  requestId: string,
  helpful: boolean,
): Promise<FeedbackResponse> {
  return apiFetch<FeedbackResponse>("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: requestId, helpful }),
  });
}

export async function getAdminSettings(): Promise<RuntimeSettings> {
  return apiFetch<RuntimeSettings>("/admin/settings");
}

export async function updateAdminSettings(
  payload: RuntimeSettingsUpdate,
): Promise<RuntimeSettings> {
  return apiFetch<RuntimeSettings>("/admin/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function listAdminDocuments(): Promise<DocumentsResponse> {
  return apiFetch<DocumentsResponse>("/admin/documents");
}

export async function uploadDocument(file: File): Promise<{ status: string; file_name: string }> {
  const formData = new FormData();
  formData.append("file", file);

  return apiFetch<{ status: string; file_name: string }>("/admin/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export async function deleteDocument(fileName: string): Promise<{ status: string; file_name: string }> {
  return apiFetch<{ status: string; file_name: string }>(
    `/admin/documents/${encodeURIComponent(fileName)}`,
    {
      method: "DELETE",
    },
  );
}

export async function triggerJob(
  job: "documents/sync" | "reindex",
): Promise<JobStatusResponse> {
  return apiFetch<JobStatusResponse>(`/admin/${job}`, { method: "POST" });
}

export async function getJobStatus(
  job: "documents/sync" | "reindex",
): Promise<JobStatusResponse> {
  return apiFetch<JobStatusResponse>(`/admin/${job}`);
}

export async function triggerNewsJob(
  mode: "bootstrap" | "sync",
): Promise<NewsJobStatusResponse> {
  return apiFetch<NewsJobStatusResponse>(`/admin/news/${mode}`, { method: "POST" });
}

export async function getNewsJobStatus(): Promise<NewsJobStatusResponse> {
  return apiFetch<NewsJobStatusResponse>("/admin/news");
}

export async function getUsageLogs(limit = 200): Promise<UsageLogResponse> {
  return apiFetch<UsageLogResponse>(`/admin/usage?limit=${encodeURIComponent(String(limit))}`);
}

export async function getUsageStats(windowDays = 7): Promise<UsageStatsResponse> {
  return apiFetch<UsageStatsResponse>(
    `/admin/usage/stats?window_days=${encodeURIComponent(String(windowDays))}`,
  );
}
