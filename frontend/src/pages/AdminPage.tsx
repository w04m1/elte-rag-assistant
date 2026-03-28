import { ChangeEvent, useEffect, useMemo, useState } from "react";
import { LoaderCircle, RefreshCw, Trash2, Upload } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  deleteDocument,
  getAdminSettings,
  getNewsJobStatus,
  getJobStatus,
  getUsageLogs,
  getUsageStats,
  listAdminDocuments,
  triggerNewsJob,
  triggerJob,
  updateAdminSettings,
  uploadDocument,
} from "@/lib/api";
import type {
  DocumentListItem,
  JobStatusResponse,
  NewsJobStatusResponse,
  RuntimeSettings,
  UsageLogEntry,
  UsageStatsResponse,
} from "@/types/api";

const generatorModelPresets = [
  "google/gemini-3-flash-preview",
  "openai/gpt-4o-mini",
  "openai/gpt-4.1-mini",
  "meta-llama/llama-3.1-70b-instruct",
  "custom",
];

const rerankerModelPresets = [
  "google/gemini-3-flash-preview",
  "openai/gpt-4o-mini",
  "openai/gpt-4.1-mini",
  "custom",
];

function presetForModel(model: string, presets: string[]): string {
  return presets.includes(model) ? model : "custom";
}

function formatRate(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function trimQuery(query: string, maxLength = 220): string {
  if (query.length <= maxLength) {
    return query;
  }
  return `${query.slice(0, maxLength - 1)}…`;
}

export function AdminPage() {
  const [settings, setSettings] = useState<RuntimeSettings | null>(null);
  const [generatorPreset, setGeneratorPreset] = useState("google/gemini-3-flash-preview");
  const [rerankerPreset, setRerankerPreset] = useState("google/gemini-3-flash-preview");
  const [customGeneratorModel, setCustomGeneratorModel] = useState("");
  const [customRerankerModel, setCustomRerankerModel] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");

  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [file, setFile] = useState<File | null>(null);

  const [documentsSyncStatus, setDocumentsSyncStatus] = useState<JobStatusResponse>({
    status: "idle",
  });
  const [reindexStatus, setReindexStatus] = useState<JobStatusResponse>({ status: "idle" });
  const [newsStatus, setNewsStatus] = useState<NewsJobStatusResponse>({ status: "idle" });
  const [usageStats, setUsageStats] = useState<UsageStatsResponse | null>(null);
  const [usageEntries, setUsageEntries] = useState<UsageLogEntry[]>([]);

  const [isLoading, setIsLoading] = useState(true);
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isPolling = useMemo(
    () =>
      [documentsSyncStatus, reindexStatus, newsStatus].some((status) =>
        ["queued", "running"].includes(status.status),
      ),
    [documentsSyncStatus, reindexStatus, newsStatus],
  );

  const loadSettings = async () => {
    const nextSettings = await getAdminSettings();
    setSettings(nextSettings);
    setSystemPrompt(nextSettings.system_prompt);
    setGeneratorPreset(presetForModel(nextSettings.generator_model, generatorModelPresets));
    setCustomGeneratorModel(nextSettings.generator_model);
    setRerankerPreset(presetForModel(nextSettings.reranker_model, rerankerModelPresets));
    setCustomRerankerModel(nextSettings.reranker_model);
  };

  const loadDocuments = async () => {
    const response = await listAdminDocuments();
    setDocuments(response.documents);
  };

  const loadStatuses = async () => {
    const [nextDocumentsSyncStatus, nextReindexStatus, nextNewsStatus] = await Promise.all([
      getJobStatus("documents/sync"),
      getJobStatus("reindex"),
      getNewsJobStatus(),
    ]);
    setDocumentsSyncStatus(nextDocumentsSyncStatus);
    setReindexStatus(nextReindexStatus);
    setNewsStatus(nextNewsStatus);
  };

  const loadUsage = async () => {
    const [nextUsageStats, nextUsageEntries] = await Promise.all([
      getUsageStats(7),
      getUsageLogs(50),
    ]);
    setUsageStats(nextUsageStats);
    setUsageEntries(nextUsageEntries.entries);
  };

  const reloadAll = async () => {
    setError(null);
    setIsLoading(true);
    try {
      await Promise.all([loadSettings(), loadDocuments(), loadStatuses(), loadUsage()]);
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Failed to load admin panel data.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void reloadAll();
  }, []);

  useEffect(() => {
    if (!isPolling) {
      return;
    }

    const interval = setInterval(() => {
      void loadStatuses();
    }, 2500);

    return () => clearInterval(interval);
  }, [isPolling]);

  const onSaveSettings = async () => {
    const generatorModel = generatorPreset === "custom" ? customGeneratorModel.trim() : generatorPreset;
    const rerankerModel = rerankerPreset === "custom" ? customRerankerModel.trim() : rerankerPreset;

    if (!generatorModel || !rerankerModel) {
      setError("Generator and reranker model names are required.");
      return;
    }

    setError(null);
    setIsSavingSettings(true);
    try {
      const nextSettings = await updateAdminSettings({
        generator_model: generatorModel,
        reranker_model: rerankerModel,
        system_prompt: systemPrompt,
      });
      setSettings(nextSettings);
      setGeneratorPreset(presetForModel(nextSettings.generator_model, generatorModelPresets));
      setCustomGeneratorModel(nextSettings.generator_model);
      setRerankerPreset(presetForModel(nextSettings.reranker_model, rerankerModelPresets));
      setCustomRerankerModel(nextSettings.reranker_model);
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Failed to update settings.";
      setError(message);
    } finally {
      setIsSavingSettings(false);
    }
  };

  const onUploadFile = async () => {
    if (!file) {
      setError("Pick a PDF first.");
      return;
    }

    setError(null);
    setIsUploading(true);
    try {
      await uploadDocument(file);
      setFile(null);
      await loadDocuments();
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Failed to upload document.";
      setError(message);
    } finally {
      setIsUploading(false);
    }
  };

  const onDeleteDocument = async (source: string) => {
    setError(null);
    try {
      await deleteDocument(source);
      await loadDocuments();
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Failed to delete document.";
      setError(message);
    }
  };

  const onTriggerJob = async (job: "documents/sync" | "reindex") => {
    setError(null);
    try {
      const status = await triggerJob(job);
      if (job === "documents/sync") {
        setDocumentsSyncStatus(status);
      } else {
        setReindexStatus(status);
      }
      await loadStatuses();
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : `Failed to start ${job}.`;
      setError(message);
    }
  };

  const onTriggerNewsJob = async (mode: "bootstrap" | "sync") => {
    setError(null);
    try {
      const status = await triggerNewsJob(mode);
      setNewsStatus(status);
      await loadStatuses();
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : `Failed to start news ${mode}.`;
      setError(message);
    }
  };

  const renderJobStatus = (status: JobStatusResponse) => {
    const result = (status.result || {}) as Record<string, unknown>;
    const discovered = result.discovered_url_count as number | undefined;
    const downloaded = result.downloaded_count as number | undefined;
    const downloadedPdf = result.downloaded_pdf_count as number | undefined;
    const downloadedDoc = result.downloaded_doc_count as number | undefined;
    const downloadedDocx = result.downloaded_docx_count as number | undefined;
    const skipped = result.skipped_count as number | undefined;

    return (
      <div className="space-y-1 text-sm text-muted-foreground">
        <p>
          Status: <span className="font-medium text-foreground">{status.status}</span>
        </p>
        {typeof status.vector_count === "number" ? <p>Vector count: {status.vector_count}</p> : null}
        {typeof discovered === "number" ? <p>Discovered URLs: {discovered}</p> : null}
        {typeof downloaded === "number" ? <p>Downloaded files: {downloaded}</p> : null}
        {typeof downloadedPdf === "number" ? <p>Downloaded PDF: {downloadedPdf}</p> : null}
        {typeof downloadedDoc === "number" ? <p>Downloaded DOC: {downloadedDoc}</p> : null}
        {typeof downloadedDocx === "number" ? <p>Downloaded DOCX: {downloadedDocx}</p> : null}
        {typeof skipped === "number" ? <p>Skipped existing: {skipped}</p> : null}
        {status.error ? <p className="text-red-700">Error: {status.error}</p> : null}
      </div>
    );
  };

  const renderNewsJobStatus = (status: NewsJobStatusResponse) => (
    <div className="space-y-1 text-sm text-muted-foreground">
      <p>
        Status: <span className="font-medium text-foreground">{status.status}</span>
      </p>
      {status.mode ? <p>Mode: {status.mode}</p> : null}
      {typeof status.pages === "number" ? <p>Pages fetched: {status.pages}</p> : null}
      {typeof status.processed_count === "number" ? (
        <p>Processed records: {status.processed_count}</p>
      ) : null}
      {typeof status.added_count === "number" ? <p>Added records: {status.added_count}</p> : null}
      {typeof status.updated_count === "number" ? <p>Updated records: {status.updated_count}</p> : null}
      {typeof status.unchanged_count === "number" ? (
        <p>Unchanged records: {status.unchanged_count}</p>
      ) : null}
      {typeof status.embedded_count === "number" ? (
        <p>News vectors: {status.embedded_count}</p>
      ) : null}
      {status.last_run_at ? <p>Last run: {new Date(status.last_run_at).toLocaleString()}</p> : null}
      {status.error ? <p className="text-red-700">Error: {status.error}</p> : null}
    </div>
  );

  if (isLoading && !settings) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <LoaderCircle className="h-4 w-4 animate-spin" />
        Loading admin panel...
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h1 className="text-lg font-semibold">Admin Panel</h1>
        <Button variant="outline" className="gap-1.5" onClick={() => void reloadAll()}>
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {error ? <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{error}</p> : null}

      <Card>
        <CardHeader>
          <CardTitle>Recent Queries</CardTitle>
        </CardHeader>
        <CardContent>
          {usageEntries.length === 0 ? (
            <p className="text-sm text-muted-foreground">No usage records yet.</p>
          ) : (
            <div className="max-h-[28rem] overflow-x-auto overflow-y-auto rounded-md border border-border">
              <table className="min-w-[980px] w-full text-left text-sm">
                <thead className="sticky top-0 bg-muted/80 text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 font-medium">Time</th>
                    <th className="w-[42%] min-w-[360px] px-3 py-2 font-medium">Query</th>
                    <th className="px-3 py-2 font-medium">Status</th>
                    <th className="px-3 py-2 font-medium">Confidence</th>
                    <th className="px-3 py-2 font-medium">Latency</th>
                    <th className="px-3 py-2 font-medium">Citations</th>
                    <th className="px-3 py-2 font-medium">Feedback</th>
                    <th className="px-3 py-2 font-medium">Model</th>
                  </tr>
                </thead>
                <tbody>
                  {usageEntries.map((entry, index) => (
                    <tr key={`${entry.timestamp_utc}-${index}`} className="border-t border-border/60">
                      <td className="px-3 py-2 align-top text-muted-foreground">
                        {formatTimestamp(entry.timestamp_utc)}
                      </td>
                      <td className="px-3 py-2 align-top whitespace-normal break-words">
                        {trimQuery(entry.query_text)}
                      </td>
                      <td className="px-3 py-2 align-top">{entry.status}</td>
                      <td className="px-3 py-2 align-top">{entry.confidence}</td>
                      <td className="px-3 py-2 align-top">{entry.latency_ms.toFixed(2)} ms</td>
                      <td className="px-3 py-2 align-top">{entry.cited_sources_count}</td>
                      <td className="px-3 py-2 align-top">
                        {entry.feedback === true
                          ? "helpful"
                          : entry.feedback === false
                            ? "not helpful"
                            : "pending"}
                      </td>
                      <td className="px-3 py-2 align-top">{entry.model_used || "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Models and Prompt</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="generator-preset">Generator model</Label>
              <Select
                id="generator-preset"
                value={generatorPreset}
                onChange={(event) => setGeneratorPreset(event.target.value)}
              >
                {generatorModelPresets.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </Select>
              {generatorPreset === "custom" ? (
                <Input
                  placeholder="Enter generator model name"
                  value={customGeneratorModel}
                  onChange={(event) => setCustomGeneratorModel(event.target.value)}
                />
              ) : null}
            </div>

            <div className="space-y-2">
              <Label htmlFor="reranker-preset">Reranker model</Label>
              <Select
                id="reranker-preset"
                value={rerankerPreset}
                onChange={(event) => setRerankerPreset(event.target.value)}
              >
                {rerankerModelPresets.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </Select>
              {rerankerPreset === "custom" ? (
                <Input
                  placeholder="Enter reranker model name"
                  value={customRerankerModel}
                  onChange={(event) => setCustomRerankerModel(event.target.value)}
                />
              ) : null}
            </div>

            <div className="space-y-2">
              <Label htmlFor="system-prompt">System prompt</Label>
              <Textarea
                id="system-prompt"
                rows={8}
                value={systemPrompt}
                onChange={(event) => setSystemPrompt(event.target.value)}
              />
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={() => void onSaveSettings()} disabled={isSavingSettings}>
                {isSavingSettings ? "Saving..." : "Save settings"}
              </Button>
              {settings ? (
                <Badge>
                  Embeddings: {settings.embedding_provider}/{settings.embedding_model}
                </Badge>
              ) : null}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Embeddings and Files</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="pdf-upload">Upload PDF</Label>
              <Input
                id="pdf-upload"
                type="file"
                accept="application/pdf"
                onChange={(event: ChangeEvent<HTMLInputElement>) => {
                  setFile(event.target.files?.[0] || null);
                }}
              />
              <Button onClick={() => void onUploadFile()} disabled={isUploading || !file} className="gap-1.5">
                <Upload className="h-4 w-4" />
                {isUploading ? "Uploading..." : "Upload document"}
              </Button>
            </div>

            <div className="space-y-2">
              <Label>Indexed documents</Label>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {documents.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No documents indexed.</p>
                ) : (
                  documents.map((document) => {
                    const canDelete = document.source.toLowerCase().endsWith(".pdf");
                    return (
                      <div
                        key={document.source}
                        className="flex items-center justify-between gap-3 rounded-md border border-border p-2"
                      >
                        <div>
                          <p className="text-sm font-medium">{document.title}</p>
                          <p className="text-xs text-muted-foreground">
                            {document.chunk_count} chunks • {document.source}
                          </p>
                        </div>
                        <Button
                          variant={canDelete ? "danger" : "ghost"}
                          disabled={!canDelete}
                          onClick={() => void onDeleteDocument(document.source)}
                          title={canDelete ? "Delete source PDF" : "Managed via document sync"}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <CardTitle>Documents Sync</CardTitle>
            <Button onClick={() => void onTriggerJob("documents/sync")}>
              Start documents sync
            </Button>
          </CardHeader>
          <CardContent>{renderJobStatus(documentsSyncStatus)}</CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <CardTitle>Reindex Vector Store</CardTitle>
            <Button onClick={() => void onTriggerJob("reindex")}>Start reindex</Button>
          </CardHeader>
          <CardContent className="space-y-2">
            {renderJobStatus(reindexStatus)}
            <p className="text-xs text-muted-foreground">
              Run reindex after ingestion changes to refresh page metadata used in citation links.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <CardTitle>News Index</CardTitle>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => void onTriggerNewsJob("bootstrap")}>
                Bootstrap (4 pages)
              </Button>
              <Button onClick={() => void onTriggerNewsJob("sync")}>Sync (2 pages)</Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            {renderNewsJobStatus(newsStatus)}
            <p className="text-xs text-muted-foreground">
              Automatic sync runs every 6 hours. Manual bootstrap creates the initial news index.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Usage Overview</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          {usageStats ? (
            <>
              <p>
                Total queries (last {usageStats.window_days} days):{" "}
                <span className="font-medium text-foreground">{usageStats.total_queries}</span>
              </p>
              <p>
                Average latency:{" "}
                <span className="font-medium text-foreground">{usageStats.avg_latency_ms.toFixed(2)} ms</span>
              </p>
              <p>
                Citation presence rate:{" "}
                <span className="font-medium text-foreground">
                  {formatRate(usageStats.citation_presence_rate)}
                </span>
              </p>
              <p>
                Non-empty answer rate:{" "}
                <span className="font-medium text-foreground">
                  {formatRate(usageStats.non_empty_answer_rate)}
                </span>
              </p>
              <p>
                Source mix (PDF / News):{" "}
                <span className="font-medium text-foreground">
                  {usageStats.source_mix_pdf_vs_news.pdf} / {usageStats.source_mix_pdf_vs_news.news}
                </span>
              </p>
              <p>
                Feedback coverage:{" "}
                <span className="font-medium text-foreground">
                  {formatRate(usageStats.feedback_coverage_rate ?? 0)}
                </span>
              </p>
              <p>
                Helpful feedback rate:{" "}
                <span className="font-medium text-foreground">
                  {formatRate(usageStats.helpful_feedback_rate ?? 0)}
                </span>
              </p>
              <p>
                Confidence distribution:{" "}
                <span className="font-medium text-foreground">
                  high {usageStats.confidence_distribution.high || 0}, medium{" "}
                  {usageStats.confidence_distribution.medium || 0}, low{" "}
                  {usageStats.confidence_distribution.low || 0}, unknown{" "}
                  {usageStats.confidence_distribution.unknown || 0}
                </span>
              </p>
              <p className="text-xs text-muted-foreground">
                Generated: {formatTimestamp(usageStats.generated_at_utc)}
              </p>
            </>
          ) : (
            <p className="text-muted-foreground">No usage statistics available yet.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
