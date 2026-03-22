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
  getJobStatus,
  listAdminDocuments,
  triggerJob,
  updateAdminSettings,
  uploadDocument,
} from "@/lib/api";
import type { DocumentListItem, JobStatusResponse, RuntimeSettings } from "@/types/api";

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

export function AdminPage() {
  const [settings, setSettings] = useState<RuntimeSettings | null>(null);
  const [generatorPreset, setGeneratorPreset] = useState("google/gemini-3-flash-preview");
  const [rerankerPreset, setRerankerPreset] = useState("google/gemini-3-flash-preview");
  const [customGeneratorModel, setCustomGeneratorModel] = useState("");
  const [customRerankerModel, setCustomRerankerModel] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");

  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [file, setFile] = useState<File | null>(null);

  const [scrapeStatus, setScrapeStatus] = useState<JobStatusResponse>({ status: "idle" });
  const [reindexStatus, setReindexStatus] = useState<JobStatusResponse>({ status: "idle" });

  const [isLoading, setIsLoading] = useState(true);
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isPolling = useMemo(
    () => [scrapeStatus, reindexStatus].some((status) => ["queued", "running"].includes(status.status)),
    [scrapeStatus, reindexStatus],
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
    const [nextScrapeStatus, nextReindexStatus] = await Promise.all([
      getJobStatus("scrape"),
      getJobStatus("reindex"),
    ]);
    setScrapeStatus(nextScrapeStatus);
    setReindexStatus(nextReindexStatus);
  };

  const reloadAll = async () => {
    setError(null);
    setIsLoading(true);
    try {
      await Promise.all([loadSettings(), loadDocuments(), loadStatuses()]);
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

  const onTriggerJob = async (job: "scrape" | "reindex") => {
    setError(null);
    try {
      const status = await triggerJob(job);
      if (job === "scrape") {
        setScrapeStatus(status);
      } else {
        setReindexStatus(status);
      }
      await loadStatuses();
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : `Failed to start ${job}.`;
      setError(message);
    }
  };

  const renderJobStatus = (status: JobStatusResponse) => {
    const result = (status.result || {}) as Record<string, unknown>;
    const discovered = result.discovered_count as number | undefined;
    const downloaded = result.downloaded_count as number | undefined;
    const newsSaved = result.news_saved_count as number | undefined;

    return (
      <div className="space-y-1 text-sm text-muted-foreground">
        <p>
          Status: <span className="font-medium text-foreground">{status.status}</span>
        </p>
        {typeof status.vector_count === "number" ? <p>Vector count: {status.vector_count}</p> : null}
        {typeof discovered === "number" ? <p>Discovered links: {discovered}</p> : null}
        {typeof downloaded === "number" ? <p>Downloaded PDFs: {downloaded}</p> : null}
        {typeof newsSaved === "number" ? <p>Indexed news articles: {newsSaved}</p> : null}
        {status.error ? <p className="text-red-700">Error: {status.error}</p> : null}
      </div>
    );
  };

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
                          title={canDelete ? "Delete source PDF" : "Managed via scraper"}
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

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <CardTitle>Scrape ELTE IK</CardTitle>
            <Button onClick={() => void onTriggerJob("scrape")}>Start scrape</Button>
          </CardHeader>
          <CardContent>{renderJobStatus(scrapeStatus)}</CardContent>
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
      </div>
    </div>
  );
}
