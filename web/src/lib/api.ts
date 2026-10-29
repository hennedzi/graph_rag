// "__PORT_8200__" подменяется прокси при деплое превью; локально UI отдаёт тот же FastAPI → относительные пути.
const API_BASE = "__PORT_8200__".startsWith("__") ? "" : "__PORT_8200__";

export const url = (path: string) => `${API_BASE}${path}`;

async function ok<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      detail = JSON.parse(body).detail ?? body;
    } catch {
      /* не JSON — показываем как есть */
    }
    throw new Error(`${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return res.json();
}

export const getJson = <T,>(path: string) => fetch(url(path)).then((r) => ok<T>(r));
export const postJson = <T,>(path: string, body: unknown) =>
  fetch(url(path), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(
    (r) => ok<T>(r),
  );
export const upload = (files: File[]) => {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  return fetch(url("/etl/documents"), { method: "POST", body: fd }).then((r) => ok<Job[]>(r));
};

// ─── Типы ответов API ───────────────────────────────────────────
export type Passage = { id: string; doc: string; text: string; score: number };
export type SearchResult = { entities: string[]; relations: string[]; passages: Passage[]; timings_ms: Timings };
export type Unsupported = { claim?: string; reason?: string };
export type AskResult = { answer: string; sources: Passage[]; unsupported: Unsupported[]; timings_ms: Timings };
export type Timings = Record<string, number>;
export type Edge = { src: string; dst: string; text: string; handles: string[] };
export type Claim = { text: string; sources: string[] };
export type Citation = { doc_id: string; doc: string; text: string };
export type HarnessResult = {
  query: string;
  answer: string;
  claims: Claim[];
  citations: Record<string, Citation>;
  unsupported: Unsupported[];
  steps: number;
  fix_rounds: number;
  context_chars: number;
  trace: { tool: string; args: Record<string, unknown>; chars: number }[];
  thoughts: { step: number; text: string }[];
  graph: Edge[];
  timings_ms: Timings;
};
export type HarnessEvent =
  | { type: "start"; query: string; budget: number; max_steps: number }
  | { type: "thought"; step: number; text: string }
  | { type: "call"; step: number; id: string; tool: string; args: Record<string, unknown> }
  | {
      type: "result";
      step: number;
      id: string;
      tool: string;
      status: string;
      out: string;
      chars: number;
      ms: number;
      total_chars: number;
      edges: Edge[];
    }
  | { type: "done"; result: HarnessResult }
  | { type: "error"; detail: string };
export type Job = {
  job_id: string;
  doc: string;
  status: "queued" | "running" | "done" | "failed";
  chunks: number;
  triples?: number;
  rows_written?: number;
  error?: string;
  timings_ms?: Timings;
};
export type Counts = Record<string, number>;
export type RunSummary = {
  id: string;
  mode: "ask" | "harness" | string;
  query: string;
  timestamp: string;
  total_ms: number;
  timings_ms: Timings;
  steps: number | null;
  context_chars: number | null;
  unsupported: number;
  answer: string;
  tools: string[];
  top_score: number | null;
};
export type Metric = {
  name: string;
  type: string;
  help: string;
  samples: { name: string; labels: Record<string, string>; value: number }[];
};

// ─── SSE поверх POST (EventSource умеет только GET) ─────────────
export async function streamHarness(query: string, onEvent: (e: HarnessEvent) => void, signal: AbortSignal) {
  const res = await fetch(url("/harness/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });
  if (!res.ok || !res.body) await ok(res);
  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const data = frame
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data));
    }
  }
}

// ─── Хелперы для метрик ─────────────────────────────────────────
export const samples = (m: Metric[] | undefined, name: string) => m?.find((x) => x.name === name)?.samples ?? [];

/** Сумма counter'а (только *_total), сгруппированная по метке. */
export function byLabel(m: Metric[] | undefined, name: string, label: string) {
  const out: Record<string, number> = {};
  for (const s of samples(m, name)) if (s.name.endsWith("_total")) out[s.labels[label]] = (out[s.labels[label]] ?? 0) + s.value;
  return out;
}

/** Кумулятивные бакеты гистограммы → счётчики по интервалам. */
export function buckets(m: Metric[] | undefined, name: string) {
  const bs = samples(m, name).filter((s) => s.name.endsWith("_bucket"));
  return bs.map((s, i) => ({
    le: s.labels.le === "+Inf" ? "∞" : String(Number(s.labels.le)),
    count: s.value - (i ? bs[i - 1].value : 0),
  }));
}

export const fmtMs = (ms?: number | null) =>
  ms == null ? "—" : ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} с` : `${Math.round(ms)} мс`;
export const fmtNum = (n?: number | null) => (n == null ? "—" : n.toLocaleString("ru-RU"));
