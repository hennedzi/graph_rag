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
export const del = <T,>(path: string) => fetch(url(path), { method: "DELETE" }).then((r) => ok<T>(r));
export const upload = (files: File[]) => {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  return fetch(url("/etl/documents"), { method: "POST", body: fd }).then((r) => ok<Job[]>(r));
};

// ─── Типы ответов API ───────────────────────────────────────────
export type Passage = { id: string; doc: string; text: string; score: number };
export type SearchResult = { entities: string[]; relations: string[]; passages: Passage[]; timings_ms: Timings };
export type Unsupported = { claim?: string; reason?: string };
/** Итог запуска с backend (FUTURE §4); в старых runs поля нет — диагностика неполная. */
export type Outcome = "answered" | "partial" | "no_evidence" | "verification_failed" | "budget_exceeded" | "cancelled" | "dependency_error";
export type AskResult = { answer: string; sources: Passage[]; unsupported: Unsupported[]; timings_ms: Timings; outcome?: Outcome };
export type Timings = Record<string, number>;
export type Edge = { src: string; dst: string; text: string; handles: string[] };
export type Claim = { text: string; sources: string[] };
export type Citation = { doc_id: string; doc: string; text: string };
export type HarnessResult = {
  query: string;
  /** agent — цикл с инструментами; chat — реплика без поиска; rewrite — пересказ прошлого ответа */
  route?: "agent" | "chat" | "rewrite" | "rewrite_fallback";
  answer: string;
  claims: Claim[];
  citations: Record<string, Citation>;
  unsupported: Unsupported[];
  outcome?: Outcome;
  outcome_note?: string;
  steps: number;
  fix_rounds: number;
  context_chars: number;
  trace: { tool: string; args: Record<string, unknown>; chars: number }[];
  thoughts: { step: number; text: string }[];
  graph: Edge[];
  timings_ms: Timings;
  session_id: string;
  turn: number;
  turn_chars: number;
  context_tokens: number;
  tokens_spent: number;
  budget: number;
  budget_left: number;
};
export type HarnessEvent =
  | { type: "start"; query: string; budget: number; max_steps: number; session_id: string; turn: number; chars: number; ctx_tokens: number; edges: Edge[] }
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
      ctx_tokens: number;
      edges: Edge[];
      speculative?: boolean;
  }
  | { type: "fast_path_taken"; top_score: number }
  | { type: "token"; text: string }
  | { type: "answer_finished"; text: string; citations: Record<string, Citation> }
  | { type: "done"; result: HarnessResult }
  | { type: "error"; detail: string; session?: "lost" };
export type Job = {
  job_id: string;
  doc: string;
  status: "queued" | "running" | "done" | "failed";
  chunks: number;
  triples?: number;
  rows_written?: number;
  replaced?: number;
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
export async function streamHarness(query: string, onEvent: (e: HarnessEvent) => void, signal: AbortSignal, sessionId?: string) {
  const res = await fetch(url("/harness/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: sessionId ?? null }),
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

// ─── Граф: PPR и пути (/graph/*) ────────────────────────────────
export type GraphStats = { nodes: number; edges: number; avg_degree: number; components: number; largest_component: number; aliases?: number };
export type GraphInfo = { stats: GraphStats; stats_raw: GraphStats; aliases: Record<string, string> };
export type GraphExplore = {
  seeds: string[];
  ranking: { entity: string; score: number; passages: number; aliases: string[] }[];
  edges: Edge[];
  paths: Edge[][];
  passages: Passage[];
  timings_ms: Timings;
};

// ─── База знаний: документы (/etl/documents) ────────────────────
export type DocSummary = {
  doc: string;
  passages: number;
  entities: number;
  relations: number;
  /** сущности, которые встречаются и в других документах — «мосты» графа */
  shared: number;
  chars: number;
  uploaded_at?: string;
  bytes?: number;
  triples?: number;
};
export type DocDetail = {
  doc: string;
  meta: { uploaded_at?: string; bytes?: number; chunks?: number; triples?: number };
  passages: { id: string; text: string; entities: string[]; relations: number }[];
  entities: { name: string; here: number; total: number }[];
  relations: { src: string; dst: string; text: string; passages: string[] }[];
};
export type Dropped = { passages: number; entities: number; relations: number };
