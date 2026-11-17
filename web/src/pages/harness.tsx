import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Brain, Check, ChevronDown, ChevronLeft, ChevronRight, Copy, Gauge, History, Loader2, MessagesSquare, Network, Plus, ShieldAlert, ShieldCheck, Sparkles, Footprints, XCircle } from "lucide-react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { BotMsg, ChatPane, Composer, ComposerChip, UserMsg, useStickToBottom } from "@/components/chat";
import { ListItem, SectionLabel, Workspace, useStatus } from "@/components/shell";
import { Card, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { GraphView } from "@/components/graph-view";
import { Claims, OUTCOME_TITLE, SourceBadge, Verdict } from "@/components/answer-card";
import { STATUS, toneClass, toolMeta } from "@/components/tools";
import { fmtMs, fmtNum, getJson, postJson, streamHarness, type Edge, type HarnessEvent, type HarnessResult, type RunSummary } from "@/lib/api";
import { cn } from "@/lib/utils";

export const EXAMPLES = [
  "Кто был учителем Леонарда Эйлера?",
  "Как связаны Иоганн Бернулли и Леонард Эйлер?",
  "Какое правило открыл учитель Леонарда Эйлера?",
  "Кто пригласил Эйлера в Петербург?",
];

type ToolStepT = {
  kind: "tool";
  key: string;
  step: number;
  tool: string;
  args: Record<string, unknown>;
  res?: Extract<HarnessEvent, { type: "result" }>;
};
type Step = { kind: "thought"; key: string; step: number; text: string } | ToolStepT;

type State = {
  query: string;
  running: boolean;
  budget: number;
  maxSteps: number;
  chars: number;
  ctx: number; // токены окна LLM (usage.prompt_tokens + оценка новых выводов)
  steps: Step[];
  edges: Edge[];
  result?: HarnessResult;
  error?: string;
  started?: number;
  finished?: number;
  dialog: number; // номер диалога в ленте: новый диалог = новая серверная сессия
  turn: number;
  chars0: number; // контекст диалога до этого вопроса
  history?: boolean; // открыт из runs/: выводы инструментов там не хранятся
  stopping?: boolean; // остановка запрошена, сервер дорабатывает текущий вызов модели
};

const initial: State = { query: "", running: false, budget: 32000, maxSteps: 8, chars: 0, ctx: 0, steps: [], edges: [], dialog: 0, turn: 1, chars0: 0 };

function reduce(s: State, e: HarnessEvent | { type: "reset"; query: string; dialog: number } | { type: "abort" } | { type: "stopping" }): State {
  switch (e.type) {
    case "stopping":
      return { ...s, stopping: true };
    case "reset":
      return { ...initial, query: e.query, dialog: e.dialog, running: true, started: Date.now() };
    case "start":
      // в продолжении диалога сервер присылает уже накопленный контекст и подграф
      return { ...s, budget: e.budget, maxSteps: e.max_steps, turn: e.turn, chars: e.chars, ctx: e.ctx_tokens, chars0: e.chars, edges: e.edges };
    case "thought":
      return { ...s, steps: [...s.steps, { kind: "thought", key: `t${s.steps.length}`, step: e.step, text: e.text }] };
    case "call":
      return { ...s, steps: [...s.steps, { kind: "tool", key: `${e.step}:${e.id}:${s.steps.length}`, step: e.step, tool: e.tool, args: e.args }] };
    case "result": {
      const steps = [...s.steps];
      for (let i = steps.length - 1; i >= 0; i--) {
        const st = steps[i];
        if (st.kind === "tool" && !st.res && st.step === e.step) {
          steps[i] = { ...st, res: e };
          break;
        }
      }
      return { ...s, steps, chars: e.total_chars, ctx: e.ctx_tokens, edges: e.edges.length ? [...s.edges, ...e.edges] : s.edges };
    }
    case "done":
      return { ...s, running: false, result: e.result, chars: e.result.context_chars, ctx: e.result.context_tokens, budget: e.result.budget, finished: Date.now() };
    case "error":
      return { ...s, running: false, error: e.detail, finished: Date.now() };
    case "abort":
      return { ...s, running: false, error: s.result ? undefined : "Остановлено вручную", finished: Date.now() };
  }
}

type RunDetail = HarnessResult & { total_ms?: number };

// Запуск из runs/*.json → ход в ленте (только чтение): выводы инструментов не сохраняются, остальное есть.
function fromRun(r: RunDetail, dialog: number): State {
  const steps: Step[] = (r.trace ?? []).map((t, i) => ({
    kind: "tool",
    key: `h${i}`,
    step: i + 1,
    tool: t.tool,
    args: t.args ?? {},
    res: { type: "result", step: i + 1, id: "", tool: t.tool, status: "ok", out: "", chars: t.chars, ms: 0, total_chars: 0, ctx_tokens: 0, edges: [] },
  }));
  return {
    ...initial,
    query: r.query,
    dialog,
    history: true,
    turn: r.turn ?? 1,
    budget: r.budget ?? 32000,
    chars: r.context_chars ?? 0,
    ctx: r.context_tokens ?? 0,
    steps,
    edges: r.graph ?? [],
    result: { ...r, claims: r.claims ?? [], citations: r.citations ?? {}, unsupported: r.unsupported ?? [], timings_ms: { ...(r.timings_ms ?? {}), total: r.total_ms ?? 0 } },
  };
}

function useNow(active: boolean) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(t);
  }, [active]);
  return now;
}

const pctTone = (p: number) => (p > 75 ? "bg-[hsl(var(--chart-5))]" : p > 50 ? "bg-[hsl(var(--chart-3))]" : "bg-primary");

function Meter({ icon: Icon, label, value, sub, pct, tone }: { icon: typeof Gauge; label: string; value: string; sub?: string; pct?: number; tone?: string }) {
  return (
    <div className="space-y-1.5 rounded-lg border bg-card p-2.5 shadow-xs">
      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <Icon className="h-3.5 w-3.5" /> {label}
      </div>
      <div className="flex flex-wrap items-baseline gap-x-1.5">
        <span className="whitespace-nowrap font-mono text-base font-semibold tabular-nums">{value}</span>
        {sub && <span className="whitespace-nowrap font-mono text-[11px] text-muted-foreground">{sub}</span>}
      </div>
      {pct != null && (
        <div className="h-1 overflow-hidden rounded-full bg-muted">
          <motion.div className={cn("h-full rounded-full", tone ?? "bg-primary")} animate={{ width: `${Math.min(pct, 100)}%` }} transition={{ duration: 0.4 }} />
        </div>
      )}
    </div>
  );
}

const argText = (v: unknown) => (typeof v === "string" ? `"${v}"` : JSON.stringify(v));
const inlineArgs = (tool: string, args: Record<string, unknown>) =>
  tool === "answer" && Array.isArray(args.claims)
    ? `${args.claims.length} утвержд.${args.no_answer ? " · no_answer" : ""}`
    : Object.entries(args)
        .map(([k, v]) => `${k}=${argText(v)}`)
        .join(" ");

// ─── Лента: строки шагов в духе Claude Code ─────────────────────
function TraceLine({ s, active, onClick }: { s: ToolStepT; active: boolean; onClick: () => void }) {
  const meta = toolMeta(s.tool);
  const status = s.res && s.res.status !== "ok" ? STATUS[s.res.status] ?? { label: s.res.status, tone: "warn" as const } : null;
  const lines = s.res?.out ? s.res.out.split("\n").filter(Boolean) : [];
  const first = s.tool === "answer" ? (s.res?.out === "принято" ? "принято" : lines[0]) : lines[0];
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn("w-full rounded-md px-2 py-1 text-left font-mono hover:bg-secondary/60", active && "bg-secondary ring-1 ring-primary/30")}
      data-testid={`step-tool-${s.key}`}
    >
      <div className="flex items-baseline gap-2 text-[12.5px]">
        {s.res ? (
          <meta.icon className="h-3.5 w-3.5 shrink-0 translate-y-0.5" style={{ color: `hsl(${meta.color})` }} />
        ) : (
          <Loader2 className="h-3 w-3 shrink-0 translate-y-0.5 animate-spin text-primary" />
        )}
        <span className="shrink-0 font-semibold" style={{ color: `hsl(${meta.color})` }}>
          {meta.label}
        </span>
        <span className="min-w-0 truncate text-muted-foreground">{inlineArgs(s.tool, s.args)}</span>
        <span className="ml-auto flex shrink-0 items-center gap-2 text-[10.5px] text-muted-foreground">
          {status && <span className={cn("rounded border px-1 py-px font-sans font-medium", toneClass[status.tone])}>{status.label}</span>}
          {s.res && (s.res.ms ? `${fmtMs(s.res.ms)} · ` : "") + `${fmtNum(s.res.chars)} симв.`}
        </span>
      </div>
      {(first || !s.res) && (
        <div className="flex gap-2 pl-1 text-[11.5px] text-muted-foreground">
          <span className="select-none">⎿</span>
          <span className="min-w-0 truncate">{s.res ? first : "выполняется…"}</span>
          {lines.length > 1 && <span className="shrink-0 opacity-70">+{lines.length - 1} стр.</span>}
        </div>
      )}
    </button>
  );
}

function ThoughtLine({ s }: { s: Extract<Step, { kind: "thought" }> }) {
  const [open, setOpen] = useState(false);
  return (
    <button type="button" onClick={() => setOpen(!open)} className="flex w-full gap-2 rounded-md px-2 py-1 text-left hover:bg-secondary/40" data-testid="step-thought">
      <Brain className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
      <span className={cn("text-[12.5px] leading-relaxed text-muted-foreground", !open && "line-clamp-2")}>{s.text}</span>
    </button>
  );
}

// «Ход работы» как Chain of Thought: пока агент работает — раскрыт, после ответа сворачивается в цепочку иконок.
function Trace({ s, sel, onPick }: { s: State; sel?: string; onPick: (key: string) => void }) {
  const [open, setOpen] = useState<boolean>();
  const tools = s.steps.filter((x): x is ToolStepT => x.kind === "tool");
  if (!s.steps.length && !s.running) return null;
  const shown = open ?? s.running;
  const total = s.result?.timings_ms?.total;
  return (
    <div data-testid="trace">
      <button type="button" onClick={() => setOpen(!shown)} className="group flex w-full items-center gap-2 text-left text-[13px] text-muted-foreground hover:text-foreground" data-testid="button-trace">
        {s.running ? <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" /> : <Brain className="h-3.5 w-3.5" />}
        <span className="font-medium text-foreground/90">{s.running ? "Агент работает" : "Ход работы"}</span>
        <span className="whitespace-nowrap font-mono text-[11px]">
          {tools.length} шагов{total ? ` · ${fmtMs(total)}` : ""}
        </span>
        <span className="flex min-w-0 items-center gap-0.5 overflow-hidden">
          {tools.map((t, i) => {
            const m = toolMeta(t.tool);
            return (
              <span key={t.key} className="flex items-center gap-0.5">
                {i > 0 && <span className="text-[10px] opacity-40">›</span>}
                <span className="flex h-5 w-5 items-center justify-center rounded-full" style={{ background: `hsl(${m.color} / 0.12)` }} title={m.label}>
                  <m.icon className="h-3 w-3" style={{ color: `hsl(${m.color})` }} />
                </span>
              </span>
            );
          })}
        </span>
        <ChevronDown className={cn("ml-auto h-4 w-4 shrink-0 transition-transform", shown && "rotate-180")} />
      </button>
      {shown && (
        <div className="ml-[6px] mt-2 space-y-0.5 border-l pl-3">
          {s.steps.map((st) => (st.kind === "thought" ? <ThoughtLine key={st.key} s={st} /> : <TraceLine key={st.key} s={st} active={st.key === sel} onClick={() => onPick(st.key)} />))}
          {s.running && (
            <div className="flex items-center gap-2 px-2 py-1 font-mono text-[12px] text-muted-foreground">
              <span className="h-2 w-2 animate-pulse rounded-full bg-primary" /> агент думает…
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function CopyBtn({ text }: { text: string }) {
  const [ok, setOk] = useState(false);
  return (
    <Button
      size="icon"
      variant="ghost"
      className="h-7 w-7 text-muted-foreground"
      aria-label="Скопировать ответ"
      onClick={() => navigator.clipboard.writeText(text).then(() => (setOk(true), setTimeout(() => setOk(false), 1200)))}
      data-testid="button-copy"
    >
      {ok ? <Check className="h-3.5 w-3.5 text-primary" /> : <Copy className="h-3.5 w-3.5" />}
    </Button>
  );
}

const ROUTE_NOTE: Record<string, string> = {
  rewrite: "пересказ прошлого ответа без поиска",
  rewrite_fallback: "пересказ не прошёл проверку источниками — показан прошлый ответ",
};

function Answer({ s }: { s: State }) {
  if (!s.result && !s.error) return null;
  if (s.result?.route === "chat")
    return (
      <p className="pt-0.5 text-sm leading-relaxed" data-testid="text-chat-reply">
        {s.result.answer}
        <span className="ml-2 font-mono text-[11px] text-muted-foreground">без поиска · 0 ток.</span>
      </p>
    );
  return (
    <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}>
      <Card className={cn("rounded-xl shadow-xs", s.error && "border-destructive/50")} data-testid="card-answer">
        <CardContent className="space-y-3.5 p-4">
          {s.error && <p className="font-mono text-sm text-destructive">{s.error}</p>}
          {s.result && (
            <>
              <Claims claims={s.result.claims} answer={s.result.answer} citations={s.result.citations} />
              <Verdict unsupported={s.result.unsupported} outcome={s.result.outcome} note={s.result.outcome_note} />
              <div className="flex items-center gap-2 border-t pt-2.5">
              <p className="min-w-0 flex-1 font-mono text-[11px] text-muted-foreground">
                {ROUTE_NOTE[s.result.route ?? ""] ? `${ROUTE_NOTE[s.result.route!]} · ` : `${s.result.steps} шагов · `}окно {fmtNum(s.result.context_tokens)} ток. · потрачено {fmtNum(s.result.tokens_spent)} ток.
                {s.result.fix_rounds > 0 && ` · исправлений ${s.result.fix_rounds}`}
                {s.history && " · из истории"}
              </p>
              <CopyBtn text={s.result.answer} />
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ─── Инспектор ──────────────────────────────────────────────────
function view(s: State, now: number) {
  const tools = s.steps.filter((x): x is ToolStepT => x.kind === "tool");
  const focus = tools.flatMap((t) => {
    const first = t.res?.out.split("\n")[0] ?? "";
    if (t.tool === "neighbors" && first.startsWith("сущность:")) return [first.slice(10).trim()];
    if (t.tool === "rank_by_ppr" && first.startsWith("seed:")) return first.slice(5).split(",").map((x) => x.trim());
    const m = t.tool === "find_paths" && first.match(/^пути (.+) → (.+):$/);
    return m ? [m[1], m[2]] : [];
  });
  // экономия контекста: сколько пассажей агент увидел превью vs сколько реально прочитал
  const seen = new Set(tools.flatMap((t) => [...(t.res?.out ?? "").matchAll(/\[(P\d+)\]/g)].map((m) => m[1])));
  const read = new Map<string, string>();
  for (const t of tools)
    if (t.tool === "read_passage" && t.res)
      for (const block of t.res.out.split("\n\n")) {
        const m = block.match(/^\[(P\d+)\] ([\s\S]+)$/);
        if (m && !m[2].startsWith("уже прочитан") && !m[2].startsWith("неизвестный хэндл")) read.set(m[1], m[2]);
      }
  if (s.history) for (const t of tools) if (t.tool === "read_passage") for (const h of (t.args.ids as string[]) ?? []) if (!read.has(h.toUpperCase())) read.set(h.toUpperCase(), "");
  return {
    tools,
    focus,
    seen,
    read,
    stepNo: Math.max(0, ...s.steps.map((x) => x.step)),
    elapsed: s.result?.timings_ms?.total ?? (s.started ? (s.finished ?? now) - s.started : 0),
    pct: (s.ctx / s.budget) * 100,
  };
}

function Breakdown({ t }: { t: Record<string, number> }) {
  const tools = Object.entries(t).filter(([k]) => k.startsWith("tool_"));
  const parts = [
    { k: "LLM агента", v: t.harness_llm ?? 0, c: "hsl(var(--chart-2))" },
    { k: "Инструменты", v: tools.reduce((a, [, v]) => a + v, 0), c: "hsl(var(--chart-1))" },
    { k: "Верификация", v: t.harness_verify ?? 0, c: "hsl(var(--chart-3))" },
  ];
  const sum = parts.reduce((a, p) => a + p.v, 0) || 1;
  return (
    <div className="space-y-2">
      <div className="flex h-2 overflow-hidden rounded-full bg-muted">
        {parts.map((p) => (
          <div key={p.k} style={{ width: `${(p.v / sum) * 100}%`, background: p.c }} />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {parts.map((p) => (
          <span key={p.k} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: p.c }} />
            {p.k} <span className="font-mono text-foreground">{fmtMs(p.v)}</span>
          </span>
        ))}
      </div>
      <div className="space-y-1 pt-1">
        {tools
          .sort((a, b) => b[1] - a[1])
          .map(([k, v]) => (
            <div key={k} className="flex justify-between font-mono text-[11px] text-muted-foreground">
              <span>{k.slice(5)}</span>
              <span className="text-foreground">{fmtMs(v)}</span>
            </div>
          ))}
      </div>
    </div>
  );
}

function StepDetail({ s, steps, onPick }: { s?: ToolStepT; steps: ToolStepT[]; onPick: (k: string) => void }) {
  if (!s) return <Empty icon={Footprints} title="Шагов ещё нет" text="Кликните по шагу в «Ходе работы» — здесь будут аргументы и полный вывод инструмента." />;
  const meta = toolMeta(s.tool);
  const i = steps.indexOf(s);
  const status = s.res ? STATUS[s.res.status] ?? { label: s.res.status, tone: "warn" as const } : null;
  return (
    <div className="space-y-3 p-3" data-testid="step-detail">
      <div className="flex items-center gap-2">
        <meta.icon className="h-4 w-4" style={{ color: `hsl(${meta.color})` }} />
        <span className="font-mono text-sm font-semibold" style={{ color: `hsl(${meta.color})` }}>
          {meta.label}
        </span>
        {status && <span className={cn("rounded border px-1.5 py-px text-[11px]", toneClass[status.tone])}>{status.label}</span>}
        <span className="ml-auto flex items-center gap-1">
          <Button size="icon" variant="ghost" className="h-6 w-6" disabled={i <= 0} onClick={() => onPick(steps[i - 1].key)} aria-label="Предыдущий шаг">
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="font-mono text-[11px] text-muted-foreground">
            {i + 1}/{steps.length}
          </span>
          <Button size="icon" variant="ghost" className="h-6 w-6" disabled={i >= steps.length - 1} onClick={() => onPick(steps[i + 1].key)} aria-label="Следующий шаг">
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </span>
      </div>
      <p className="text-xs text-muted-foreground">
        шаг {s.step} · {meta.hint}
        {s.res && ` · ${fmtMs(s.res.ms)} · ${fmtNum(s.res.chars)} симв.`}
      </p>
      <div>
        <p className="mb-1 text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground">Аргументы</p>
        <pre className="whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-2 font-mono text-[11.5px]">{JSON.stringify(s.args, null, 2)}</pre>
      </div>
      <div>
        <p className="mb-1 text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground">Вывод для агента</p>
        <pre className="whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-2 font-mono text-[11.5px] leading-relaxed text-muted-foreground">
          {s.res ? s.res.out || "(в истории запусков вывод не сохраняется)" : "выполняется…"}
        </pre>
      </div>
    </div>
  );
}

function Sources({ s, v }: { s: State; v: ReturnType<typeof view> }) {
  const cited = new Set((s.result?.claims ?? []).flatMap((c) => c.sources));
  const rows = [...v.read.entries()];
  return (
    <div className="space-y-3 p-3">
      <p className="text-xs text-muted-foreground">
        Увидел в превью <span className="font-mono text-foreground">{v.seen.size}</span>, прочитал целиком <span className="font-mono text-foreground">{v.read.size}</span> — полный текст только там, где он нужен для ответа.
      </p>
      {rows.map(([h, text]) => (
        <div key={h} className={cn("space-y-1 rounded-lg border bg-card p-2.5 shadow-xs", cited.has(h) && "border-primary/40")} data-testid={`source-${h}`}>
          <div className="flex items-center gap-2 font-mono text-[11px]">
            <SourceBadge h={h} cite={s.result?.citations[h] ?? { doc_id: "", doc: "", text }} />
            <span className="truncate text-muted-foreground">{s.result?.citations[h]?.doc}</span>
            {cited.has(h) && <span className="ml-auto text-primary">в ответе</span>}
          </div>
          {(text || s.result?.citations[h]?.text) && <p className="line-clamp-4 text-xs leading-relaxed text-muted-foreground">{text || s.result?.citations[h]?.text}</p>}
        </div>
      ))}
      {!rows.length && <Empty icon={Network} title="Пока ничего не прочитано" text="Агент открывает полный текст пассажа инструментом read_passage." />}
    </div>
  );
}

const TAB = "h-7 gap-1 rounded-md px-0 text-xs data-[state=active]:bg-background data-[state=active]:shadow-sm focus-visible:ring-1 focus-visible:ring-offset-0";
const Count = ({ n }: { n?: number }) => (n ? <span className="font-mono text-[10.5px] text-muted-foreground">{n}</span> : null);

function runStatus(s: State) {
  if (s.stopping && s.running) return { icon: Loader2, t: "Останавливается: ждём текущий вызов модели", c: "text-muted-foreground", spin: true };
  if (s.running) return { icon: Loader2, t: "Агент идёт по базе знаний", c: "text-primary", spin: true };
  if (s.error) return { icon: XCircle, t: "Ошибка", c: "text-destructive" };
  if (s.result?.route === "chat") return { icon: MessagesSquare, t: "Реплика без поиска", c: "text-muted-foreground" };
  if (s.result?.unsupported.length) return { icon: ShieldAlert, t: "Есть неподтверждённое", c: "text-[hsl(var(--chart-5))]" };
  const o = s.result?.outcome;
  if (o === "answered") return { icon: ShieldCheck, t: OUTCOME_TITLE.answered, c: "text-primary" };
  return { icon: ShieldAlert, t: o ? OUTCOME_TITLE[o] ?? o : "Статус проверки не сохранён", c: "text-[hsl(var(--chart-5))]" };
}

// Шапка инспектора: что сейчас происходит и три счётчика — видны на любой вкладке.
function RunHeader({ cur, v }: { cur: State; v: ReturnType<typeof view> }) {
  const st = runStatus(cur);
  return (
    <div className="shrink-0 space-y-3 border-b p-3" data-testid="run-header">
      <div className="flex items-center gap-2">
        <span className={cn("flex h-7 w-7 items-center justify-center rounded-full", st.c)} style={{ background: "color-mix(in srgb, currentColor 12%, transparent)" }}>
          <st.icon className={cn("h-3.5 w-3.5", st.spin && "animate-spin")} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-[13px] font-medium">{st.t}</p>
          <p className="truncate text-[11px] text-muted-foreground">
            {cur.history ? "из истории запусков" : `вопрос ${cur.turn} диалога`} · {fmtMs(v.elapsed)}
          </p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <Meter icon={Footprints} label="Шаги" value={String(v.stepNo)} sub={`/ ${cur.maxSteps}`} pct={cur.maxSteps ? (v.stepNo / cur.maxSteps) * 100 : 0} tone="bg-[hsl(var(--chart-2))]" />
        <Meter icon={Gauge} label="Окно" value={fmtNum(cur.ctx)} sub={`/ ${fmtNum(cur.budget)}`} pct={v.pct} tone={pctTone(v.pct)} />
        <Meter icon={Network} label="Граф" value={String(cur.edges.length)} sub="связей" />
      </div>
    </div>
  );
}

function Empty({ icon: Icon, title, text }: { icon: typeof Gauge; title: string; text: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
      <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/10">
        <Icon className="h-5 w-5 text-primary" />
      </span>
      <p className="text-sm font-medium">{title}</p>
      <p className="max-w-[260px] text-xs text-muted-foreground">{text}</p>
    </div>
  );
}

function Inspector({ cur, v, tab, setTab, pick, sel }: { cur?: State; v?: ReturnType<typeof view>; tab: string; setTab: (t: string) => void; pick: (k: string) => void; sel?: ToolStepT }) {
  if (!cur || !v) return <Empty icon={Network} title="Инспектор запроса" text="Задайте вопрос — здесь появятся подграф, по которому шёл агент, каждый шаг и прочитанные источники." />;
  return (
    <div className="flex h-full flex-col">
      <RunHeader cur={cur} v={v} />
      <Tabs value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col">
        <TabsList className="mx-3 mt-3 grid h-8 shrink-0 grid-cols-4 rounded-lg bg-muted p-0.5">
          <TabsTrigger value="graph" className={TAB} data-testid="tab-graph">
            Граф <Count n={cur.edges.length} />
          </TabsTrigger>
          <TabsTrigger value="step" className={TAB} data-testid="tab-step">
            Шаг <Count n={v.tools.length} />
          </TabsTrigger>
          <TabsTrigger value="sources" className={TAB} data-testid="tab-sources">
            Источники <Count n={v.read.size} />
          </TabsTrigger>
          <TabsTrigger value="summary" className={TAB} data-testid="tab-summary">
            Время
          </TabsTrigger>
        </TabsList>
        <TabsContent value="graph" className="mt-0 flex min-h-0 flex-1 flex-col gap-2 p-3 data-[state=inactive]:hidden">
          {cur.edges.length ? (
            <>
              <div className="min-h-0 flex-1 overflow-hidden rounded-xl border bg-card">
                <GraphView edges={cur.edges} focus={v.focus} />
              </div>
              {!!v.focus.length && (
                <div className="flex flex-wrap items-center gap-1 text-[11px] text-muted-foreground">
                  в фокусе агента:
                  {[...new Set(v.focus)].map((f) => (
                    <span key={f} className="rounded-full border border-primary/30 bg-primary/10 px-2 py-px text-primary">
                      {f}
                    </span>
                  ))}
                </div>
              )}
            </>
          ) : (
            <Empty icon={Network} title="Связей пока нет" text={cur.running ? "Подграф вырастет, когда агент пойдёт по связям." : "Агент ответил без обхода связей: смотрите шаги и источники."} />
          )}
        </TabsContent>
        <TabsContent value="step" className="mt-0 min-h-0 flex-1 overflow-y-auto data-[state=inactive]:hidden">
          <StepDetail s={sel} steps={v.tools} onPick={pick} />
        </TabsContent>
        <TabsContent value="sources" className="mt-0 min-h-0 flex-1 overflow-y-auto data-[state=inactive]:hidden">
          <Sources s={cur} v={v} />
        </TabsContent>
        <TabsContent value="summary" className="mt-0 min-h-0 flex-1 space-y-3 overflow-y-auto p-3 data-[state=inactive]:hidden">
          {cur.result?.route === "chat" ? (
            <Empty icon={MessagesSquare} title="Без LLM и поиска" text="Короткая реплика: ответ шаблоном, 0 токенов." />
          ) : cur.result ? (
            <div className="space-y-3 rounded-xl border bg-card p-3 shadow-xs">
              <p className="text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground">Где потрачено время</p>
              <Breakdown t={cur.result.timings_ms} />
              <p className="border-t pt-2 font-mono text-[11px] text-muted-foreground">
                потрачено {fmtNum(cur.result.tokens_spent)} ток. · инструменты вернули {fmtNum(cur.result.turn_chars ?? cur.result.context_chars)} симв.
              </p>
            </div>
          ) : (
            <Empty icon={Gauge} title="Ещё считаем" text="Разбивка по времени появится после ответа." />
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

// ─── Боковая панель: вопросы диалога и история запусков ─────────
type Ev = HarnessEvent | { type: "reset"; query: string } | { type: "abort" } | { type: "stopping" };
type Chat = { turns: State[]; sel: number; session?: string; dialog: number };

// Каждое событие стрима относится к последнему ходу; reset открывает новый ход в текущем диалоге.
function reduceChat(c: Chat, e: Ev | { type: "select"; i: number } | { type: "new" } | { type: "load"; run: RunDetail }): Chat {
  if (e.type === "select") return { ...c, sel: e.i };
  if (e.type === "new") return { ...c, session: undefined, dialog: c.dialog + 1 };
  if (e.type === "load") return { turns: [...c.turns, fromRun(e.run, c.dialog + 1)], sel: c.turns.length, session: undefined, dialog: c.dialog + 2 };
  if (e.type === "reset") return { ...c, turns: [...c.turns, reduce(initial, { ...e, dialog: c.dialog })], sel: c.turns.length };
  if (!c.turns.length) return c;
  const turns = [...c.turns];
  turns[turns.length - 1] = reduce(turns[turns.length - 1], e);
  const session = e.type === "start" ? e.session_id : e.type === "error" && e.session === "lost" ? undefined : c.session;
  return { ...c, turns, session, dialog: e.type === "error" && e.session === "lost" ? c.dialog + 1 : c.dialog };
}

function TurnIcon({ t }: { t: State }) {
  if (t.running) return <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />;
  if (t.error) return <XCircle className="h-3.5 w-3.5 text-destructive" />;
  if (t.result?.unsupported.length) return <ShieldAlert className="h-3.5 w-3.5 text-[hsl(var(--chart-5))]" />;
  if (t.result?.outcome === "answered") return <ShieldCheck className="h-3.5 w-3.5 text-primary" />;
  if (t.result?.outcome) return <ShieldAlert className="h-3.5 w-3.5 text-[hsl(var(--chart-5))]" />;
  return <History className="h-3.5 w-3.5 text-muted-foreground" />;
}

function Sidebar({ c, onSelect, onNew, onLoad, busy }: { c: Chat; onSelect: (i: number) => void; onNew: () => void; onLoad: (id: string) => void; busy: boolean }) {
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => getJson<RunSummary[]>("/runs?limit=200") });
  const hist = (runs.data ?? []).filter((r) => r.mode === "harness").reverse().slice(0, 30);
  return (
    <div className="pb-3">
      <SectionLabel
        action={
          <Button size="sm" variant="ghost" className="h-6 gap-1 px-1.5 text-[11px] normal-case tracking-normal" onClick={onNew} disabled={busy || !c.session} data-testid="button-new-dialog">
            <Plus className="h-3 w-3" /> новый
          </Button>
        }
      >
        Диалог
      </SectionLabel>
      {!c.turns.length && <p className="px-3 py-2 text-xs text-muted-foreground">Вопросы этой вкладки появятся здесь.</p>}
      {c.turns.map((t, i) => (
        <div key={i}>
          {i > 0 && t.dialog !== c.turns[i - 1].dialog && <div className="mx-3 my-1 border-t border-dashed" />}
          <ListItem
            active={i === c.sel}
            onClick={() => onSelect(i)}
            icon={<TurnIcon t={t} />}
            title={t.query}
            meta={t.running ? "агент работает…" : `${t.steps.filter((x) => x.kind === "tool").length} вызовов · ${fmtNum(t.ctx)} ток.`}
            testid={`turn-item-${i}`}
          />
        </div>
      ))}
      <SectionLabel>История</SectionLabel>
      {!hist.length && <p className="px-3 py-2 text-xs text-muted-foreground">Запусков harness ещё нет.</p>}
      {hist.map((r) => (
        <ListItem
          key={r.id}
          onClick={() => onLoad(r.id)}
          icon={r.unsupported ? <ShieldAlert className="h-3.5 w-3.5 text-[hsl(var(--chart-5))]" /> : <History className="h-3.5 w-3.5 text-muted-foreground" />}
          title={r.query}
          meta={`${new Date(r.timestamp).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })} · ${r.steps ?? "—"} шагов · ${fmtMs(r.total_ms)}`}
          testid={`run-item-${r.id}`}
        />
      ))}
    </div>
  );
}

// Индикатор окна как Context из AI Elements: кольцо заполнения, подробности во всплывашке.
function ContextRing({ s }: { s: State }) {
  const pct = Math.min((s.ctx / s.budget) * 100, 100);
  const r = 7, len = 2 * Math.PI * r;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="flex h-7 items-center gap-1.5 rounded-full px-2 font-mono text-[11px] text-muted-foreground hover:bg-secondary" data-testid="bar-dialog">
          <svg viewBox="0 0 18 18" className="h-4 w-4 -rotate-90">
            <circle cx="9" cy="9" r={r} fill="none" stroke="hsl(var(--muted))" strokeWidth="2.5" />
            <circle cx="9" cy="9" r={r} fill="none" stroke={pct > 75 ? "hsl(var(--chart-5))" : pct > 50 ? "hsl(var(--chart-3))" : "hsl(var(--primary))"} strokeWidth="2.5" strokeLinecap="round" strokeDasharray={`${(pct / 100) * len} ${len}`} />
          </svg>
          {Math.round(pct)}%
        </span>
      </TooltipTrigger>
      <TooltipContent className="space-y-0.5 text-xs">
        <p>Окно диалога · вопрос {s.turn}</p>
        <p className="font-mono text-muted-foreground">
          {fmtNum(s.ctx)} / {fmtNum(s.budget)} ток. · осталось {fmtNum(Math.max(s.budget - s.ctx, 0))}
        </p>
        {s.ctx >= s.budget && <p className="text-[hsl(var(--chart-5))]">бюджет исчерпан — начните новый диалог</p>}
      </TooltipContent>
    </Tooltip>
  );
}

const FEATURES = [
  { icon: Network, t: "Граф вместо простыни", d: "Связи сущностей, Personalized PageRank и цепочки путей — агент читает только нужные пассажи." },
  { icon: Gauge, t: "Бюджет контекста", d: "Короткие ответы инструментов и превью вместо полных текстов экономят окно." },
  { icon: Sparkles, t: "Проверяемый ответ", d: "Утверждения с источниками [P#]; код и верификатор отклоняют неподтверждённое." },
];

export default function HarnessPage() {
  const [c, dispatch] = useReducer(reduceChat, { turns: [], sel: 0, dialog: 0 });
  const [tab, setTab] = useState("graph");
  const [pickKey, setPickKey] = useState<string>();
  const abort = useRef<AbortController | null>(null);
  const qc = useQueryClient();
  const last = c.turns[c.turns.length - 1];
  const running = !!last?.running;
  const now = useNow(running);
  const cur = c.turns[c.sel];
  const v = useMemo(() => (cur ? view(cur, now) : undefined), [cur, now]);
  const scroll = useStickToBottom(`${c.turns.length}:${last?.steps.length}:${!!last?.result}`);
  // выбранный шаг; пока не выбран — инспектор следует за последним шагом
  const sel = v?.tools.find((t) => t.key === pickKey) ?? v?.tools[v.tools.length - 1];

  useStatus(
    cur && (
      <>
        {c.session && <span>сессия {c.session.slice(0, 6)}</span>}
        <span>вопрос {cur.turn}</span>
        <span className={cn((v?.pct ?? 0) > 75 && "text-[hsl(var(--chart-5))]")}>
          окно {fmtNum(cur.ctx)}/{fmtNum(cur.budget)} ток.
        </span>
        <span>
          шаг {v?.stepNo ?? 0}/{cur.maxSteps}
        </span>
        <span>{fmtMs(v?.elapsed ?? 0)}</span>
      </>
    ),
    [c.session, cur?.turn, cur?.ctx, cur?.budget, v?.stepNo, cur?.maxSteps, Math.round((v?.elapsed ?? 0) / 500)],
  );

  const run = async (query: string) => {
    if (running) return;
    abort.current = new AbortController();
    setPickKey(undefined);
    dispatch({ type: "reset", query });
    try {
      await streamHarness(query, dispatch, abort.current.signal, c.session);
    } catch (e) {
      if ((e as Error).name === "AbortError") dispatch({ type: "abort" });
      else dispatch({ type: "error", detail: (e as Error).message });
    } finally {
      qc.invalidateQueries({ queryKey: ["runs"] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
    }
  };
  // Остановка на сервере: новых шагов не будет, придёт done с outcome=cancelled и накопленными материалами.
  // Обрываем соединение, только если сервер запуска не знает (сессии ещё нет или 404).
  const stop = () => {
    if (!c.session) return abort.current?.abort();
    dispatch({ type: "stopping" });
    postJson("/harness/stop", { session_id: c.session }).catch(() => abort.current?.abort());
  };
  // Остановка на сервере: новых шагов не будет, придёт done с outcome=cancelled и накопленными материалами.
  // Обрываем соединение, только если сервер запуска не знает (сессии ещё нет или 404).
  const stop = () => {
    if (!c.session) return abort.current?.abort();
    dispatch({ type: "stopping" });
    postJson("/harness/stop", { session_id: c.session }).catch(() => abort.current?.abort());
  };
  const select = (i: number) => {
    dispatch({ type: "select", i });
    setPickKey(undefined);
    document.getElementById(`turn-${i}`)?.scrollIntoView({ block: "start", behavior: "smooth" });
  };
  const pick = (i: number, key: string) => {
    dispatch({ type: "select", i });
    setPickKey(key);
    setTab("step");
  };
  const load = async (id: string) => {
    if (running) return;
    const r = await getJson<RunDetail>(`/runs/${id}`);
    setPickKey(undefined);
    dispatch({ type: "load", run: r });
  };

  return (
    <Workspace
      id="harness"
      title={
        <span className="flex items-center gap-2">
          Harness
          {cur && <span className="font-normal normal-case tracking-normal text-muted-foreground/80">· {cur.query}</span>}
        </span>
      }
      sidebarTitle="Сессия"
      sidebar={<Sidebar c={c} onSelect={select} onNew={() => dispatch({ type: "new" })} onLoad={load} busy={running} />}
      inspector={<Inspector cur={cur} v={v} tab={tab} setTab={setTab} pick={(k) => setPickKey(k)} sel={sel} />}
    >
      <ChatPane
        scrollRef={scroll}
        composer={
          <Composer
            busy={running}
            onSend={run}
            onStop={stop}
            placeholder={c.session ? "Уточняющий вопрос — агент помнит прочитанное…" : "Задайте вопрос по базе знаний…"}
            examples={c.turns.length ? undefined : EXAMPLES}
            tools={
              <>
                <span className="flex h-7 items-center gap-1.5 rounded-full bg-primary/10 px-2.5 text-xs font-medium text-primary" title="Агент отвечает шагами по графу знаний">
                  <Network className="h-3.5 w-3.5" /> Граф-агент
                </span>
                {c.session && (
                  <ComposerChip onClick={() => dispatch({ type: "new" })} disabled={running} title="Сбросить контекст и начать новый диалог" testid="button-new-dialog-composer">
                    <Plus className="h-3.5 w-3.5" /> Новый диалог
                  </ComposerChip>
                )}
                {c.session && last && !last.history && <ContextRing s={last} />}
              </>
            }
          />
        }
      >
        {!c.turns.length ? (
          <div className="flex flex-col items-center gap-6 pt-14 text-center">
            <span className="flex h-12 w-12 items-center justify-center rounded-2xl border bg-card shadow-sm">
              <Sparkles className="h-6 w-6 text-primary" />
            </span>
            <div className="space-y-1.5">
              <h1 className="text-2xl font-semibold tracking-tight">Спросите базу знаний</h1>
              <p className="mx-auto max-w-md text-sm text-muted-foreground">Агент идёт по графу связей, читает только нужные пассажи и подтверждает каждое утверждение источником.</p>
            </div>
            <div className="grid w-full gap-2 text-left sm:grid-cols-3">
              {FEATURES.map((f) => (
                <div key={f.t} className="space-y-1.5 rounded-xl border bg-card p-3 shadow-xs">
                  <f.icon className="h-4 w-4 text-primary" />
                  <p className="text-[13px] font-medium">{f.t}</p>
                  <p className="text-xs leading-relaxed text-muted-foreground">{f.d}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          c.turns.map((t, i) => (
            <div key={i} id={`turn-${i}`} className="scroll-mt-4 space-y-3" data-testid={`turn-${i}`}>
              {i > 0 && t.dialog !== c.turns[i - 1].dialog && (
                <div className="flex items-center gap-3 py-1 text-[11px] text-muted-foreground" data-testid="divider-dialog">
                  <span className="h-px flex-1 bg-border" /> {t.history ? "из истории" : "новый диалог — контекст сброшен"} <span className="h-px flex-1 bg-border" />
                </div>
              )}
              <UserMsg text={t.query} active={c.turns.length > 1 && i === c.sel} onClick={() => select(i)} />
              <BotMsg>
                <Trace s={t} sel={i === c.sel ? sel?.key : undefined} onPick={(k) => pick(i, k)} />
                <Answer s={t} />
              </BotMsg>
            </div>
          ))
        )}
      </ChatPane>
    </Workspace>
  );
}
