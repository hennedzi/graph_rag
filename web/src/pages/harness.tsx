import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Brain, ChevronDown, CircleStop, Gauge, Loader2, Network, Play, Sparkles, Timer, Footprints } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { GraphView } from "@/components/graph-view";
import { Claims, Verdict } from "@/components/answer-card";
import { STATUS, TOOLS, toneClass, toolMeta } from "@/components/tools";
import { fmtMs, fmtNum, streamHarness, type Edge, type HarnessEvent, type HarnessResult } from "@/lib/api";
import { cn } from "@/lib/utils";

export const EXAMPLES = [
  "Кто был учителем Леонарда Эйлера?",
  "Как связаны Иоганн Бернулли и Леонард Эйлер?",
  "Какой закон сформулировал Даниил Бернулли?",
  "Кто пригласил Эйлера в Петербург?",
];

type Step =
  | { kind: "thought"; key: string; step: number; text: string }
  | {
      kind: "tool";
      key: string;
      step: number;
      tool: string;
      args: Record<string, unknown>;
      res?: Extract<HarnessEvent, { type: "result" }>;
    };

type State = {
  query: string;
  running: boolean;
  budget: number;
  maxSteps: number;
  chars: number;
  steps: Step[];
  edges: Edge[];
  result?: HarnessResult;
  error?: string;
  started?: number;
  finished?: number;
};

const initial: State = { query: "", running: false, budget: 16000, maxSteps: 8, chars: 0, steps: [], edges: [] };

function reduce(s: State, e: HarnessEvent | { type: "reset"; query: string } | { type: "abort" }): State {
  switch (e.type) {
    case "reset":
      return { ...initial, query: e.query, running: true, started: Date.now() };
    case "start":
      return { ...s, budget: e.budget, maxSteps: e.max_steps };
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
      return { ...s, steps, chars: e.total_chars, edges: e.edges.length ? [...s.edges, ...e.edges] : s.edges };
    }
    case "done":
      return { ...s, running: false, result: e.result, chars: e.result.context_chars, finished: Date.now() };
    case "error":
      return { ...s, running: false, error: e.detail, finished: Date.now() };
    case "abort":
      return { ...s, running: false, error: s.result ? undefined : "Остановлено вручную", finished: Date.now() };
  }
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

function Meter({ icon: Icon, label, value, sub, pct, tone }: { icon: typeof Gauge; label: string; value: string; sub?: string; pct?: number; tone?: string }) {
  return (
    <div className="space-y-2 rounded-lg border bg-card p-3">
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Icon className="h-3.5 w-3.5" /> {label}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="font-mono text-lg font-semibold tabular-nums">{value}</span>
        {sub && <span className="font-mono text-xs text-muted-foreground">{sub}</span>}
      </div>
      {pct != null && (
        <div className="h-1.5 overflow-hidden rounded-full bg-muted">
          <motion.div className={cn("h-full rounded-full", tone ?? "bg-primary")} animate={{ width: `${Math.min(pct, 100)}%` }} transition={{ duration: 0.4 }} />
        </div>
      )}
    </div>
  );
}

function ArgsView({ tool, args }: { tool: string; args: Record<string, unknown> }) {
  if (tool === "answer" && Array.isArray(args.claims))
    return (
      <ul className="space-y-1 text-xs">
        {(args.claims as { text: string; sources?: string[] }[]).map((c, i) => (
          <li key={i} className="flex gap-2">
            <span className="text-muted-foreground">•</span>
            <span>
              {c.text} <span className="font-mono text-primary">{(c.sources ?? []).map((h) => `[${h}]`).join("") || "[—]"}</span>
            </span>
          </li>
        ))}
      </ul>
    );
  return (
    <div className="flex flex-wrap gap-1.5">
      {Object.entries(args).map(([k, v]) => (
        <span key={k} className="rounded-md border bg-muted/50 px-2 py-0.5 font-mono text-[11px]">
          <span className="text-muted-foreground">{k}=</span>
          {typeof v === "string" ? v : JSON.stringify(v)}
        </span>
      ))}
    </div>
  );
}

function ToolStep({ s, n }: { s: Extract<Step, { kind: "tool" }>; n: number }) {
  const meta = toolMeta(s.tool);
  const Icon = meta.icon;
  const status = s.res ? STATUS[s.res.status] ?? { label: s.res.status, tone: "warn" as const } : null;
  const [open, setOpen] = useState(false);
  const long = (s.res?.out.split("\n").length ?? 0) > 6 || (s.res?.out.length ?? 0) > 500;
  return (
    <div className="relative pl-10" data-testid={`step-tool-${n}`}>
      <span
        className="absolute left-0 top-3 flex h-7 w-7 items-center justify-center rounded-full border bg-card"
        style={{ color: `hsl(${meta.color})`, borderColor: `hsl(${meta.color} / 0.4)` }}
      >
        {s.res ? <Icon className="h-3.5 w-3.5" /> : <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      </span>
      <Card className={cn("overflow-hidden", !s.res && "border-primary/40 shadow-[0_0_0_3px_hsl(var(--primary)/0.08)]")}>
        <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
          <span className="font-mono text-[11px] text-muted-foreground">шаг {s.step}</span>
          <span className="font-mono text-sm font-semibold" style={{ color: `hsl(${meta.color})` }}>
            {meta.label}
          </span>
          <span className="hidden text-xs text-muted-foreground sm:inline">{meta.hint}</span>
          <span className="ml-auto flex items-center gap-2">
            {s.res && (
              <span className="font-mono text-[11px] text-muted-foreground">
                {fmtMs(s.res.ms)} · {fmtNum(s.res.chars)} симв.
              </span>
            )}
            {status && <span className={cn("rounded-md border px-1.5 py-0.5 text-[11px] font-medium", toneClass[status.tone])}>{status.label}</span>}
          </span>
        </div>
        <div className="space-y-2 px-3 py-2.5">
          <ArgsView tool={s.tool} args={s.args} />
          {s.res && s.res.out !== "принято" && (
            <Collapsible open={open} onOpenChange={setOpen}>
              <pre
                className={cn(
                  "whitespace-pre-wrap break-words rounded-md bg-muted/40 p-2.5 font-mono text-[11.5px] leading-relaxed text-muted-foreground",
                  !open && long && "max-h-28 overflow-hidden [mask-image:linear-gradient(to_bottom,black_60%,transparent)]",
                )}
              >
                {s.res.out}
              </pre>
              {long && (
                <CollapsibleTrigger asChild>
                  <button type="button" className="mt-1 flex items-center gap-1 text-xs text-primary" data-testid={`button-expand-${n}`}>
                    <ChevronDown className={cn("h-3 w-3 transition-transform", open && "rotate-180")} />
                    {open ? "свернуть" : "весь результат"}
                  </button>
                </CollapsibleTrigger>
              )}
              <CollapsibleContent />
            </Collapsible>
          )}
          {!s.res && <div className="h-2 w-1/2 animate-pulse rounded bg-muted" />}
        </div>
      </Card>
    </div>
  );
}

function ThoughtStep({ s }: { s: Extract<Step, { kind: "thought" }> }) {
  const [open, setOpen] = useState(false);
  const long = s.text.length > 220;
  return (
    <div className="relative pl-10">
      <span className="absolute left-0 top-1 flex h-7 w-7 items-center justify-center rounded-full border border-dashed bg-background text-muted-foreground">
        <Brain className="h-3.5 w-3.5" />
      </span>
      <button type="button" onClick={() => setOpen(!open)} className="w-full text-left" data-testid="step-thought">
        <p className={cn("pt-1.5 text-[13px] leading-relaxed text-muted-foreground", !open && long && "line-clamp-2")}>{s.text}</p>
      </button>
    </div>
  );
}

function Breakdown({ t }: { t: Record<string, number> }) {
  const tools = Object.entries(t).filter(([k]) => k.startsWith("tool_")).reduce((a, [, v]) => a + v, 0);
  const parts = [
    { k: "LLM агента", v: t.harness_llm ?? 0, c: "hsl(var(--chart-2))" },
    { k: "Инструменты", v: tools, c: "hsl(var(--chart-1))" },
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
    </div>
  );
}

export default function HarnessPage() {
  const [q, setQ] = useState(EXAMPLES[0]);
  const [s, dispatch] = useReducer(reduce, initial);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const qc = useQueryClient();
  const now = useNow(s.running);

  const run = async (query: string) => {
    if (!query.trim() || s.running) return;
    abort.current = new AbortController();
    dispatch({ type: "reset", query });
    try {
      await streamHarness(query, dispatch, abort.current.signal);
    } catch (e) {
      if ((e as Error).name === "AbortError") dispatch({ type: "abort" });
      else dispatch({ type: "error", detail: (e as Error).message });
    } finally {
      qc.invalidateQueries({ queryKey: ["runs"] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
    }
  };

  useEffect(() => {
    if (s.running) bottom.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [s.steps.length, s.running]);

  const tools = s.steps.filter((x): x is Extract<Step, { kind: "tool" }> => x.kind === "tool");
  const stepNo = Math.max(0, ...s.steps.map((x) => x.step));
  const elapsed = s.started ? (s.finished ?? now) - s.started : 0;
  const pct = (s.chars / s.budget) * 100;
  const focus = useMemo(() => {
    const ents = tools.filter((t) => t.tool === "neighbors" && t.res?.out.startsWith("сущность:")).map((t) => t.res!.out.split("\n")[0].slice(10).trim());
    return ents;
  }, [tools]);
  // экономия контекста: сколько пассажей агент увидел превью vs сколько реально прочитал
  const seen = new Set(tools.flatMap((t) => [...(t.res?.out ?? "").matchAll(/\[(P\d+)\]/g)].map((m) => m[1])));
  const read = new Set(tools.filter((t) => t.tool === "read_passage" && t.res).flatMap((t) => ((t.args.ids as string[]) ?? []).map((h) => h.toUpperCase())));
  const started = s.running || s.steps.length > 0 || s.result || s.error;

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden">
        <div className="relative">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top_left,hsl(var(--primary)/0.14),transparent_55%),radial-gradient(ellipse_at_bottom_right,hsl(var(--chart-2)/0.12),transparent_50%)]" />
          <CardContent className="relative space-y-4 p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h1 className="text-xl font-semibold tracking-tight">Harness</h1>
                <p className="text-sm text-muted-foreground">Агент отвечает атомарными шагами по графу знаний — каждый шаг виден вживую.</p>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(TOOLS).map(([k, t]) => (
                  <span key={k} className="flex items-center gap-1 rounded-md border bg-background/60 px-2 py-0.5 font-mono text-[11px]" style={{ color: `hsl(${t.color})` }}>
                    <t.icon className="h-3 w-3" />
                    {k}
                  </span>
                ))}
              </div>
            </div>
            <form
              className="flex flex-col gap-2 sm:flex-row"
              onSubmit={(e) => {
                e.preventDefault();
                run(q);
              }}
            >
              <Textarea
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    run(q);
                  }
                }}
                rows={2}
                placeholder="Задайте вопрос по базе знаний…"
                className="min-h-[52px] resize-none bg-background/70 text-base"
                data-testid="input-query"
              />
              {s.running ? (
                <Button type="button" variant="outline" className="h-auto min-h-[52px] gap-2 sm:w-36" onClick={() => abort.current?.abort()} data-testid="button-stop">
                  <CircleStop className="h-4 w-4" /> Стоп
                </Button>
              ) : (
                <Button type="submit" className="h-auto min-h-[52px] gap-2 sm:w-36" data-testid="button-run">
                  <Play className="h-4 w-4" /> Запустить
                </Button>
              )}
            </form>
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  disabled={s.running}
                  onClick={() => {
                    setQ(ex);
                    run(ex);
                  }}
                  className="hover-elevate rounded-full border bg-background/60 px-3 py-1 text-xs text-muted-foreground disabled:opacity-50"
                  data-testid="button-example"
                >
                  {ex}
                </button>
              ))}
            </div>
          </CardContent>
        </div>
      </Card>

      {!started ? (
        <div className="grid gap-4 md:grid-cols-3">
          {[
            { icon: Network, t: "Граф вместо простыни", d: "Агент ходит по связям сущностей (1–2 хопа) и читает только нужные пассажи." },
            { icon: Gauge, t: "Бюджет контекста", d: "Каждый ответ инструмента короткий; превью вместо полных текстов экономит окно." },
            { icon: Sparkles, t: "Проверяемый ответ", d: "Утверждения с источниками [P#]; код и верификатор отклоняют неподтверждённое." },
          ].map((c) => (
            <Card key={c.t}>
              <CardContent className="space-y-2 p-5">
                <c.icon className="h-5 w-5 text-primary" />
                <p className="font-medium">{c.t}</p>
                <p className="text-sm text-muted-foreground">{c.d}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-12">
          <section className="order-2 min-w-0 space-y-3 lg:order-1 lg:col-span-7">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-medium text-muted-foreground">Ход рассуждения</h2>
              <span className="font-mono text-xs text-muted-foreground">{tools.length} вызовов</span>
            </div>
            <div className="relative space-y-3 before:absolute before:bottom-2 before:left-[13px] before:top-2 before:w-px before:bg-border">
              <AnimatePresence initial={false}>
                {s.steps.map((st, i) => (
                  <motion.div key={st.key} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
                    {st.kind === "thought" ? <ThoughtStep s={st} /> : <ToolStep s={st} n={i} />}
                  </motion.div>
                ))}
              </AnimatePresence>
              {s.running && (
                <div className="relative flex items-center gap-2 pl-10 text-sm text-muted-foreground">
                  <span className="absolute left-[7px] h-3 w-3 animate-pulse rounded-full bg-primary" />
                  агент думает…
                </div>
              )}
              <div ref={bottom} />
            </div>
          </section>

          <aside className="order-1 min-w-0 space-y-4 lg:sticky lg:top-20 lg:order-2 lg:col-span-5 lg:max-h-[calc(100dvh-6rem)] lg:self-start lg:overflow-y-auto lg:pr-1">
            <div className="grid grid-cols-3 gap-2">
              <Meter icon={Footprints} label="Шаги" value={String(stepNo)} sub={`/ ${s.maxSteps}`} pct={(stepNo / s.maxSteps) * 100} tone="bg-[hsl(var(--chart-2))]" />
              <Meter
                icon={Gauge}
                label="Контекст"
                value={fmtNum(s.chars)}
                sub={`/ ${fmtNum(s.budget)}`}
                pct={pct}
                tone={pct > 75 ? "bg-[hsl(var(--chart-5))]" : pct > 50 ? "bg-[hsl(var(--chart-3))]" : "bg-primary"}
              />
              <Meter icon={Timer} label="Время" value={fmtMs(elapsed)} sub={s.running ? "идёт" : "итого"} />
            </div>

            <AnimatePresence>
              {(s.result || s.error) && (
                <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}>
                  <Card className={cn(s.error ? "border-destructive/50" : "border-primary/40 shadow-[0_0_40px_-12px_hsl(var(--primary)/0.5)]")} data-testid="card-answer">
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center justify-between text-base">
                        <span>{s.error ? "Ошибка" : "Ответ"}</span>
                        {s.result && (
                          <span className="flex gap-1.5">
                            <Badge variant="secondary" className="font-mono text-[11px]">
                              {s.result.steps} шагов
                            </Badge>
                            {s.result.fix_rounds > 0 && (
                              <Badge variant="secondary" className="font-mono text-[11px]">
                                исправлений {s.result.fix_rounds}
                              </Badge>
                            )}
                          </span>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {s.error && <p className="font-mono text-sm text-destructive">{s.error}</p>}
                      {s.result && (
                        <>
                          <Claims claims={s.result.claims} answer={s.result.answer} citations={s.result.citations} />
                          <Verdict unsupported={s.result.unsupported} />
                          <Breakdown t={s.result.timings_ms} />
                        </>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center justify-between text-base">
                  <span className="flex items-center gap-2">
                    <Network className="h-4 w-4 text-primary" /> Исследованный подграф
                  </span>
                  <span className="font-mono text-xs font-normal text-muted-foreground">{s.edges.length} связей</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <GraphView edges={s.edges} focus={focus} />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="grid grid-cols-2 gap-4 p-4 text-sm">
                <div>
                  <p className="text-xs text-muted-foreground">Пассажей увидел (превью)</p>
                  <p className="font-mono text-lg font-semibold">{seen.size}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Прочитал целиком</p>
                  <p className="font-mono text-lg font-semibold text-primary">{read.size}</p>
                </div>
                <p className="col-span-2 text-xs text-muted-foreground">
                  Агент читает полный текст только там, где это нужно для ответа, — остальное остаётся одной строкой превью.
                </p>
              </CardContent>
            </Card>
          </aside>
        </div>
      )}
    </div>
  );
}
