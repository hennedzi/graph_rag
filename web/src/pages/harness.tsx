import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { BotMsg, ChatLayout, Composer, UserMsg, useStickToBottom } from "@/components/chat";
import { AnimatePresence, motion } from "framer-motion";
import { Brain, ChevronDown, Gauge, Loader2, MessagesSquare, Network, Plus, Sparkles, Timer, Footprints } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { GraphView } from "@/components/graph-view";
import { Claims, Verdict } from "@/components/answer-card";
import { STATUS, toneClass, toolMeta } from "@/components/tools";
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
  dialog: number; // номер диалога в ленте: новый диалог = новая серверная сессия
  turn: number;
  chars0: number; // контекст диалога до этого вопроса
};

const initial: State = { query: "", running: false, budget: 16000, maxSteps: 8, chars: 0, steps: [], edges: [], dialog: 0, turn: 1, chars0: 0 };

function reduce(s: State, e: HarnessEvent | { type: "reset"; query: string; dialog: number } | { type: "abort" }): State {
  switch (e.type) {
    case "reset":
      return { ...initial, query: e.query, dialog: e.dialog, running: true, started: Date.now() };
    case "start":
      // в продолжении диалога сервер присылает уже накопленный контекст и подграф
      return { ...s, budget: e.budget, maxSteps: e.max_steps, turn: e.turn, chars: e.chars, chars0: e.chars, edges: e.edges };
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
      <div className="flex flex-wrap items-baseline gap-x-1.5">
        <span className="whitespace-nowrap font-mono text-lg font-semibold tabular-nums">{value}</span>
        {sub && <span className="whitespace-nowrap font-mono text-xs text-muted-foreground">{sub}</span>}
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

type Ev = HarnessEvent | { type: "reset"; query: string } | { type: "abort" };
type Chat = { turns: State[]; sel: number; session?: string; dialog: number };

// Каждое событие стрима относится к последнему ходу; reset открывает новый ход в текущем диалоге.
function reduceChat(c: Chat, e: Ev | { type: "select"; i: number } | { type: "new" }): Chat {
  if (e.type === "select") return { ...c, sel: e.i };
  if (e.type === "new") return { ...c, session: undefined, dialog: c.dialog + 1 };
  if (e.type === "reset") return { ...c, turns: [...c.turns, reduce(initial, { ...e, dialog: c.dialog })], sel: c.turns.length };
  if (!c.turns.length) return c;
  const turns = [...c.turns];
  turns[turns.length - 1] = reduce(turns[turns.length - 1], e);
  const session = e.type === "start" ? e.session_id : e.type === "error" && e.session === "lost" ? undefined : c.session;
  return { ...c, turns, session, dialog: e.type === "error" && e.session === "lost" ? c.dialog + 1 : c.dialog };
}

function view(s: State, now: number) {
  const tools = s.steps.filter((x): x is Extract<Step, { kind: "tool" }> => x.kind === "tool");
  const focus = tools.filter((t) => t.tool === "neighbors" && t.res?.out.startsWith("сущность:")).map((t) => t.res!.out.split("\n")[0].slice(10).trim());
  // экономия контекста: сколько пассажей агент увидел превью vs сколько реально прочитал
  const seen = new Set(tools.flatMap((t) => [...(t.res?.out ?? "").matchAll(/\[(P\d+)\]/g)].map((m) => m[1])));
  const read = new Set(tools.filter((t) => t.tool === "read_passage" && t.res).flatMap((t) => ((t.args.ids as string[]) ?? []).map((h) => h.toUpperCase())));
  return {
    tools,
    focus,
    seen,
    read,
    stepNo: Math.max(0, ...s.steps.map((x) => x.step)),
    elapsed: s.started ? (s.finished ?? now) - s.started : 0,
    pct: (s.chars / s.budget) * 100,
  };
}

function Trace({ s, last }: { s: State; last: boolean }) {
  const [open, setOpen] = useState(true);
  const calls = s.steps.filter((x) => x.kind === "tool").length;
  // ход старых вопросов сворачивается, чтобы лента оставалась чатом; раскрыть можно кликом
  useEffect(() => setOpen(last), [last]);
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger asChild>
        <button type="button" className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground" data-testid="button-trace">
          {s.running ? <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" /> : <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")} />}
          {s.running ? "агент работает" : "ход рассуждения"} · <span className="font-mono">{calls} вызовов</span>
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="relative mt-3 space-y-3 before:absolute before:bottom-2 before:left-[13px] before:top-2 before:w-px before:bg-border">
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
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function Answer({ s }: { s: State }) {
  if (!s.result && !s.error) return null;
  return (
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
      <Card className={cn("rounded-2xl rounded-tl-sm", s.error ? "border-destructive/50" : "border-primary/40 shadow-[0_0_40px_-16px_hsl(var(--primary)/0.5)]")} data-testid="card-answer">
        <CardContent className="space-y-4 p-4">
          {s.error && <p className="font-mono text-sm text-destructive">{s.error}</p>}
          {s.result && (
            <>
              <Claims claims={s.result.claims} answer={s.result.answer} citations={s.result.citations} />
              <Verdict unsupported={s.result.unsupported} />
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge variant="secondary" className="font-mono text-[11px]">
                  {s.result.steps} шагов
                </Badge>
                <Badge variant="secondary" className="font-mono text-[11px]">
                  {s.turn > 1 ? `+${fmtNum(s.result.turn_chars)}` : fmtNum(s.result.context_chars)} симв. контекста
                </Badge>
                {s.result.fix_rounds > 0 && (
                  <Badge variant="secondary" className="font-mono text-[11px]">
                    исправлений {s.result.fix_rounds}
                  </Badge>
                )}
              </div>
              <Breakdown t={s.result.timings_ms} />
            </>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

function DialogBar({ s, onNew, busy }: { s: State; onNew: () => void; busy: boolean }) {
  const left = Math.max(s.budget - s.chars, 0);
  const pct = (s.chars / s.budget) * 100;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-xl border bg-card/80 px-3 py-2 text-xs backdrop-blur" data-testid="bar-dialog">
      <MessagesSquare className="h-3.5 w-3.5 text-primary" />
      <span>
        Диалог · вопрос <span className="font-mono">{s.turn}</span>
      </span>
      <span className="flex min-w-[120px] flex-1 items-center gap-2">
        <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
          <span
            className={cn("block h-full rounded-full", pct > 75 ? "bg-[hsl(var(--chart-5))]" : pct > 50 ? "bg-[hsl(var(--chart-3))]" : "bg-primary")}
            style={{ width: `${Math.min(pct, 100)}%` }}
          />
        </span>
        <span className="whitespace-nowrap font-mono text-muted-foreground">
          осталось {fmtNum(left)} / {fmtNum(s.budget)}
        </span>
      </span>
      {left === 0 && <span className="text-[hsl(var(--chart-5))]">бюджет исчерпан: ответы только по прочитанному</span>}
      <Button size="sm" variant="outline" className="h-7 gap-1 px-2 text-xs" onClick={onNew} disabled={busy} data-testid="button-new-dialog">
        <Plus className="h-3.5 w-3.5" /> Новый диалог
      </Button>
    </div>
  );
}

const FEATURES = [
  { icon: Network, t: "Граф вместо простыни", d: "Агент ходит по связям сущностей (1–2 хопа) и читает только нужные пассажи." },
  { icon: Gauge, t: "Бюджет контекста", d: "Каждый ответ инструмента короткий; превью вместо полных текстов экономит окно." },
  { icon: Sparkles, t: "Проверяемый ответ", d: "Утверждения с источниками [P#]; код и верификатор отклоняют неподтверждённое." },
];

export default function HarnessPage() {
  const [c, dispatch] = useReducer(reduceChat, { turns: [], sel: 0, dialog: 0 });
  const abort = useRef<AbortController | null>(null);
  const qc = useQueryClient();
  const last = c.turns[c.turns.length - 1];
  const running = !!last?.running;
  const now = useNow(running);
  const cur = c.turns[c.sel];
  const v = useMemo(() => (cur ? view(cur, now) : undefined), [cur, now]);
  useStickToBottom(`${c.turns.length}:${last?.steps.length}:${!!last?.result}`);

  const run = async (query: string) => {
    if (running) return;
    abort.current = new AbortController();
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

  const aside = (
    <>
      <div className="grid grid-cols-3 gap-2">
        <Meter icon={Footprints} label="Шаги" value={String(v?.stepNo ?? 0)} sub={`/ ${cur?.maxSteps ?? 8}`} pct={v ? (v.stepNo / cur!.maxSteps) * 100 : 0} tone="bg-[hsl(var(--chart-2))]" />
        <Meter
          icon={Gauge}
          label={cur && cur.turn > 1 ? "Контекст диалога" : "Контекст"}
          value={fmtNum(cur?.chars ?? 0)}
          sub={`/ ${fmtNum(cur?.budget ?? 16000)}`}
          pct={v?.pct ?? 0}
          tone={(v?.pct ?? 0) > 75 ? "bg-[hsl(var(--chart-5))]" : (v?.pct ?? 0) > 50 ? "bg-[hsl(var(--chart-3))]" : "bg-primary"}
        />
        <Meter icon={Timer} label="Время" value={fmtMs(v?.elapsed ?? 0)} sub={cur?.running ? "идёт" : "итого"} />
      </div>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-base">
            <span className="flex items-center gap-2">
              <Network className="h-4 w-4 text-primary" /> Исследованный подграф
            </span>
            <span className="font-mono text-xs font-normal text-muted-foreground">{cur?.edges.length ?? 0} связей</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {cur ? <GraphView edges={cur.edges} focus={v!.focus} /> : <p className="py-16 text-center text-sm text-muted-foreground">Граф появится, когда агент пойдёт по связям.</p>}
        </CardContent>
      </Card>
      <Card>
        <CardContent className="grid grid-cols-2 gap-4 p-4 text-sm">
          <div>
            <p className="text-xs text-muted-foreground">Пассажей увидел (превью)</p>
            <p className="font-mono text-lg font-semibold">{v?.seen.size ?? 0}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Прочитал целиком</p>
            <p className="font-mono text-lg font-semibold text-primary">{v?.read.size ?? 0}</p>
          </div>
          <p className="col-span-2 text-xs text-muted-foreground">
            Агент читает полный текст только там, где это нужно для ответа, — остальное остаётся одной строкой превью.
            {c.turns.length > 1 && " Кликните по своему вопросу в чате, чтобы увидеть его метрики."}
          </p>
        </CardContent>
      </Card>
    </>
  );

  return (
    <ChatLayout
      aside={aside}
      composer={
        <Composer
          busy={running}
          onSend={run}
          onStop={() => abort.current?.abort()}
          placeholder={c.session ? "Уточняющий вопрос — агент помнит прочитанное…" : "Задайте вопрос по базе знаний…"}
          examples={c.turns.length ? undefined : EXAMPLES}
          toolbar={c.session && last && <DialogBar s={last} onNew={() => dispatch({ type: "new" })} busy={running} />}
        />
      }
    >
      {!c.turns.length ? (
        <div className="space-y-6 pt-6">
          <div className="space-y-1">
            <h1 className="text-2xl font-semibold tracking-tight">Harness</h1>
            <p className="text-sm text-muted-foreground">Агент отвечает атомарными шагами по графу знаний — каждый шаг виден вживую.</p>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            {FEATURES.map((f) => (
              <Card key={f.t}>
                <CardContent className="space-y-2 p-4">
                  <f.icon className="h-5 w-5 text-primary" />
                  <p className="text-sm font-medium">{f.t}</p>
                  <p className="text-xs text-muted-foreground">{f.d}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      ) : (
        c.turns.map((t, i) => (
          <div key={i} className="space-y-4" data-testid={`turn-${i}`}>
            {i > 0 && t.dialog !== c.turns[i - 1].dialog && (
              <div className="flex items-center gap-3 py-2 text-xs text-muted-foreground" data-testid="divider-dialog">
                <span className="h-px flex-1 bg-border" /> новый диалог — контекст сброшен <span className="h-px flex-1 bg-border" />
              </div>
            )}
            <UserMsg text={t.query} active={c.turns.length > 1 && i === c.sel} onClick={() => dispatch({ type: "select", i })} />
            <BotMsg>
              <Trace s={t} last={i === c.turns.length - 1} />
              <Answer s={t} />
            </BotMsg>
          </div>
        ))
      )}
    </ChatLayout>
  );
}
