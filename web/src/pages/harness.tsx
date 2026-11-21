import { useEffect, useReducer, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Brain, Check, ChevronDown, Copy, Loader2, Network, Plus, Sparkles } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { BotMsg, ChatPane, Composer, ComposerChip, UserMsg, useStickToBottom } from "@/components/chat";
import { Card, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { ProseAnswer, SourcesList, Verdict } from "@/components/answer-card";
import { STATUS, toneClass, toolMeta } from "@/components/tools";
import { fmtMs, fmtNum, postJson, streamHarness, type Edge, type HarnessEvent, type HarnessResult } from "@/lib/api";
import { cn } from "@/lib/utils";

export const EXAMPLES = [];

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
  ctx: number;
  steps: Step[];
  edges: Edge[];
  result?: HarnessResult;
  error?: string;
  started?: number;
  finished?: number;
  dialog: number;
  turn: number;
  chars0: number;
  history?: boolean;
  stopping?: boolean;
  stream?: string;
};

const initial: State = { query: "", running: false, budget: 64000, maxSteps: 8, chars: 0, ctx: 0, steps: [], edges: [], dialog: 0, turn: 1, chars0: 0 };

function reduce(s: State, e: HarnessEvent | { type: "reset"; query: string; dialog: number } | { type: "abort" } | { type: "stopping" }): State {
  switch (e.type) {
    case "stopping":
      return { ...s, stopping: true };
    case "reset":
      return { ...initial, query: e.query, dialog: e.dialog, running: true, started: Date.now() };
    case "start":
      return { ...s, budget: e.budget, maxSteps: e.max_steps, turn: e.turn, chars: e.chars, ctx: e.ctx_tokens, chars0: e.chars, edges: e.edges };
    case "thought":
      return { ...s, steps: [...s.steps, { kind: "thought", key: `t${s.steps.length}`, step: e.step, text: e.text }] };
    case "fast_path_taken":
      return { ...s, steps: [...s.steps, { kind: "thought", key: `f${s.steps.length}`, step: s.steps.length + 1,
                                          text: `быстрый путь: реранкер уверен (score ${(e.top_score ?? 0).toFixed(2)})` }] };
    case "token":
      return { ...s, stream: (s.stream ?? "") + e.text };
    case "answer_finished":
      return { ...s, stream: e.text };
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
    default:
      return s;
  }
}

type RunDetail = HarnessResult & { total_ms?: number };

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

const argText = (v: unknown) => (typeof v === "string" ? `"${v}"` : JSON.stringify(v));
const inlineArgs = (tool: string, args: Record<string, unknown>) =>
  tool === "answer" && typeof args.text === "string"
    ? `ответ · ${(args.text as string).length} симв.${args.no_answer ? " · no_answer" : ""}`
    : tool === "answer" && Array.isArray(args.claims)
      ? `${args.claims.length} утвержд.${args.no_answer ? " · no_answer" : ""}`
      : Object.entries(args)
          .map(([k, v]) => `${k}=${argText(v)}`)
          .join(" ");

function TraceLine({ s, active }: { s: ToolStepT; active: boolean }) {
  const meta = toolMeta(s.tool);
  const status = s.res && s.res.status !== "ok" ? STATUS[s.res.status] ?? { label: s.res.status, tone: "warn" as const } : null;
  const lines = s.res?.out ? s.res.out.split("\n").filter(Boolean) : [];
  const first = s.tool === "answer" ? (s.res?.out === "принято" ? "принято" : lines[0]) : lines[0];
  const [open, setOpen] = useState(false);
  return (
    <div className="w-full">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className={cn("w-full rounded-md px-2 py-1 text-left font-mono hover:bg-secondary/60", active && "bg-secondary ring-1 ring-primary/30")}
        data-testid={`step-tool-${s.key}`}
        aria-expanded={open}
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
            {s.res?.speculative && <span className="rounded border px-1 py-px font-sans font-medium">из кэша</span>}
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
      {open && s.res && (
        <div className="mt-1 space-y-2 p-2 rounded-md border bg-muted/40 text-[11px]">
          <div className="flex gap-2">
            <span className="font-semibold text-muted-foreground">Аргументы</span>
            <pre className="min-w-0 flex-1 overflow-x-auto whitespace-pre-wrap break-all font-mono text-[10px] text-muted-foreground">{JSON.stringify(s.args, null, 2)}</pre>
          </div>
          <div>
            <span className="font-semibold text-muted-foreground block mb-1">Вывод</span>
            <pre className="whitespace-pre-wrap break-words font-mono text-[10px]">{s.res.out}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

function Typewriter({ text, cps = 240, className }: { text: string; cps?: number; className?: string }) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    setCount(0);
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const tick = (t: number) => {
      acc += ((t - last) / 1000) * cps;
      last = t;
      const next = Math.min(text.length, Math.floor(acc));
      setCount((c) => (next > c ? next : c));
      if (next < text.length) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [text, cps]);
  return <span className={className}>{text.slice(0, count)}</span>;
}

function ThoughtLine({ s, animate }: { s: Extract<Step, { kind: "thought" }>; animate?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <button type="button" onClick={() => setOpen(!open)} className="flex w-full items-start gap-2 text-left" data-testid="thought">
      <Brain className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
      {animate && !open ? (
        <Typewriter text={s.text} className="line-clamp-2 text-[12.5px] leading-relaxed text-muted-foreground" />
      ) : (
        <span className={cn("text-[12.5px] leading-relaxed text-muted-foreground", !open && "line-clamp-2")}>{s.text}</span>
      )}
    </button>
  );
}

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
          {s.steps.map((st) => (st.kind === "thought" ? <ThoughtLine key={st.key} s={st} animate={s.running} /> : <TraceLine key={st.key} s={st} active={st.key === sel} />))}
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
  if (!s.result && !s.error && !s.stream) return null;
  if (!s.result && !s.error && s.stream)
    return (
      <div className="pt-0.5" data-testid="answer-stream">
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{s.stream}</p>
        {s.running && <span className="mt-1 inline-block h-4 w-1.5 animate-pulse bg-primary align-middle" />}
      </div>
    );
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
              <ProseAnswer text={s.result.answer} citations={s.result.citations} />
              <Verdict unsupported={s.result.unsupported} outcome={s.result.outcome} note={s.result.outcome_note} />
              <SourcesList citations={s.result.citations} />
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
  { icon: Check, t: "Бюджет контекста", d: "Короткие ответы инструментов и превью вместо полных текстов экономят окно." },
  { icon: Sparkles, t: "Проверяемый ответ", d: "Утверждения с источниками [P#]; код и верификатор отклоняют неподтверждённое." },
];

type Ev = HarnessEvent | { type: "reset"; query: string } | { type: "abort" } | { type: "stopping" };
type Chat = { turns: State[]; sel: number; session?: string; dialog: number };

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

export default function HarnessPage() {
  const [c, dispatch] = useReducer(reduceChat, { turns: [], sel: 0, dialog: 0 });
  const abort = useRef<AbortController | null>(null);
  const last = c.turns[c.turns.length - 1];
  const running = !!last?.running;
  const now = useNow(running);
  const cur = c.turns[c.sel];
  const scroll = useStickToBottom(`${c.turns.length}:${last?.steps.length}:${Math.floor((last?.stream?.length ?? 0) / 80)}:${!!last?.result}`);
  const tools = cur?.steps.filter((x): x is ToolStepT => x.kind === "tool") ?? [];
  const sel = tools[tools.length - 1]?.key;

  const run = async (query: string) => {
    if (running) return;
    abort.current = new AbortController();
    dispatch({ type: "reset", query });
    try {
      await streamHarness(query, dispatch, abort.current.signal, c.session);
    } catch (e) {
      if ((e as Error).name === "AbortError") dispatch({ type: "abort" });
      else dispatch({ type: "error", detail: (e as Error).message });
    }
  };
  const stop = () => {
    if (!c.session) return abort.current?.abort();
    dispatch({ type: "stopping" });
    postJson("/harness/stop", { session_id: c.session }).catch(() => abort.current?.abort());
  };
  const select = (i: number) => {
    dispatch({ type: "select", i });
    document.getElementById(`turn-${i}`)?.scrollIntoView({ block: "start", behavior: "smooth" });
  };
  const pick = (i: number, key: string) => {
    dispatch({ type: "select", i });
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {running ? "Идёт поиск по базе знаний." : cur?.error ? `Ошибка: ${cur.error}` : cur?.result ? "Ответ готов." : ""}
      </div>
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
                <Trace s={t} sel={i === c.sel ? sel : undefined} onPick={() => pick(i, sel ?? "")} />
                <Answer s={t} />
              </BotMsg>
            </div>
          ))
        )}
      </ChatPane>
    </div>
  );
}
