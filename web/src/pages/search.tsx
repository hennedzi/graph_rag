import { useState } from "react";
import { FileText, GitBranch, Loader2, Search as SearchIcon, Tag, Timer } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BotMsg, ChatPane, Composer, UserMsg, useStickToBottom } from "@/components/chat";
import { ListItem, SectionLabel, Workspace } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { SourceBadge, Verdict } from "@/components/answer-card";
import { EXAMPLES } from "@/pages/harness";
import { fmtMs, postJson, type AskResult, type Passage, type SearchResult, type Timings } from "@/lib/api";

function Timeline({ t }: { t: Timings }) {
  const entries = Object.entries(t).filter(([k]) => k !== "total");
  const max = Math.max(1, ...entries.map(([, v]) => v));
  return (
    <div className="space-y-1.5">
      {entries.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[110px_1fr_56px] items-center gap-2 text-xs">
          <span className="truncate font-mono text-muted-foreground">{k}</span>
          <div className="h-1.5 rounded-full bg-muted">
            <div className="h-full rounded-full bg-[hsl(var(--chart-4))]" style={{ width: `${(v / max) * 100}%` }} />
          </div>
          <span className="text-right font-mono">{fmtMs(v)}</span>
        </div>
      ))}
    </div>
  );
}

function PassageRow({ p, i, maxScore }: { p: Passage; i: number; maxScore: number }) {
  return (
    <div className="space-y-2 rounded-lg border p-3" data-testid={`passage-${i}`}>
      <div className="flex items-center gap-2 text-xs">
        <span className="font-mono text-primary">#{i + 1}</span>
        <span className="truncate text-muted-foreground">{p.doc}</span>
        <span className="ml-auto flex items-center gap-2 font-mono">
          <span className="hidden h-1.5 w-20 rounded-full bg-muted sm:block">
            <span className="block h-full rounded-full bg-primary" style={{ width: `${Math.max(4, (p.score / maxScore) * 100)}%` }} />
          </span>
          {p.score.toFixed(2)}
        </span>
      </div>
      <p className="text-sm leading-relaxed">{p.text}</p>
    </div>
  );
}

type Turn = { q: string; search?: SearchResult; ask?: AskResult; sErr?: string; aErr?: string };

function Retrieval({ r, err }: { r?: SearchResult; err?: string }) {
  if (err) return <p className="font-mono text-sm text-destructive">{err}</p>;
  if (!r) return <Skeleton className="h-72 w-full" />;
  const maxScore = Math.max(1e-6, ...r.passages.map((p) => p.score));
  return (
    <>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Timer className="h-4 w-4 text-[hsl(var(--chart-4))]" /> Этапы retrieval
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Timeline t={r.timings_ms} />
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Tag className="h-4 w-4 text-[hsl(var(--chart-4))]" /> Сущности
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-1.5">
          {r.entities.map((e) => (
            <Badge key={e} variant="secondary">
              {e}
            </Badge>
          ))}
          {!r.entities.length && <span className="text-sm text-muted-foreground">нет</span>}
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            <GitBranch className="h-4 w-4 text-[hsl(var(--chart-2))]" /> Связи
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {r.relations.map((x) => (
            <p key={x} className="font-mono text-xs text-muted-foreground">
              {x}
            </p>
          ))}
          {!r.relations.length && <span className="text-sm text-muted-foreground">нет</span>}
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-primary" /> Пассажи после реранка
            </span>
            <span className="font-mono text-xs font-normal text-muted-foreground">score реранкера</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {r.passages.map((p, i) => (
            <PassageRow key={p.id} p={p} i={i} maxScore={maxScore} />
          ))}
          {!r.passages.length && <p className="text-sm text-muted-foreground">Ничего выше порога релевантности.</p>}
        </CardContent>
      </Card>
    </>
  );
}

function AskAnswer({ t }: { t: Turn }) {
  if (t.aErr) return <p className="font-mono text-sm text-destructive">{t.aErr}</p>;
  if (!t.ask)
    return (
      <div className="space-y-2 pt-1" data-testid="ask-pending">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
        <Skeleton className="h-4 w-2/3" />
      </div>
    );
  return (
    <Card className="border-primary/30" data-testid="card-ask">
      <CardContent className="space-y-4 p-4">
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{t.ask.answer}</p>
        <div className="flex flex-wrap gap-1">
          {t.ask.sources.map((p, i) => (
            <SourceBadge key={p.id} h={String(i + 1)} cite={{ doc_id: p.id, doc: p.doc, text: p.text }} />
          ))}
        </div>
        <Verdict unsupported={t.ask.unsupported} />
        <Timeline t={t.ask.timings_ms} />
      </CardContent>
    </Card>
  );
}

export default function SearchPage() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sel, setSel] = useState(0);
  const patch = (i: number, p: Partial<Turn>) => setTurns((ts) => ts.map((t, j) => (j === i ? { ...t, ...p } : t)));
  const last = turns[turns.length - 1];
  const busy = !!last && ((!last.search && !last.sErr) || (!last.ask && !last.aErr));
  const scroll = useStickToBottom(`${turns.length}:${!!last?.ask}`);

  const go = (q: string) => {
    const i = turns.length;
    setTurns((ts) => [...ts, { q }]);
    setSel(i);
    postJson<SearchResult>("/search", { query: q }).then((search) => patch(i, { search }), (e: Error) => patch(i, { sErr: e.message }));
    postJson<AskResult>("/ask", { query: q }).then((ask) => patch(i, { ask }), (e: Error) => patch(i, { aErr: e.message }));
  };
  const cur = turns[sel];

  const select = (i: number) => {
    setSel(i);
    document.getElementById(`sturn-${i}`)?.scrollIntoView({ block: "start", behavior: "smooth" });
  };

  return (
    <Workspace
      id="search"
      title="Поиск и классический ответ"
      sidebarTitle="Запросы"
      sidebar={
        <div className="pb-3">
          <SectionLabel>Эта вкладка</SectionLabel>
          {!turns.length && <p className="px-3 py-2 text-xs text-muted-foreground">Запросы появятся здесь.</p>}
          {turns.map((t, i) => (
            <ListItem
              key={i}
              active={i === sel}
              onClick={() => select(i)}
              icon={!t.ask && !t.aErr ? <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" /> : <SearchIcon className="h-3.5 w-3.5 text-muted-foreground" />}
              title={t.q}
              meta={t.search ? `${t.search.passages.length} пассажей · ${fmtMs(Object.values(t.search.timings_ms).reduce((a, b) => a + b, 0))}` : "…"}
            />
          ))}
        </div>
      }
      inspector={
        <div className="flex h-full flex-col">
          <div className="flex h-9 shrink-0 items-center border-b px-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Retrieval</div>
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
            {cur ? <Retrieval r={cur.search} err={cur.sErr} /> : <p className="p-6 text-center text-sm text-muted-foreground">Здесь появятся сущности, связи, пассажи и тайминги retrieval.</p>}
          </div>
        </div>
      }
    >
      <ChatPane
        scrollRef={scroll}
        composer={<Composer busy={busy} onSend={go} placeholder="Вопрос для /search + /ask…" examples={turns.length ? undefined : EXAMPLES} testid="input-search" />}
      >
        {!turns.length ? (
          <div className="space-y-1 pt-8">
            <h1 className="text-xl font-semibold tracking-tight">Поиск и классический ответ</h1>
            <p className="text-sm text-muted-foreground">/search — гибридный retrieval (dense + BM25 → RRF → реранк), /ask — один проход генерации с верификацией.</p>
          </div>
        ) : (
          turns.map((t, i) => (
            <div key={i} id={`sturn-${i}`} className="scroll-mt-4 space-y-3" data-testid={`turn-${i}`}>
              <UserMsg text={t.q} active={turns.length > 1 && i === sel} onClick={() => select(i)} />
              <BotMsg>
                <AskAnswer t={t} />
              </BotMsg>
            </div>
          ))
        )}
      </ChatPane>
    </Workspace>
  );
}
