import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { FileText, GitBranch, Loader2, MessageSquareText, Search, Tag } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
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

export default function SearchPage() {
  const [q, setQ] = useState(EXAMPLES[0]);
  const search = useMutation({ mutationFn: (query: string) => postJson<SearchResult>("/search", { query }) });
  const ask = useMutation({ mutationFn: (query: string) => postJson<AskResult>("/ask", { query }) });
  const go = () => {
    if (!q.trim()) return;
    search.mutate(q);
    ask.mutate(q);
  };
  const r = search.data;
  const maxScore = Math.max(1e-6, ...(r?.passages ?? []).map((p) => p.score));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Поиск и классический ответ</h1>
        <p className="text-sm text-muted-foreground">
          /search — гибридный retrieval (dense + BM25 → RRF → реранк), /ask — один проход генерации с верификацией.
        </p>
      </div>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          go();
        }}
      >
        <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Вопрос…" className="h-11" data-testid="input-search" />
        <Button type="submit" className="h-11 gap-2" disabled={search.isPending || ask.isPending} data-testid="button-search">
          {search.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          <span className="hidden sm:inline">Найти</span>
        </Button>
      </form>

      <div className="grid gap-6 lg:grid-cols-12">
        <section className="min-w-0 space-y-4 lg:col-span-7">
          {search.isError && <p className="font-mono text-sm text-destructive">{(search.error as Error).message}</p>}
          {search.isPending && <Skeleton className="h-64 w-full" />}
          {r && (
            <>
              <div className="grid gap-4 sm:grid-cols-2">
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
              </div>
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
          )}
          {!r && !search.isPending && (
            <Card>
              <CardContent className="p-8 text-center text-sm text-muted-foreground">Введите вопрос — здесь появятся сущности, связи и пассажи.</CardContent>
            </Card>
          )}
        </section>

        <aside className="min-w-0 space-y-4 lg:col-span-5">
          <Card data-testid="card-ask">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <MessageSquareText className="h-4 w-4 text-primary" /> Ответ /ask
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {ask.isPending && (
                <div className="space-y-2">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-5/6" />
                  <Skeleton className="h-4 w-2/3" />
                </div>
              )}
              {ask.isError && <p className="font-mono text-sm text-destructive">{(ask.error as Error).message}</p>}
              {ask.data && (
                <>
                  <p className="whitespace-pre-wrap text-sm leading-relaxed">{ask.data.answer}</p>
                  <div className="flex flex-wrap gap-1">
                    {ask.data.sources.map((p, i) => (
                      <SourceBadge key={p.id} h={String(i + 1)} cite={{ doc_id: p.id, doc: p.doc, text: p.text }} />
                    ))}
                  </div>
                  <Verdict unsupported={ask.data.unsupported} />
                  <Timeline t={ask.data.timings_ms} />
                </>
              )}
              {!ask.data && !ask.isPending && !ask.isError && <p className="text-sm text-muted-foreground">Ответ появится здесь.</p>}
            </CardContent>
          </Card>
          {r && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Этапы retrieval</CardTitle>
              </CardHeader>
              <CardContent>
                <Timeline t={r.timings_ms} />
              </CardContent>
            </Card>
          )}
        </aside>
      </div>
    </div>
  );
}
