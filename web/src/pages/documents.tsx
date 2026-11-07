import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "wouter";
import { CheckCircle2, CircleDashed, FileText, FileUp, Loader2, Orbit, RefreshCw, Search as SearchIcon, Trash2, UploadCloud, XCircle } from "lucide-react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { ListItem, SectionLabel, Workspace, useStatus } from "@/components/shell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { predicate } from "@/components/graph-view";
import { useToast } from "@/hooks/use-toast";
import { byLabel, del, fmtMs, fmtNum, getJson, upload, type Counts, type DocDetail, type DocSummary, type Dropped, type Job, type Metric } from "@/lib/api";
import { cn } from "@/lib/utils";

const JOB = {
  queued: { i: CircleDashed, c: "text-muted-foreground", l: "в очереди" },
  running: { i: Loader2, c: "text-primary animate-spin", l: "обработка" },
  done: { i: CheckCircle2, c: "text-primary", l: "готово" },
  failed: { i: XCircle, c: "text-destructive", l: "ошибка" },
};

const fmtDate = (s?: string) =>
  s ? new Date(s).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", year: "2-digit", hour: "2-digit", minute: "2-digit" }) : "—";
const fmtBytes = (n?: number) => (n == null ? "—" : n < 1024 ? `${n} Б` : n < 2 ** 20 ? `${(n / 1024).toFixed(1)} КБ` : `${(n / 2 ** 20).toFixed(1)} МБ`);
const dropped = (d: Dropped) => `пассажей ${d.passages}, сущностей ${d.entities}, связей ${d.relations}`;

/** Кнопка удаления с подтверждением вторым кликом — без модальных окон. */
function ConfirmDelete({ onConfirm, label, busy, small }: { onConfirm: () => void; label: string; busy?: boolean; small?: boolean }) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(t);
  }, [armed]);
  return (
    <Button
      type="button"
      size={small ? "icon" : "sm"}
      variant={armed ? "destructive" : "ghost"}
      className={cn(small ? "h-7 w-auto px-2" : "h-8 gap-1.5", !armed && "text-muted-foreground hover:text-destructive")}
      disabled={busy}
      onClick={() => (armed ? (setArmed(false), onConfirm()) : setArmed(true))}
      aria-label={label}
      data-testid={`button-delete${small ? "-passage" : ""}`}
    >
      {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
      {armed ? <span className="text-xs">Точно удалить?</span> : !small && <span>{label}</span>}
    </Button>
  );
}

function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="min-w-[96px] rounded-lg border bg-card px-3 py-2" title={hint}>
      <p className="text-[11px] text-muted-foreground">{label}</p>
      <p className="font-mono text-base font-semibold tabular-nums">{value}</p>
    </div>
  );
}

function DropZone({ onFiles, busy, compact }: { onFiles: (f: File[]) => void; busy: boolean; compact?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        onFiles([...e.dataTransfer.files]);
      }}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed text-center transition-colors",
        compact ? "p-2" : "p-10",
        drag ? "border-primary bg-primary/5" : "border-border hover:border-primary/40",
      )}
      data-testid="dropzone"
    >
      {busy ? <Loader2 className="h-6 w-6 animate-spin text-primary" /> : !compact && <UploadCloud className="h-6 w-6 text-primary" />}
      {!compact && <p className="font-medium">Перетащите файлы или выберите</p>}
      {!compact && (
        <p className="max-w-md text-sm text-muted-foreground">
          .txt, .md — чанки и триплеты через LLM; .json — готовые {"[{text, triples}]"}. Файл с тем же именем заменяет прошлую версию.
        </p>
      )}
      <Button type="button" variant="outline" size="sm" className="gap-2" onClick={() => input.current?.click()} data-testid="button-pick">
        <FileUp className="h-4 w-4" /> {compact ? "Загрузить или перетащить" : "Выбрать файлы"}
      </Button>
      <input ref={input} type="file" multiple accept=".txt,.md,.json" className="hidden" onChange={(e) => e.target.files && onFiles([...e.target.files])} data-testid="input-file" />
    </label>
  );
}

function Overview({ counts, docs, onFiles, busy }: { counts?: Counts; docs: DocSummary[]; onFiles: (f: File[]) => void; busy: boolean }) {
  const top = [...docs].sort((a, b) => b.shared - a.shared).slice(0, 5);
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-6 p-6">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight">База знаний</h1>
          <p className="text-sm text-muted-foreground">Документ → чанки → триплеты (LLM) → сущности, связи и пассажи в Milvus. Выберите документ слева, чтобы посмотреть, что из него извлечено.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Stat label="Документы" value={fmtNum(docs.length)} />
          <Stat label="Пассажи" value={fmtNum(counts?.passage)} />
          <Stat label="Сущности" value={fmtNum(counts?.entity)} />
          <Stat label="Связи" value={fmtNum(counts?.relation)} />
        </div>
        <DropZone onFiles={onFiles} busy={busy} />
        {top.some((d) => d.shared) && (
          <div className="space-y-2">
            <p className="text-sm font-medium">Как документы связаны в графе</p>
            <p className="text-xs text-muted-foreground">Общие сущности — мосты, по которым агент переходит между документами (multi-hop).</p>
            {top.map((d) => (
              <div key={d.doc} className="flex items-center gap-3 text-sm">
                <span className="w-48 truncate">{d.doc}</span>
                <span className="h-1.5 flex-1 rounded-full bg-muted">
                  <span className="block h-full rounded-full bg-primary" style={{ width: `${(d.shared / Math.max(1, d.entities)) * 100}%` }} />
                </span>
                <span className="w-24 text-right font-mono text-xs text-muted-foreground">
                  {d.shared}/{d.entities} общих
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function DocView({ name, summary, onReplace, onDeleted }: { name: string; summary?: DocSummary; onReplace: (f: File) => void; onDeleted: () => void }) {
  const qc = useQueryClient();
  const { toast } = useToast();
  const replace = useRef<HTMLInputElement>(null);
  const [q, setQ] = useState("");
  const d = useQuery({ queryKey: ["doc", name], queryFn: () => getJson<DocDetail>(`/etl/documents/${encodeURIComponent(name)}`) });
  const refresh = () => ["docs", "doc", "stats", "graph-stats"].forEach((k) => qc.invalidateQueries({ queryKey: [k] }));
  const rmDoc = useMutation({
    mutationFn: () => del<Dropped>(`/etl/documents/${encodeURIComponent(name)}`),
    onSuccess: (r) => {
      toast({ title: `Удалён ${name}`, description: dropped(r) });
      refresh();
      onDeleted();
    },
    onError: (e) => toast({ title: "Не удалось удалить", description: (e as Error).message, variant: "destructive" }),
  });
  const rmPassage = useMutation({
    mutationFn: (id: string) => del<Dropped>(`/etl/passages/${id}`),
    onSuccess: (r) => {
      toast({ title: "Пассаж удалён", description: dropped(r) });
      refresh();
    },
    onError: (e) => toast({ title: "Не удалось удалить", description: (e as Error).message, variant: "destructive" }),
  });
  const ql = q.trim().toLowerCase();
  const passages = useMemo(
    () => (d.data?.passages ?? []).map((p, i) => ({ ...p, n: i + 1 })).filter((p) => !ql || p.text.toLowerCase().includes(ql) || p.entities.some((e) => e.includes(ql))),
    [d.data, ql],
  );
  const relations = (d.data?.relations ?? []).filter((r) => !ql || r.text.includes(ql));
  const entities = (d.data?.entities ?? []).filter((e) => !ql || e.name.includes(ql));
  const meta = d.data?.meta;

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 space-y-3 border-b px-5 py-4">
        <div className="flex flex-wrap items-start gap-2">
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-lg font-semibold tracking-tight" data-testid="text-doc-name">
              {name}
            </h1>
            <p className="font-mono text-[11px] text-muted-foreground">
              загружен {fmtDate(meta?.uploaded_at)} · {fmtBytes(meta?.bytes)} · {fmtNum(summary?.chars)} симв. · триплетов {fmtNum(meta?.triples)}
            </p>
          </div>
          <Button size="sm" variant="outline" className="h-8 gap-1.5" onClick={() => replace.current?.click()} data-testid="button-replace">
            <RefreshCw className="h-3.5 w-3.5" /> Заменить
          </Button>
          <input
            ref={replace}
            type="file"
            accept=".txt,.md,.json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              // имя документа = имя файла; новая версия под тем же именем вытесняет старые пассажи
              if (f) onReplace(new File([f], name, { type: f.type }));
              e.target.value = "";
            }}
          />
          <ConfirmDelete label="Удалить" onConfirm={() => rmDoc.mutate()} busy={rmDoc.isPending} />
        </div>
        <div className="flex flex-wrap gap-2">
          <Stat label="Пассажи" value={fmtNum(summary?.passages)} />
          <Stat label="Сущности" value={fmtNum(summary?.entities)} />
          <Stat label="Связи" value={fmtNum(summary?.relations)} />
          <Stat label="Общие с другими" value={fmtNum(summary?.shared)} hint="Сущности, которые упоминаются и в других документах" />
        </div>
      </div>

      {d.isLoading ? (
        <div className="flex flex-1 items-center justify-center text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      ) : d.error ? (
        <p className="p-5 font-mono text-sm text-destructive">{(d.error as Error).message}</p>
      ) : (
        <Tabs defaultValue="passages" className="flex min-h-0 flex-1 flex-col">
          <div className="flex shrink-0 items-center gap-2 border-b px-5 py-2">
            <TabsList className="h-8">
              <TabsTrigger value="passages" className="text-xs">
                Пассажи {passages.length}
              </TabsTrigger>
              <TabsTrigger value="relations" className="text-xs">
                Связи {relations.length}
              </TabsTrigger>
              <TabsTrigger value="entities" className="text-xs">
                Сущности {entities.length}
              </TabsTrigger>
            </TabsList>
            <div className="relative ml-auto w-full max-w-xs">
              <SearchIcon className="absolute left-2.5 top-2 h-4 w-4 text-muted-foreground" />
              <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Фильтр по тексту и сущностям" className="h-8 pl-8 text-sm" data-testid="input-doc-filter" />
            </div>
          </div>
          <TabsContent value="passages" className="mt-0 min-h-0 flex-1 overflow-y-auto">
            <div className="space-y-2 p-5">
              {passages.map((p) => (
                <div key={p.id} className="group rounded-lg border bg-card p-3" data-testid={`passage-${p.n}`}>
                  <div className="mb-1.5 flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
                    <span className="rounded bg-muted px-1.5 py-0.5 text-foreground">#{p.n}</span>
                    <span>{p.text.length} симв.</span>
                    <span>· связей {p.relations}</span>
                    <span className="ml-auto opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                      <ConfirmDelete small label="Удалить пассаж" onConfirm={() => rmPassage.mutate(p.id)} busy={rmPassage.isPending && rmPassage.variables === p.id} />
                    </span>
                  </div>
                  <p className="text-[13px] leading-relaxed">{p.text}</p>
                  {p.entities.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {p.entities.map((e) => (
                        <span key={e} className="rounded-md bg-accent px-1.5 py-0.5 text-[11px] text-accent-foreground">
                          {e}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {!passages.length && <p className="py-10 text-center text-sm text-muted-foreground">Ничего не найдено.</p>}
            </div>
          </TabsContent>
          <TabsContent value="relations" className="mt-0 min-h-0 flex-1 overflow-y-auto">
            <div className="divide-y px-5 py-2">
              {relations.map((r) => (
                <div key={r.text} className="flex items-baseline gap-2 py-1.5 text-[13px]">
                  <span className="font-medium">{r.src}</span>
                  <span className="text-muted-foreground">—{predicate({ ...r, handles: [] })}→</span>
                  <span className="font-medium">{r.dst}</span>
                  <span className="ml-auto shrink-0 font-mono text-[11px] text-muted-foreground">{r.passages.length} пасс.</span>
                </div>
              ))}
              {!relations.length && <p className="py-10 text-center text-sm text-muted-foreground">Связей нет.</p>}
            </div>
          </TabsContent>
          <TabsContent value="entities" className="mt-0 min-h-0 flex-1 overflow-y-auto">
            <div className="grid gap-1.5 p-5 sm:grid-cols-2 xl:grid-cols-3">
              {entities.map((e) => (
                <Link
                  key={e.name}
                  href={`/graph?e=${encodeURIComponent(e.name)}`}
                  className="flex items-center gap-2 rounded-md border bg-card px-2.5 py-1.5 text-[13px] hover:border-primary/40 hover:bg-accent"
                  title="Открыть окрестность в графе (PPR)"
                >
                  <Orbit className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  <span className="min-w-0 flex-1 truncate">{e.name}</span>
                  <span className={cn("font-mono text-[11px]", e.total > e.here ? "text-primary" : "text-muted-foreground")} title="здесь / во всей базе">
                    {e.here}/{e.total}
                  </span>
                </Link>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}

export default function DocumentsPage() {
  const qc = useQueryClient();
  const { toast } = useToast();
  const [sel, setSel] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: () => getJson<Job[]>("/etl/jobs"),
    refetchInterval: (q) => (q.state.data?.some((j) => j.status === "queued" || j.status === "running") ? 1500 : 10000),
  });
  const docs = useQuery({ queryKey: ["docs"], queryFn: () => getJson<DocSummary[]>("/etl/documents") });
  const stats = useQuery({ queryKey: ["stats"], queryFn: () => getJson<Counts>("/stats"), refetchInterval: 10000 });
  const dash = useQuery({ queryKey: ["dashboard"], queryFn: () => getJson<Metric[]>("/dashboard"), refetchInterval: 10000 });
  const send = useMutation({
    mutationFn: upload,
    onSuccess: (js) => {
      toast({ title: "Поставлено в очередь", description: js.map((j) => j.doc).join(", ") });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (e) => toast({ title: "Не удалось загрузить", description: (e as Error).message, variant: "destructive" }),
  });
  const finished = jobs.data?.filter((j) => j.status === "done" || j.status === "failed").length ?? 0;
  useEffect(() => {
    // задача завершилась → база изменилась, не ждём следующего опроса
    ["docs", "doc", "stats", "dashboard", "graph-stats"].forEach((k) => qc.invalidateQueries({ queryKey: [k] }));
  }, [finished, qc]);
  const onFiles = (fl: File[]) => fl.length && send.mutate(fl);

  const list = (docs.data ?? []).filter((d) => d.doc.toLowerCase().includes(filter.trim().toLowerCase()));
  const active = (jobs.data ?? []).filter((j) => j.status !== "done");
  const recent = (jobs.data ?? []).filter((j) => j.status === "done").slice(0, 8);
  const tripleData = Object.entries(byLabel(dash.data, "etl_triples", "result")).map(([k, v]) => ({ k, v }));
  const summary = docs.data?.find((d) => d.doc === sel);

  useStatus(
    <>
      <span>документов {fmtNum(docs.data?.length)}</span>
      {active.some((j) => j.status !== "failed") && <span className="text-primary">ETL: {active.filter((j) => j.status !== "failed").length} в работе</span>}
    </>,
    [docs.data?.length, active.length],
  );

  return (
    <Workspace
      id="documents"
      title={sel ? "Документ" : "База знаний"}
      sidebarTitle="Документы"
      actions={
        sel && (
          <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={() => setSel(null)}>
            Обзор
          </Button>
        )
      }
      sidebar={
        <div className="flex h-full flex-col">
          <div className="flex shrink-0 gap-1.5 p-2">
            <div className="relative flex-1">
              <SearchIcon className="absolute left-2 top-2 h-3.5 w-3.5 text-muted-foreground" />
              <Input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Найти документ" className="h-7 pl-7 text-xs" data-testid="input-doc-search" />
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto pb-3">
            {active.length > 0 && (
              <>
                <SectionLabel>Обработка</SectionLabel>
                {active.map((j) => {
                  const s = JOB[j.status];
                  return <ListItem key={j.job_id} icon={<s.i className={cn("h-3.5 w-3.5", s.c)} />} title={j.doc} meta={j.error ? j.error : `${s.l} · ${j.chunks} чанков`} />;
                })}
              </>
            )}
            <SectionLabel>В базе {docs.data && `(${docs.data.length})`}</SectionLabel>
            {docs.isLoading && <p className="px-3 text-xs text-muted-foreground">Загрузка…</p>}
            {docs.data && !docs.data.length && <p className="px-3 text-xs text-muted-foreground">Пусто — загрузите первый документ.</p>}
            {list.map((d) => (
              <ListItem
                key={d.doc}
                active={d.doc === sel}
                onClick={() => setSel(d.doc)}
                icon={<FileText className="h-3.5 w-3.5 text-muted-foreground" />}
                title={d.doc}
                meta={`${d.passages} пасс. · ${d.entities} сущн. · ${fmtDate(d.uploaded_at)}`}
                testid={`doc-${d.doc}`}
              />
            ))}
          </div>
          <div className="shrink-0 border-t p-2">
            <DropZone compact onFiles={onFiles} busy={send.isPending} />
          </div>
        </div>
      }
      inspector={
        <div className="h-full overflow-y-auto">
          <SectionLabel>Последние загрузки</SectionLabel>
          {!recent.length && <p className="px-3 text-xs text-muted-foreground">Задачи хранятся в памяти процесса и пропадают при рестарте.</p>}
          {recent.map((j) => (
            <div key={j.job_id} className="border-l-2 border-transparent px-3 py-1.5">
              <p className="truncate text-[13px]">{j.doc}</p>
              <p className="font-mono text-[10.5px] text-muted-foreground">
                {j.chunks} чанков · {j.triples ?? "—"} трипл. · {fmtMs(j.timings_ms ? Object.values(j.timings_ms).reduce((a, b) => a + b, 0) : undefined)}
                {!!j.replaced && ` · заменено ${j.replaced}`}
              </p>
            </div>
          ))}
          <SectionLabel>Валидация триплетов</SectionLabel>
          <div className="px-3 pb-4">
            {tripleData.length ? (
              <ChartContainer config={{ v: { label: "Триплеты", color: "hsl(var(--chart-2))" } }} className="h-44 w-full">
                <BarChart data={tripleData}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="k" tickLine={false} axisLine={false} fontSize={10} />
                  <YAxis tickLine={false} axisLine={false} fontSize={10} width={28} allowDecimals={false} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar isAnimationActive={false} dataKey="v" fill="var(--color-v)" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ChartContainer>
            ) : (
              <p className="text-xs text-muted-foreground">Появится после первой загрузки через LLM.</p>
            )}
          </div>
        </div>
      }
    >
      {sel ? (
        <DocView key={sel} name={sel} summary={summary} onReplace={(f) => send.mutate([f])} onDeleted={() => setSel(null)} />
      ) : (
        <Overview counts={stats.data} docs={docs.data ?? []} onFiles={onFiles} busy={send.isPending} />
      )}
    </Workspace>
  );
}
