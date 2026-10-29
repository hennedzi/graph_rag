import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, CircleDashed, FileUp, Loader2, UploadCloud, XCircle } from "lucide-react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { useToast } from "@/hooks/use-toast";
import { byLabel, fmtMs, fmtNum, getJson, upload, type Counts, type Job, type Metric } from "@/lib/api";
import { cn } from "@/lib/utils";

const ICON = {
  queued: { i: CircleDashed, c: "text-muted-foreground", l: "в очереди" },
  running: { i: Loader2, c: "text-primary animate-spin", l: "обработка" },
  done: { i: CheckCircle2, c: "text-primary", l: "готово" },
  failed: { i: XCircle, c: "text-destructive", l: "ошибка" },
};

export default function DocumentsPage() {
  const qc = useQueryClient();
  const { toast } = useToast();
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: () => getJson<Job[]>("/etl/jobs"),
    refetchInterval: (q) => (q.state.data?.some((j) => j.status === "queued" || j.status === "running") ? 1500 : 10000),
  });
  const stats = useQuery({ queryKey: ["stats"], queryFn: () => getJson<Counts>("/stats"), refetchInterval: 5000 });
  const dash = useQuery({ queryKey: ["dashboard"], queryFn: () => getJson<Metric[]>("/dashboard"), refetchInterval: 5000 });
  const send = useMutation({
    mutationFn: upload,
    onSuccess: (js) => {
      toast({ title: "Загружено", description: `${js.length} файл(ов) поставлено в очередь ETL` });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (e) => toast({ title: "Не удалось загрузить", description: (e as Error).message, variant: "destructive" }),
  });
  const finished = jobs.data?.filter((j) => j.status === "done" || j.status === "failed").length ?? 0;
  useEffect(() => {
    // задача завершилась → счётчики Milvus и ETL-метрики изменились, не ждём следующего опроса
    qc.invalidateQueries({ queryKey: ["stats"] });
    qc.invalidateQueries({ queryKey: ["dashboard"] });
  }, [finished, qc]);
  const pick = (fl: FileList | null) => fl && fl.length && send.mutate([...fl]);

  const triples = byLabel(dash.data, "etl_triples", "result");
  const rows = byLabel(dash.data, "etl_rows_written", "type");
  const tripleData = Object.entries(triples).map(([k, v]) => ({ k, v }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Документы и ETL</h1>
        <p className="text-sm text-muted-foreground">Файл → чанки → триплеты (LLM) → сущности, связи, пассажи в Milvus.</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        {(["passage", "entity", "relation"] as const).map((t, i) => (
          <Card key={t}>
            <CardContent className="p-4">
              <p className="text-xs text-muted-foreground">{["Пассажи", "Сущности", "Связи"][i]}</p>
              <p className="font-mono text-xl font-semibold tabular-nums" data-testid={`text-count-${t}`}>
                {fmtNum(stats.data?.[t])}
              </p>
              <p className="text-xs text-muted-foreground">записано ETL за сессию: {fmtNum(rows[t] ?? 0)}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-12">
        <div className="min-w-0 space-y-6 lg:col-span-8">
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              pick(e.dataTransfer.files);
            }}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 text-center transition-colors",
              drag ? "border-primary bg-primary/5" : "border-border",
            )}
            data-testid="dropzone"
          >
            {send.isPending ? <Loader2 className="h-8 w-8 animate-spin text-primary" /> : <UploadCloud className="h-8 w-8 text-primary" />}
            <div>
              <p className="font-medium">Перетащите файлы сюда</p>
              <p className="text-sm text-muted-foreground">.txt, .md — разбиваются на чанки; .json — [{"{"}"text", "triples"{"}"}]</p>
            </div>
            <Button type="button" variant="outline" size="sm" className="gap-2" onClick={() => input.current?.click()} data-testid="button-pick">
              <FileUp className="h-4 w-4" /> Выбрать файлы
            </Button>
            <input ref={input} type="file" multiple accept=".txt,.md,.json" className="hidden" onChange={(e) => pick(e.target.files)} data-testid="input-file" />
          </label>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Задачи ETL</CardTitle>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Документ</TableHead>
                    <TableHead>Статус</TableHead>
                    <TableHead className="text-right">Чанки</TableHead>
                    <TableHead className="text-right">Триплеты</TableHead>
                    <TableHead className="text-right">Строк</TableHead>
                    <TableHead className="text-right">Время</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(jobs.data ?? []).map((j) => {
                    const s = ICON[j.status];
                    const total = j.timings_ms ? Object.values(j.timings_ms).reduce((a, b) => a + b, 0) : undefined;
                    return (
                      <TableRow key={j.job_id} data-testid={`row-job-${j.job_id}`}>
                        <TableCell className="max-w-[220px] truncate font-medium">{j.doc}</TableCell>
                        <TableCell>
                          <span className="flex items-center gap-1.5 text-sm" title={j.error}>
                            <s.i className={cn("h-4 w-4", s.c)} /> {s.l}
                          </span>
                        </TableCell>
                        <TableCell className="text-right font-mono">{j.chunks}</TableCell>
                        <TableCell className="text-right font-mono">{j.triples ?? "—"}</TableCell>
                        <TableCell className="text-right font-mono">{j.rows_written ?? "—"}</TableCell>
                        <TableCell className="text-right font-mono">{fmtMs(total)}</TableCell>
                      </TableRow>
                    );
                  })}
                  {!jobs.data?.length && (
                    <TableRow>
                      <TableCell colSpan={6} className="py-8 text-center text-sm text-muted-foreground">
                        Пока нет задач. Задачи хранятся в памяти процесса и пропадают при рестарте сервера.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>

        <Card className="min-w-0 lg:col-span-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Валидация триплетов</CardTitle>
          </CardHeader>
          <CardContent>
            {tripleData.length ? (
              <ChartContainer config={{ v: { label: "Триплеты", color: "hsl(var(--chart-2))" } }} className="h-56 w-full">
                <BarChart data={tripleData}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="k" tickLine={false} axisLine={false} fontSize={11} />
                  <YAxis tickLine={false} axisLine={false} fontSize={11} width={32} allowDecimals={false} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar isAnimationActive={false} dataKey="v" fill="var(--color-v)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartContainer>
            ) : (
              <p className="py-10 text-center text-sm text-muted-foreground">Появится после первой загрузки.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
