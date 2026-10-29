import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, Bot, Coins, Gauge, ShieldCheck, Timer } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Pie,
  PieChart,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import { TOOLS, STATUS } from "@/components/tools";
import { buckets, byLabel, fmtMs, fmtNum, getJson, samples, type Metric, type RunSummary } from "@/lib/api";

const BUDGET = 16000;
const median = (xs: number[]) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};
const p95 = (xs: number[]) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(xs.length * 0.95))] : null);

function Kpi({ icon: Icon, label, value, hint, testid }: { icon: typeof Activity; label: string; value: string; hint?: string; testid: string }) {
  return (
    <Card>
      <CardContent className="space-y-1 p-4">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Icon className="h-3.5 w-3.5" /> {label}
        </div>
        <p className="font-mono text-xl font-semibold tabular-nums" data-testid={testid}>
          {value}
        </p>
        {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      </CardContent>
    </Card>
  );
}

function Empty({ h = "h-56" }: { h?: string }) {
  return <div className={`flex ${h} items-center justify-center text-sm text-muted-foreground`}>Нет данных — сделайте пару запросов</div>;
}

const STAGE_LABEL: Record<string, string> = {
  harness_llm: "LLM агента",
  harness_verify: "верификация",
  query_embed: "эмбеддинг",
  hybrid_search: "гибридный поиск",
  load_passages: "загрузка пассажей",
  rerank: "реранк",
  generation: "генерация",
  verification: "верификация",
};

export default function MetricsPage() {
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => getJson<RunSummary[]>("/runs?limit=300"), refetchInterval: 5000 });
  const dash = useQuery({ queryKey: ["dashboard"], queryFn: () => getJson<Metric[]>("/dashboard"), refetchInterval: 5000 });
  const [mode, setMode] = useState<"harness" | "ask">("harness");
  const R = runs.data ?? [];
  const M = dash.data;

  const harness = R.filter((r) => r.mode === "harness");
  const ask = R.filter((r) => r.mode === "ask");
  const req = byLabel(M, "rag_requests", "outcome");
  const reqTotal = Object.values(req).reduce((a, b) => a + b, 0);
  const tokens = byLabel(M, "rag_llm_tokens", "kind");
  const tokTotal = Object.values(tokens).reduce((a, b) => a + b, 0);
  const verifiedRate = R.length ? R.filter((r) => !r.unsupported).length / R.length : null;

  const latency = R.map((r, i) => ({ i: i + 1, [r.mode]: r.total_ms, q: r.query }));
  const latencyCfg = { harness: { label: "Harness", color: "hsl(var(--chart-1))" }, ask: { label: "/ask", color: "hsl(var(--chart-2))" } } satisfies ChartConfig;

  const stages = useMemo(() => {
    const rs = mode === "harness" ? harness : ask;
    const acc: Record<string, number> = {};
    for (const r of rs)
      for (const [k, v] of Object.entries(r.timings_ms)) {
        // в harness этапы поиска вложены в tool_search_passages — не считаем дважды
        if (mode === "harness" && !(k.startsWith("tool_") || k.startsWith("harness_"))) continue;
        acc[k] = (acc[k] ?? 0) + v;
      }
    return Object.entries(acc)
      .map(([k, v]) => ({ stage: STAGE_LABEL[k] ?? k.replace(/^tool_/, ""), ms: Math.round(v / Math.max(rs.length, 1)) }))
      .sort((a, b) => b.ms - a.ms);
  }, [R, mode]);

  const toolSamples = samples(M, "harness_tool_calls").filter((s) => s.name.endsWith("_total"));
  const statuses = [...new Set(toolSamples.map((s) => s.labels.status))];
  const toolData = Object.keys(TOOLS)
    .map((tool) => ({
      tool,
      ...Object.fromEntries(statuses.map((st) => [st, toolSamples.filter((s) => s.labels.tool === tool && s.labels.status === st).reduce((a, s) => a + s.value, 0)])),
    }))
    .filter((d) => statuses.some((st) => (d as Record<string, unknown>)[st]));
  const statusColor: Record<string, string> = {
    ok: "hsl(var(--chart-1))",
    bad_query: "hsl(var(--chart-3))",
    bad_args: "hsl(var(--chart-4))",
    bad_answer: "hsl(var(--chart-5))",
    unsupported: "hsl(var(--chart-2))",
  };
  const toolCfg = Object.fromEntries(statuses.map((s) => [s, { label: STATUS[s]?.label ?? s, color: statusColor[s] ?? "hsl(var(--muted-foreground))" }])) as ChartConfig;

  const outcomes = Object.entries(byLabel(M, "harness_answers", "outcome")).map(([k, v]) => ({ k, v }));
  const outColor: Record<string, string> = { answered: "hsl(var(--chart-1))", unsupported: "hsl(var(--chart-5))", no_context: "hsl(var(--chart-3))", max_steps: "hsl(var(--chart-2))" };
  const outLabel: Record<string, string> = { answered: "подтверждён", unsupported: "есть неподтв.", no_context: "нет контекста", max_steps: "лимит шагов" };

  const ctx = harness.map((r, i) => ({ i: i + 1, ctx: r.context_chars ?? 0, steps: r.steps ?? 0 }));
  const ctxCfg = { ctx: { label: "Контекст, симв.", color: "hsl(var(--chart-1))" }, steps: { label: "Шаги", color: "hsl(var(--chart-3))" } } satisfies ChartConfig;
  const scores = buckets(M, "rag_top_rerank_score");
  const hasScores = scores.some((b) => b.count > 0);

  if (runs.isLoading || dash.isLoading)
    return (
      <div className="grid gap-4 md:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
    );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Метрики</h1>
          <p className="text-sm text-muted-foreground">Prometheus-счётчики процесса и история runs/*.json · обновление каждые 5 с</p>
        </div>
        <Badge variant="outline" className="gap-1.5 font-mono text-[11px]">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" /> live
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        <Kpi icon={Activity} label="Запросов" value={fmtNum(reqTotal)} hint="с момента старта" testid="kpi-requests" />
        <Kpi icon={ShieldCheck} label="Без неподтверждённых" value={verifiedRate == null ? "—" : `${Math.round(verifiedRate * 100)}%`} hint={`${R.length} запусков`} testid="kpi-verified" />
        <Kpi icon={Bot} label="Harness p50 / p95" value={fmtMs(median(harness.map((r) => r.total_ms)))} hint={`p95 ${fmtMs(p95(harness.map((r) => r.total_ms)))}`} testid="kpi-harness-latency" />
        <Kpi icon={Timer} label="/ask p50" value={fmtMs(median(ask.map((r) => r.total_ms)))} hint={`p95 ${fmtMs(p95(ask.map((r) => r.total_ms)))}`} testid="kpi-ask-latency" />
        <Kpi
          icon={Gauge}
          label="Контекст harness"
          value={fmtNum(median(harness.map((r) => r.context_chars ?? 0)))}
          hint={`медиана из ${fmtNum(BUDGET)}`}
          testid="kpi-context"
        />
        <Kpi icon={Coins} label="Токены LLM" value={fmtNum(tokTotal)} hint={Object.entries(tokens).map(([k, v]) => `${k} ${fmtNum(v)}`).join(" · ") || "—"} testid="kpi-tokens" />
      </div>

      <div className="grid gap-4 lg:grid-cols-12">
        <Card className="min-w-0 lg:col-span-8">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Задержка по запускам</CardTitle>
            <CardDescription>Полное время ответа, мс</CardDescription>
          </CardHeader>
          <CardContent>
            {latency.length ? (
              <ChartContainer config={latencyCfg} className="h-64 w-full">
                <AreaChart data={latency} margin={{ left: 4, right: 8 }}>
                  <defs>
                    {(["harness", "ask"] as const).map((k) => (
                      <linearGradient key={k} id={`g-${k}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={`var(--color-${k})`} stopOpacity={0.35} />
                        <stop offset="100%" stopColor={`var(--color-${k})`} stopOpacity={0} />
                      </linearGradient>
                    ))}
                  </defs>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="i" tickLine={false} axisLine={false} fontSize={11} />
                  <YAxis tickLine={false} axisLine={false} fontSize={11} width={48} tickFormatter={(v) => fmtMs(v)} />
                  <ChartTooltip content={<ChartTooltipContent labelFormatter={(_, p) => p?.[0]?.payload?.q} />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Area isAnimationActive={false} dataKey="harness" type="monotone" stroke="var(--color-harness)" fill="url(#g-harness)" strokeWidth={2} connectNulls dot={{ r: 2 }} />
                  <Area isAnimationActive={false} dataKey="ask" type="monotone" stroke="var(--color-ask)" fill="url(#g-ask)" strokeWidth={2} connectNulls dot={{ r: 2 }} />
                </AreaChart>
              </ChartContainer>
            ) : (
              <Empty h="h-64" />
            )}
          </CardContent>
        </Card>

        <Card className="min-w-0 lg:col-span-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Итоги ответов harness</CardTitle>
            <CardDescription>harness_answers_total</CardDescription>
          </CardHeader>
          <CardContent>
            {outcomes.length ? (
              <ChartContainer
                config={Object.fromEntries(outcomes.map((o) => [o.k, { label: outLabel[o.k] ?? o.k, color: outColor[o.k] ?? "hsl(var(--muted-foreground))" }]))}
                className="mx-auto h-64 w-full"
              >
                <PieChart>
                  <ChartTooltip content={<ChartTooltipContent nameKey="k" hideLabel />} />
                  <Pie isAnimationActive={false} data={outcomes} dataKey="v" nameKey="k" innerRadius={58} outerRadius={88} paddingAngle={outcomes.length > 1 ? 3 : 0} strokeWidth={0}>
                    {outcomes.map((o) => (
                      <Cell key={o.k} fill={outColor[o.k] ?? "hsl(var(--muted-foreground))"} />
                    ))}
                  </Pie>
                  <ChartLegend content={<ChartLegendContent nameKey="k" />} />
                </PieChart>
              </ChartContainer>
            ) : (
              <Empty h="h-64" />
            )}
          </CardContent>
        </Card>

        <Card className="min-w-0 lg:col-span-6">
          <CardHeader className="flex flex-row items-start justify-between gap-2 pb-2">
            <div className="space-y-1.5">
              <CardTitle className="text-sm">Куда уходит время</CardTitle>
              <CardDescription>Среднее по этапам, мс</CardDescription>
            </div>
            <Tabs value={mode} onValueChange={(v) => setMode(v as "harness" | "ask")}>
              <TabsList className="h-8">
                <TabsTrigger value="harness" className="text-xs" data-testid="tab-stage-harness">
                  Harness
                </TabsTrigger>
                <TabsTrigger value="ask" className="text-xs" data-testid="tab-stage-ask">
                  /ask
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </CardHeader>
          <CardContent>
            {stages.length ? (
              <ChartContainer config={{ ms: { label: "мс", color: mode === "harness" ? "hsl(var(--chart-1))" : "hsl(var(--chart-2))" } }} className="h-64 w-full">
                <BarChart data={stages} layout="vertical" margin={{ left: 8, right: 16 }}>
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" />
                  <XAxis type="number" tickLine={false} axisLine={false} fontSize={11} tickFormatter={(v) => fmtMs(v)} />
                  <YAxis type="category" dataKey="stage" tickLine={false} axisLine={false} fontSize={11} width={118} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar isAnimationActive={false} dataKey="ms" fill="var(--color-ms)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ChartContainer>
            ) : (
              <Empty h="h-64" />
            )}
          </CardContent>
        </Card>

        <Card className="min-w-0 lg:col-span-6">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Вызовы инструментов</CardTitle>
            <CardDescription>harness_tool_calls_total по статусу</CardDescription>
          </CardHeader>
          <CardContent>
            {toolData.length ? (
              <ChartContainer config={toolCfg} className="h-64 w-full">
                <BarChart data={toolData} margin={{ left: 0, right: 8 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="tool" tickLine={false} axisLine={false} fontSize={10} interval={0} tickFormatter={(v: string) => v.replace("_", " ")} />
                  <YAxis tickLine={false} axisLine={false} fontSize={11} width={32} allowDecimals={false} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  {statuses.map((st, i) => (
                    <Bar isAnimationActive={false} key={st} dataKey={st} stackId="s" fill={`var(--color-${st})`} radius={i === statuses.length - 1 ? [4, 4, 0, 0] : 0} />
                  ))}
                </BarChart>
              </ChartContainer>
            ) : (
              <Empty h="h-64" />
            )}
          </CardContent>
        </Card>

        <Card className="min-w-0 lg:col-span-7">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Контекст и шаги harness</CardTitle>
            <CardDescription>Символов в окне агента на запуск · линия — бюджет {fmtNum(BUDGET)}</CardDescription>
          </CardHeader>
          <CardContent>
            {ctx.length ? (
              <ChartContainer config={ctxCfg} className="h-64 w-full">
                <ComposedChart data={ctx} margin={{ left: 4, right: 4 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="i" tickLine={false} axisLine={false} fontSize={11} />
                  <YAxis yAxisId="c" tickLine={false} axisLine={false} fontSize={11} width={44} domain={[0, (max: number) => Math.max(max, BUDGET) * 1.05]} tickFormatter={(v) => `${Math.round(v / 1000)}k`} />
                  <YAxis yAxisId="s" orientation="right" tickLine={false} axisLine={false} fontSize={11} width={24} allowDecimals={false} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  <ReferenceLine yAxisId="c" y={BUDGET} stroke="hsl(var(--chart-5))" strokeDasharray="4 4" />
                  <Bar isAnimationActive={false} yAxisId="c" dataKey="ctx" fill="var(--color-ctx)" radius={[4, 4, 0, 0]} maxBarSize={28} />
                  <Line isAnimationActive={false} yAxisId="s" dataKey="steps" stroke="var(--color-steps)" strokeWidth={2} dot={{ r: 3 }} />
                </ComposedChart>
              </ChartContainer>
            ) : (
              <Empty h="h-64" />
            )}
          </CardContent>
        </Card>

        <Card className="min-w-0 lg:col-span-5">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Лучший score реранкера</CardTitle>
            <CardDescription>rag_top_rerank_score (логиты), запросы /ask</CardDescription>
          </CardHeader>
          <CardContent>
            {hasScores ? (
              <ChartContainer config={{ count: { label: "Запросов", color: "hsl(var(--chart-4))" } }} className="h-64 w-full">
                <BarChart data={scores}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="le" tickLine={false} axisLine={false} fontSize={11} tickFormatter={(v) => `≤${v}`} />
                  <YAxis tickLine={false} axisLine={false} fontSize={11} width={28} allowDecimals={false} />
                  <ChartTooltip content={<ChartTooltipContent labelFormatter={(v) => `score ≤ ${v}`} />} />
                  <Bar isAnimationActive={false} dataKey="count" fill="var(--color-count)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartContainer>
            ) : (
              <Empty h="h-64" />
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Последние запуски</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Время</TableHead>
                <TableHead>Режим</TableHead>
                <TableHead>Вопрос</TableHead>
                <TableHead className="text-right">Длит.</TableHead>
                <TableHead className="text-right">Шаги</TableHead>
                <TableHead className="text-right">Контекст</TableHead>
                <TableHead className="text-right">Неподтв.</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...R].reverse().slice(0, 15).map((r) => (
                <TableRow key={r.id} data-testid={`row-run-${r.id}`}>
                  <TableCell className="whitespace-nowrap font-mono text-xs text-muted-foreground">
                    {new Date(r.timestamp).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                  </TableCell>
                  <TableCell>
                    <Badge variant={r.mode === "harness" ? "default" : "secondary"} className="text-[11px]">
                      {r.mode}
                    </Badge>
                  </TableCell>
                  <TableCell className="max-w-[340px] truncate" title={r.answer}>
                    {r.query}
                  </TableCell>
                  <TableCell className="text-right font-mono">{fmtMs(r.total_ms)}</TableCell>
                  <TableCell className="text-right font-mono">{r.steps ?? "—"}</TableCell>
                  <TableCell className="text-right font-mono">{r.context_chars != null ? fmtNum(r.context_chars) : "—"}</TableCell>
                  <TableCell className="text-right font-mono">{r.unsupported || "—"}</TableCell>
                </TableRow>
              ))}
              {!R.length && (
                <TableRow>
                  <TableCell colSpan={7} className="py-8 text-center text-sm text-muted-foreground">
                    Запусков пока нет.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
