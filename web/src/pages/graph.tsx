import { useEffect, useState } from "react";
import { useSearch } from "wouter";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Loader2, Orbit, Route as RouteIcon, X } from "lucide-react";
import { ListItem, SectionLabel, Workspace, useStatus } from "@/components/shell";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { GraphView, predicate } from "@/components/graph-view";
import { fmtMs, fmtNum, getJson, postJson, type Edge, type GraphExplore, type GraphInfo, type GraphStats } from "@/lib/api";
import { cn } from "@/lib/utils";

const ROWS: [keyof GraphStats, string][] = [
  ["nodes", "вершины"],
  ["edges", "рёбра"],
  ["avg_degree", "ср. степень"],
  ["components", "компоненты"],
  ["largest_component", "крупнейшая"],
];

function StatsTable({ info }: { info?: GraphInfo }) {
  return (
    <div className="px-3">
      <table className="w-full font-mono text-[11.5px]">
        <thead>
          <tr className="text-muted-foreground">
            <th className="py-1 text-left font-normal" />
            <th className="py-1 text-right font-normal">до склейки</th>
            <th className="py-1 text-right font-normal">после</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map(([k, l]) => (
            <tr key={k} className="border-t border-dashed">
              <td className="py-1 font-sans text-xs text-muted-foreground">{l}</td>
              <td className="py-1 text-right">{fmtNum(info?.stats_raw[k] as number)}</td>
              <td className={cn("py-1 text-right", info && info.stats[k] !== info.stats_raw[k] && "text-primary")}>{fmtNum(info?.stats[k] as number)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const pathText = (p: Edge[], start: string) => {
  let at = start;
  return p.reduce((s, e) => {
    const fwd = e.src === at;
    at = fwd ? e.dst : e.src;
    return `${s} ${fwd ? `—${predicate(e)}→` : `←${predicate(e)}—`} ${at}`;
  }, start);
};

export default function GraphPage() {
  const info = useQuery({ queryKey: ["graph-stats"], queryFn: () => getJson<GraphInfo>("/graph/stats") });
  const [seeds, setSeeds] = useState<string[]>([]);
  const [text, setText] = useState("");
  const [path, setPath] = useState<number | null>(null);
  const explore = useMutation({ mutationFn: (entities: string[]) => postJson<GraphExplore>("/graph/explore", { entities, k: 15 }) });
  const r = explore.data;

  const go = (list: string[]) => {
    const clean = [...new Set(list.map((x) => x.trim()).filter(Boolean))].slice(0, 5);
    setSeeds(clean);
    setPath(null);
    if (clean.length) explore.mutate(clean, { onSuccess: (d) => setSeeds(d.seeds.length ? d.seeds : clean) });
  };
  const search = useSearch();
  useEffect(() => {
    // переход из «Документов»: #/graph?e=сущность
    const e = new URLSearchParams(search).getAll("e");
    if (e.length) go(e);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search]);
  const toggleSeed = (id: string) => go(seeds.includes(id) ? seeds.filter((s) => s !== id) : [...seeds, id]);
  const weights = Object.fromEntries((r?.ranking ?? []).map((x) => [x.entity, x.score]));
  const lit = path != null && r ? new Set(r.paths[path].map((e) => `${e.src}|${e.dst}`)) : undefined;
  const maxScore = Math.max(1e-9, ...(r?.ranking ?? []).map((x) => x.score));

  useStatus(
    r && (
      <>
        <span>seed {r.seeds.length}</span>
        <span>топ {r.ranking.length}</span>
        <span>путей {r.paths.length}</span>
        <span>{fmtMs(Object.values(r.timings_ms).reduce((a, b) => a + b, 0))}</span>
      </>
    ),
    [r],
  );

  return (
    <Workspace
      id="graph"
      title="Граф: Personalized PageRank и пути"
      sidebarTitle="Структура графа"
      sidebar={
        <div className="pb-3">
          <SectionLabel>Канонизация сущностей</SectionLabel>
          <StatsTable info={info.data} />
          <p className="px-3 pt-2 text-[11px] text-muted-foreground">
            Однословное имя склеивается с полным, если полное имя с такой фамилией одно («эйлер» → «леонард эйлер»).
          </p>
          <SectionLabel>Псевдонимы {info.data && `(${Object.keys(info.data.aliases).length})`}</SectionLabel>
          {info.data && !Object.keys(info.data.aliases).length && <p className="px-3 text-xs text-muted-foreground">Не найдено — граф не фрагментирован по фамилиям.</p>}
          {Object.entries(info.data?.aliases ?? {}).map(([a, c]) => (
            <ListItem key={a} title={`${a} → ${c}`} onClick={() => go([c])} />
          ))}
        </div>
      }
      inspector={
        <div className="flex h-full flex-col">
          <div className="flex h-9 shrink-0 items-center gap-2 border-b px-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            <Orbit className="h-3.5 w-3.5" /> Ранжирование PPR
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {r ? (
              <>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="h-8 text-xs">Сущность</TableHead>
                      <TableHead className="h-8 text-right text-xs">масса</TableHead>
                      <TableHead className="h-8 text-right text-xs">пасс.</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {r.ranking.map((x) => (
                      <TableRow key={x.entity} className="cursor-pointer" onClick={() => toggleSeed(x.entity)} data-testid={`rank-${x.entity}`}>
                        <TableCell className="py-1.5 text-xs">
                          <span className={cn(r.seeds.includes(x.entity) && "font-semibold text-primary")}>{x.entity}</span>
                          <span className="mt-1 block h-1 rounded-full bg-muted">
                            <span className="block h-full rounded-full bg-[hsl(var(--chart-2))]" style={{ width: `${(x.score / maxScore) * 100}%` }} />
                          </span>
                        </TableCell>
                        <TableCell className="py-1.5 text-right font-mono text-xs">{(x.score * 100).toFixed(1)}%</TableCell>
                        <TableCell className="py-1.5 text-right font-mono text-xs text-muted-foreground">{x.passages}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <SectionLabel>Пассажи по массе сущностей</SectionLabel>
                <div className="space-y-2 px-3 pb-3">
                  {r.passages.map((p) => (
                    <div key={p.id} className="space-y-1 rounded-md border p-2.5">
                      <div className="flex justify-between font-mono text-[11px] text-muted-foreground">
                        <span className="truncate">{p.doc}</span>
                        <span>{(p.score * 100).toFixed(2)}%</span>
                      </div>
                      <p className="line-clamp-4 text-xs leading-relaxed">{p.text}</p>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="p-6 text-center text-sm text-muted-foreground">Задайте seed-сущности — здесь будет их окрестность, ранжированная Personalized PageRank.</p>
            )}
          </div>
        </div>
      }
    >
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b px-3 py-2">
        {seeds.map((s) => (
          <span key={s} className="flex items-center gap-1 rounded-md border border-primary/40 bg-primary/10 px-2 py-0.5 text-xs text-primary">
            {s}
            <button type="button" onClick={() => toggleSeed(s)} aria-label={`Убрать ${s}`}>
              <X className="h-3 w-3" />
            </button>
          </span>
        ))}
        <form
          className="flex min-w-[220px] flex-1 gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            go([...seeds, ...text.split(",")]);
            setText("");
          }}
        >
          <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="Сущности через запятую: Эйлер, Лагранж" className="h-8 text-sm" data-testid="input-seeds" />
          <Button type="submit" size="sm" className="h-8" disabled={explore.isPending || (!text.trim() && !seeds.length)} data-testid="button-explore">
            {explore.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Orbit className="h-4 w-4" />} PPR
          </Button>
        </form>
      </div>
      {explore.error && <p className="px-3 py-2 font-mono text-sm text-destructive">{(explore.error as Error).message}</p>}
      <ResizablePanelGroup direction="vertical" autoSaveId="graph-center" className="min-h-0 flex-1">
        <ResizablePanel defaultSize={72} minSize={30}>
          <div className="h-full p-2">
            {r ? (
              <GraphView edges={r.edges} focus={r.seeds} weights={weights} highlight={lit} onNode={toggleSeed} />
            ) : (
              <div className="flex h-full items-center justify-center p-6 text-center text-sm text-muted-foreground">
                Размер вершины — масса Personalized PageRank от seed. Клик по вершине добавляет/убирает её из seed.
              </div>
            )}
          </div>
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel defaultSize={28} minSize={10} collapsible>
          <div className="flex h-full flex-col">
            <div className="flex h-8 shrink-0 items-center gap-2 border-b px-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              <RouteIcon className="h-3.5 w-3.5" /> Пути {r?.seeds.length ? `${r.seeds[0]} → ${r.seeds[1] ?? "…"}` : ""}
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto py-1">
              {r && r.seeds.length < 2 && <p className="px-3 py-2 text-xs text-muted-foreground">Добавьте вторую сущность — найдём цепочки связей между первыми двумя.</p>}
              {r && r.seeds.length > 1 && !r.paths.length && <p className="px-3 py-2 text-xs text-muted-foreground">Путей до 3 хопов нет.</p>}
              {r?.paths.map((p, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setPath(path === i ? null : i)}
                  className={cn("block w-full border-l-2 border-transparent px-3 py-1 text-left font-mono text-[12px] hover:bg-secondary/60", path === i && "border-[hsl(var(--chart-3))] bg-secondary")}
                  data-testid={`path-${i}`}
                >
                  <span className="mr-2 text-muted-foreground">{p.length} хоп.</span>
                  {pathText(p, r.seeds[0])}
                </button>
              ))}
            </div>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </Workspace>
  );
}
