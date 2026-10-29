import { useEffect, useMemo, useRef, useState } from "react";
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY, type SimulationNodeDatum } from "d3-force";
import type { Edge } from "@/lib/api";
import { cn } from "@/lib/utils";

type Node = SimulationNodeDatum & { id: string; degree: number; born: number };
type Link = { source: Node; target: Node; text: string; pred: string; born: number };

/** «иоганн бернулли учитель леонард эйлер» → «учитель» (текст связи = src + предикат + dst). */
export function predicate(e: Edge) {
  const t = e.text.trim();
  if (t.startsWith(e.src) && t.endsWith(e.dst)) return t.slice(e.src.length, t.length - e.dst.length).trim() || t;
  return t;
}

export function GraphView({ edges, focus, height = 380 }: { edges: Edge[]; focus?: string[]; height?: number }) {
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(480);
  const H = height;
  useEffect(() => {
    // viewBox = реальный размер в px: шрифт и радиусы не мельчают при росте графа
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.round(e.contentRect.width))));
    if (box.current) ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  const nodes = useRef(new Map<string, Node>());
  const [, tick] = useState(0);
  const [hover, setHover] = useState<string | null>(null);
  const tickCount = useRef(0);

  const links: Link[] = useMemo(() => {
    const m = nodes.current;
    const now = Date.now();
    const seen = new Set<string>();
    const out: Link[] = [];
    for (const e of edges) {
      for (const id of [e.src, e.dst])
        if (!m.has(id)) m.set(id, { id, degree: 0, born: now, x: W / 2 + (Math.random() - 0.5) * 80, y: H / 2 + (Math.random() - 0.5) * 80 });
      const key = `${e.src}|${e.dst}|${e.text}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ source: m.get(e.src)!, target: m.get(e.dst)!, text: e.text, pred: predicate(e), born: now });
    }
    for (const n of m.values()) n.degree = 0;
    for (const l of out) {
      l.source.degree++;
      l.target.degree++;
    }
    for (const id of [...m.keys()]) if (!m.get(id)!.degree) m.delete(id);
    return out;
  }, [edges]);

  useEffect(() => {
    const list = [...nodes.current.values()];
    if (!list.length) return;
    const sim = forceSimulation(list)
      .force("link", forceLink<Node, Link>(links).distance(85).strength(0.35))
      .force("charge", forceManyBody().strength(-380))
      .force("center", forceCenter(W / 2, H / 2))
      .force("x", forceX(W / 2).strength(0.04))
      .force("y", forceY(H / 2).strength(0.06))
      .force("collide", forceCollide(30))
      .alpha(0.9)
      .on("tick", () => {
        for (const n of list) {
          n.x = Math.max(40, Math.min(W - 40, n.x ?? 0));
          n.y = Math.max(16, Math.min(H - 26, n.y ?? 0));
        }
        tickCount.current++;
        if (tickCount.current % 2 === 0) tick((t) => t + 1);
      });
    return () => {
      sim.stop();
    };
  }, [links, W, H]);

  const list = [...nodes.current.values()];
  if (!list.length)
    return (
      <div ref={box} className="flex items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground" style={{ height }}>
        Граф появится, когда агент начнёт ходить по связям
      </div>
    );
  const focusSet = new Set(focus ?? []);
  const maxDeg = Math.max(...list.map((n) => n.degree));
  const near = (id: string) => hover === id || links.some((l) => (l.source.id === hover && l.target.id === id) || (l.target.id === hover && l.source.id === id));
  const fresh = (born: number) => Date.now() - born < 2500;

  return (
    <div ref={box}>
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full select-none rounded-lg bg-[radial-gradient(circle_at_center,hsl(var(--primary)/0.07),transparent_70%)]"
      style={{ height }}
      data-testid="graph-view"
    >
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0 0 10 5 0 10z" className="fill-muted-foreground/60" />
        </marker>
      </defs>
      {links.map((l, i) => {
        const on = hover && (l.source.id === hover || l.target.id === hover);
        return (
          <g key={i} opacity={hover && !on ? 0.15 : 1}>
            <line
              x1={l.source.x}
              y1={l.source.y}
              x2={l.target.x}
              y2={l.target.y}
              markerEnd="url(#arrow)"
              strokeWidth={on ? 2 : 1.2}
              className={fresh(l.born) ? "stroke-primary" : "stroke-muted-foreground/40"}
            >
              <title>{l.text}</title>
            </line>
            {(on || links.length <= 10) && (
              <text
                x={((l.source.x ?? 0) + (l.target.x ?? 0)) / 2}
                y={((l.source.y ?? 0) + (l.target.y ?? 0)) / 2 - 4}
                textAnchor="middle"
                className="fill-muted-foreground font-mono"
                fontSize={9}
              >
                {l.pred.length > 22 ? l.pred.slice(0, 21) + "…" : l.pred}
              </text>
            )}
          </g>
        );
      })}
      {list.map((n) => {
        const r = 6 + (n.degree / maxDeg) * 8;
        const isFocus = focusSet.has(n.id);
        return (
          <g
            key={n.id}
            transform={`translate(${n.x},${n.y})`}
            onMouseEnter={() => setHover(n.id)}
            onMouseLeave={() => setHover(null)}
            opacity={hover && !near(n.id) ? 0.25 : 1}
            className="cursor-default"
          >
            {fresh(n.born) && <circle r={r + 6} className="animate-ping fill-primary/30" style={{ transformBox: "fill-box", transformOrigin: "center" }} />}
            <circle
              r={r}
              strokeWidth={2}
              className={isFocus ? "fill-primary stroke-primary/40" : "fill-[hsl(var(--chart-2))] stroke-background"}
            />
          </g>
        );
      })}
      {list.map((n) => {
        const r = 6 + (n.degree / maxDeg) * 8;
        const show = !hover ? focusSet.has(n.id) || n.degree > 1 || list.length <= 14 : near(n.id);
        if (!show) return null;
        return (
          <text
            key={n.id}
            x={n.x}
            y={(n.y ?? 0) + r + 12}
            textAnchor="middle"
            fontSize={11}
            className={cn("pointer-events-none font-medium", focusSet.has(n.id) ? "fill-primary" : "fill-foreground")}
            style={{ paintOrder: "stroke", stroke: "hsl(var(--card))", strokeWidth: 3 }}
          >
            {n.id.length > 22 ? n.id.slice(0, 21) + "…" : n.id}
          </text>
        );
      })}
    </svg>
    </div>
  );
}
