import { ShieldAlert, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import type { Citation, Claim, Unsupported } from "@/lib/api";

export function SourceBadge({ h, cite }: { h: string; cite?: Citation }) {
  return (
    <HoverCard openDelay={80} closeDelay={60}>
      <HoverCardTrigger asChild>
        <button type="button" data-testid={`badge-source-${h}`}>
          <Badge variant="outline" className="cursor-help border-primary/40 font-mono text-[11px] text-primary">
            {h}
          </Badge>
        </button>
      </HoverCardTrigger>
      <HoverCardContent className="w-96 max-w-[90vw]">
        <div className="mb-2 flex items-center justify-between gap-2 font-mono text-xs text-muted-foreground">
          <span className="text-primary">{h}</span>
          <span className="truncate">{cite?.doc || cite?.doc_id}</span>
        </div>
        <p className="max-h-64 overflow-y-auto text-sm leading-relaxed">{cite?.text || "Текст пассажа не загружен."}</p>
      </HoverCardContent>
    </HoverCard>
  );
}

export function Claims({ claims, answer, citations }: { claims: Claim[]; answer: string; citations: Record<string, Citation> }) {
  if (!claims.length) return <p className="whitespace-pre-wrap text-sm leading-relaxed">{answer}</p>;
  return (
    <ol className="space-y-3">
      {claims.map((c, i) => (
        <li key={i} className="flex gap-3" data-testid={`claim-${i}`}>
          <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/15 font-mono text-[11px] text-primary">
            {i + 1}
          </span>
          <div className="space-y-1.5">
            <p className="text-sm leading-relaxed">{c.text}</p>
            <div className="flex flex-wrap gap-1">
              {c.sources.map((h) => (
                <SourceBadge key={h} h={h} cite={citations[h]} />
              ))}
            </div>
          </div>
        </li>
      ))}
    </ol>
  );
}

export function Verdict({ unsupported }: { unsupported: Unsupported[] }) {
  if (!unsupported.length)
    return (
      <div className="flex items-center gap-2 rounded-md border border-primary/30 bg-primary/10 px-3 py-2 text-sm text-primary" data-testid="status-verified">
        <ShieldCheck className="h-4 w-4" /> Все утверждения подтверждены источниками
      </div>
    );
  return (
    <div className="space-y-1 rounded-md border border-[hsl(var(--chart-5)/0.35)] bg-[hsl(var(--chart-5)/0.1)] px-3 py-2 text-sm" data-testid="status-unsupported">
      <div className="flex items-center gap-2 font-medium text-[hsl(var(--chart-5))]">
        <ShieldAlert className="h-4 w-4" /> Не подтверждено: {unsupported.length}
      </div>
      {unsupported.map((u, i) => (
        <p key={i} className="text-muted-foreground">
          {u.claim}
          {u.reason ? ` — ${u.reason}` : ""}
        </p>
      ))}
    </div>
  );
}
