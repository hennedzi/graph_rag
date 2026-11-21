import { Fragment } from "react";
import { ShieldAlert, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import type { Citation, Outcome, Unsupported } from "@/lib/api";

export const OUTCOME_TITLE: Record<Outcome, string> = {
  answered: "Ответ подтверждён источниками",
  partial: "Ответ частичный",
  no_evidence: "В документах нет ответа",
  verification_failed: "Проверка не выполнена",
  budget_exceeded: "Достигнут лимит шагов",
  cancelled: "Остановлено",
  dependency_error: "Недоступна модель",
};

export function SourceBadge({ h, cite }: { h: string; cite?: Citation }) {
  return (
    <HoverCard openDelay={80} closeDelay={60}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          data-testid={`badge-source-${h}`}
          aria-label={`Источник ${h}${cite?.doc ? `, документ ${cite.doc}` : ""}`}
          className="inline-flex align-baseline"
        >
          <Badge variant="outline" className="cursor-help border-primary/40 font-mono text-[11px] text-primary">
            {h}
          </Badge>
        </button>
      </HoverCardTrigger>
      <HoverCardContent side="top" align="start" className="flex max-h-[min(22rem,var(--radix-hover-card-content-available-height))] w-[min(26rem,calc(100vw-24px))] flex-col p-0">
        <div className="flex items-center justify-between gap-3 border-b px-3 py-2 font-mono text-xs text-muted-foreground">
          <span className="shrink-0 text-primary">{h}</span>
          <span className="min-w-0 truncate">{cite?.doc || cite?.doc_id}</span>
        </div>
        <p className="min-h-0 overflow-y-auto whitespace-pre-wrap break-words px-3 py-2.5 text-sm leading-relaxed">{cite?.text || "Текст пассажа не загружен."}</p>
      </HoverCardContent>
    </HoverCard>
  );
}

// Инлайн-ссылка [P#] внутри текста → кликабельный бейдж с текстом пассажа.
function withCitations(text: string, citations: Record<string, Citation>) {
  return text.split(/(\[(?:P\d+)\])/g).map((part, i) => {
    const m = part.match(/^\[(P\d+)\]$/);
    return m ? <SourceBadge key={i} h={m[1]} cite={citations[m[1]]} /> : <Fragment key={i}>{part}</Fragment>;
  });
}

// Ответ — связный текст абзацами с инлайн-ссылками [P#], а не нумерованный список тезисов.
export function ProseAnswer({ text, citations }: { text: string; citations: Record<string, Citation> }) {
  const paras = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  return (
    <div className="space-y-3 text-[15px] leading-relaxed text-foreground" data-testid="answer-prose">
      {paras.map((p, i) => (
        <p key={i} className="whitespace-pre-wrap break-words">
          {withCitations(p, citations)}
        </p>
      ))}
    </div>
  );
}

// Список ссылок внизу ответа: на основе каких пассажей он построен.
export function SourcesList({ citations }: { citations: Record<string, Citation> }) {
  const items = Object.entries(citations);
  if (!items.length) return null;
  return (
    <div className="space-y-2 border-t pt-3" data-testid="answer-sources">
      <p className="text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground">
        Источники · {items.length}
      </p>
      <div className="grid gap-1.5 sm:grid-cols-2">
        {items.map(([h, c]) => (
          <div key={h} className="flex min-w-0 gap-2 rounded-lg border bg-muted/30 p-2">
            <SourceBadge h={h} cite={c} />
            <div className="min-w-0 flex-1">
              <p className="truncate font-mono text-[10.5px] text-muted-foreground">{c.doc || c.doc_id}</p>
              <p className="line-clamp-2 text-[11px] leading-snug text-muted-foreground">{c.text}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Зелёный статус — только при outcome=answered с backend: пустой unsupported не доказывает, что проверка была.
export function Verdict({ unsupported, outcome, note }: { unsupported: Unsupported[]; outcome?: Outcome; note?: string }) {
  if (!unsupported.length && outcome !== "answered")
    return (
      <div className="flex items-start gap-2 rounded-md border border-[hsl(var(--chart-5)/0.35)] bg-[hsl(var(--chart-5)/0.1)] px-3 py-2 text-sm" data-testid={`status-${outcome ?? "unknown"}`}>
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-[hsl(var(--chart-5))]" />
        <span>
          <span className="font-medium text-[hsl(var(--chart-5))]">{outcome ? OUTCOME_TITLE[outcome] ?? outcome : "Статус проверки не сохранён"}</span>
          <span className="text-muted-foreground"> — {note || (outcome ? "подробности в ходе работы" : "старый запуск, диагностика неполная")}</span>
        </span>
      </div>
    );
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
