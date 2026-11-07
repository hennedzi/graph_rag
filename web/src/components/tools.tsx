import { BookOpen, CheckCheck, Database, FileSearch, Orbit, Route, ScanSearch, Share2, Wrench, type LucideIcon } from "lucide-react";

export const TOOLS: Record<string, { icon: LucideIcon; label: string; color: string; hint: string }> = {
  find_entities: { icon: ScanSearch, label: "find_entities", color: "var(--chart-4)", hint: "сущности по смыслу" },
  neighbors: { icon: Share2, label: "neighbors", color: "var(--chart-2)", hint: "связи графа, 1–2 хопа" },
  rank_by_ppr: { icon: Orbit, label: "rank_by_ppr", color: "var(--chart-2)", hint: "Personalized PageRank по графу" },
  find_paths: { icon: Route, label: "find_paths", color: "var(--chart-4)", hint: "цепочки связей между двумя" },
  query_db: { icon: Database, label: "query_db", color: "var(--chart-3)", hint: "фильтр-выражение Milvus" },
  search_passages: { icon: FileSearch, label: "search_passages", color: "var(--chart-1)", hint: "гибрид dense+BM25 + реранк" },
  read_passage: { icon: BookOpen, label: "read_passage", color: "var(--chart-5)", hint: "полный текст по P#" },
  answer: { icon: CheckCheck, label: "answer", color: "var(--primary)", hint: "утверждения + источники" },
};

export const toolMeta = (name: string) => TOOLS[name] ?? { icon: Wrench, label: name, color: "var(--muted-foreground)", hint: "" };

export const STATUS: Record<string, { label: string; tone: "ok" | "warn" | "bad" }> = {
  ok: { label: "ok", tone: "ok" },
  bad_query: { label: "ошибка фильтра", tone: "warn" },
  bad_args: { label: "ошибка аргументов", tone: "warn" },
  budget: { label: "бюджет исчерпан", tone: "warn" },
  unknown_tool: { label: "нет инструмента", tone: "warn" },
  bad_answer: { label: "отклонён кодом", tone: "bad" },
  unsupported: { label: "не подтверждён", tone: "bad" },
};

export const toneClass = {
  ok: "border-primary/30 bg-primary/10 text-primary",
  warn: "border-[hsl(var(--chart-3)/0.35)] bg-[hsl(var(--chart-3)/0.12)] text-[hsl(var(--chart-3))]",
  bad: "border-[hsl(var(--chart-5)/0.35)] bg-[hsl(var(--chart-5)/0.12)] text-[hsl(var(--chart-5))]",
};
