import { useEffect, useState, type ReactNode } from "react";
import { Switch, Route, Router, Link, useLocation } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import { Activity, Bot, FileUp, Moon, Network, Sun } from "lucide-react";
import { queryClient } from "./lib/queryClient";
import { getJson, fmtNum, type Counts } from "./lib/api";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider, Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Logo } from "@/components/logo";
import { StatusSlot, kbd } from "@/components/shell";
import HarnessPage from "@/pages/harness";
import GraphPage from "@/pages/graph";
import DocumentsPage from "@/pages/documents";
import MetricsPage from "@/pages/metrics";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Harness — агент", icon: Bot },
  { href: "/graph", label: "Граф: PPR и пути", icon: Network },
  { href: "/documents", label: "Документы и ETL", icon: FileUp },
  { href: "/metrics", label: "Метрики", icon: Activity },
];

// Панель активности слева, как в VS Code / Zed: иконки разделов, активный — с полосой слева.
function ActivityBar({ dark, onTheme }: { dark: boolean; onTheme: () => void }) {
  const [loc, go] = useLocation();
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const n = Number(e.key);
      if ((e.metaKey || e.ctrlKey) && !e.altKey && n >= 1 && n <= NAV.length) {
        e.preventDefault();
        go(NAV[n - 1].href);
      }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [go]);
  return (
    <nav className="flex w-12 shrink-0 flex-col items-center gap-1 border-r bg-sidebar/60 py-2" aria-label="Разделы">
      <Link href="/" className="mb-2 flex h-9 w-9 items-center justify-center" data-testid="link-home" aria-label="Graph RAG">
        <Logo />
      </Link>
      {NAV.map(({ href, label, icon: Icon }, i) => {
        const active = href === "/" ? loc === "/" : loc.startsWith(href);
        return (
          <Tooltip key={href}>
            <TooltipTrigger asChild>
              <Link
                href={href}
                aria-label={label}
                data-testid={`nav-${href.slice(1) || "harness"}`}
                className={cn(
                  "relative flex h-10 w-10 items-center justify-center rounded-md text-muted-foreground hover:bg-secondary hover:text-foreground",
                  active && "text-foreground before:absolute before:-left-1 before:top-2 before:h-6 before:w-0.5 before:rounded-full before:bg-primary",
                )}
              >
                <Icon className="h-5 w-5" />
              </Link>
            </TooltipTrigger>
            <TooltipContent side="right">
              {label} <span className="ml-1 font-mono text-muted-foreground">{kbd(`Ctrl+${i + 1}`)}</span>
            </TooltipContent>
          </Tooltip>
        );
      })}
      <button
        type="button"
        onClick={onTheme}
        className="mt-auto flex h-10 w-10 items-center justify-center rounded-md text-muted-foreground hover:bg-secondary hover:text-foreground"
        data-testid="button-theme"
        aria-label="Сменить тему"
      >
        {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      </button>
    </nav>
  );
}

// Строка состояния: здоровье API и счётчики Milvus всегда, справа — показатели текущей страницы.
function StatusBar({ slot }: { slot: ReactNode }) {
  const health = useQuery({ queryKey: ["health"], queryFn: () => getJson<{ ok: boolean }>("/health"), refetchInterval: 10000 });
  const stats = useQuery({ queryKey: ["stats"], queryFn: () => getJson<Counts>("/stats"), refetchInterval: 15000 });
  const ok = health.data?.ok;
  return (
    <footer className="flex h-6 shrink-0 items-center gap-4 border-t bg-sidebar/60 px-3 font-mono text-[11px] text-muted-foreground" data-testid="status-bar">
      <span className="flex items-center gap-1.5" data-testid="status-health">
        <span className={cn("h-2 w-2 rounded-full", ok ? "bg-primary" : "bg-destructive")} />
        {ok ? "API" : "API недоступен"}
      </span>
      <span className="hidden gap-3 sm:flex" data-testid="text-db-counts">
        <span>пассажи {fmtNum(stats.data?.passage)}</span>
        <span>сущности {fmtNum(stats.data?.entity)}</span>
        <span>связи {fmtNum(stats.data?.relation)}</span>
      </span>
      <span className="ml-auto flex min-w-0 items-center gap-4 truncate">{slot}</span>
    </footer>
  );
}

const Scroll = ({ children }: { children: ReactNode }) => (
  <div className="h-full overflow-y-auto">
    <div className="mx-auto max-w-[1440px] px-4 py-6 md:px-6">{children}</div>
  </div>
);

export default function App() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const [slot, setSlot] = useState<ReactNode>(null);
  const toggle = () => {
    document.documentElement.classList.toggle("dark", !dark);
    localStorage.setItem("theme", dark ? "light" : "dark");
    setDark(!dark);
  };
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={150}>
        <StatusSlot.Provider value={setSlot}>
          <Router hook={useHashLocation}>
            <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
              <div className="flex min-h-0 flex-1">
                <ActivityBar dark={dark} onTheme={toggle} />
                <main className="min-w-0 flex-1">
                  <Switch>
                    <Route path="/" component={HarnessPage} />
                    <Route path="/graph" component={GraphPage} />
                    <Route path="/documents" component={DocumentsPage} />
                    <Route path="/metrics">
                      <Scroll>
                        <MetricsPage />
                      </Scroll>
                    </Route>
                    <Route>
                      <p className="p-6 text-muted-foreground">Страница не найдена.</p>
                    </Route>
                  </Switch>
                </main>
              </div>
              <StatusBar slot={slot} />
            </div>
          </Router>
        </StatusSlot.Provider>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}
