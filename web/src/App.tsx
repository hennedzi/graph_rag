import { useState } from "react";
import { Switch, Route, Router, Link, useLocation } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import { Activity, Bot, FileUp, Moon, Search, Sun } from "lucide-react";
import { queryClient } from "./lib/queryClient";
import { getJson, fmtNum, type Counts } from "./lib/api";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider, Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Logo } from "@/components/logo";
import HarnessPage from "@/pages/harness";
import SearchPage from "@/pages/search";
import DocumentsPage from "@/pages/documents";
import MetricsPage from "@/pages/metrics";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Harness", icon: Bot },
  { href: "/search", label: "Поиск", icon: Search },
  { href: "/documents", label: "Документы", icon: FileUp },
  { href: "/metrics", label: "Метрики", icon: Activity },
];

function Header() {
  const [loc] = useLocation();
  const [dark, setDark] = useState(true);
  const health = useQuery({ queryKey: ["health"], queryFn: () => getJson<{ ok: boolean }>("/health"), refetchInterval: 10000 });
  const stats = useQuery({ queryKey: ["stats"], queryFn: () => getJson<Counts>("/stats"), refetchInterval: 15000 });
  const toggle = () => {
    document.documentElement.classList.toggle("dark", !dark);
    setDark(!dark);
  };
  return (
    <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1440px] items-center gap-4 px-4 md:px-6">
        <Link href="/" className="flex items-center gap-2" data-testid="link-home">
          <Logo />
          <span className="hidden font-semibold tracking-tight sm:inline">Graph RAG</span>
        </Link>
        <nav className="flex items-center gap-1 overflow-x-auto">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? loc === "/" : loc.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                data-testid={`nav-${href.slice(1) || "harness"}`}
                className={cn(
                  "hover-elevate flex items-center gap-2 rounded-md px-3 py-1.5 text-sm text-muted-foreground",
                  active && "bg-secondary text-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden md:inline">{label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-3">
          <div className="hidden items-center gap-3 font-mono text-xs text-muted-foreground lg:flex" data-testid="text-db-counts">
            <span>пассажи {fmtNum(stats.data?.passage)}</span>
            <span>сущности {fmtNum(stats.data?.entity)}</span>
            <span>связи {fmtNum(stats.data?.relation)}</span>
          </div>
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="relative flex h-2.5 w-2.5" data-testid="status-health">
                {health.data?.ok && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />}
                <span className={cn("relative inline-flex h-2.5 w-2.5 rounded-full", health.data?.ok ? "bg-primary" : "bg-destructive")} />
              </span>
            </TooltipTrigger>
            <TooltipContent>{health.data?.ok ? "API на связи" : "API недоступен"}</TooltipContent>
          </Tooltip>
          <Button size="icon" variant="ghost" onClick={toggle} data-testid="button-theme" aria-label="Сменить тему">
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </header>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={150}>
        <Router hook={useHashLocation}>
          <div className="min-h-dvh bg-background text-foreground">
            <Header />
            <main className="mx-auto max-w-[1440px] px-4 py-6 md:px-6">
              <Switch>
                <Route path="/" component={HarnessPage} />
                <Route path="/search" component={SearchPage} />
                <Route path="/documents" component={DocumentsPage} />
                <Route path="/metrics" component={MetricsPage} />
                <Route>
                  <p className="text-muted-foreground">Страница не найдена.</p>
                </Route>
              </Switch>
            </main>
          </div>
        </Router>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}
