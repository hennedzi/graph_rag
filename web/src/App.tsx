import { Switch, Route, Router, Link, useLocation } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import HarnessPage from "@/pages/harness";
import GraphPage from "@/pages/graph";
import DocumentsPage from "@/pages/documents";
import MetricsPage from "@/pages/metrics";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Harness — агент", icon: "Агент" },
  { href: "/graph", label: "Граф: PPR и пути", icon: "Граф" },
  { href: "/documents", label: "Документы и ETL", icon: "Документы" },
  { href: "/metrics", label: "Метрики", icon: "Метрики" },
];

export default function App() {
  const [loc, go] = useLocation();
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={150}>
        <Router hook={useHashLocation}>
          <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
            <nav className="shrink-0 border-b px-4 py-2 flex gap-2 overflow-x-auto">
              {NAV.map(({ href, label }) => {
                const active = href === "/" ? loc === "/" : loc.startsWith(href);
                return (
                  <Link
                    key={href}
                    href={href}
                    onClick={() => {}}
                    className={cn(
                      "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                      active
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground",
                    )}
                  >
                    {label}
                  </Link>
                );
              })}
            </nav>
            <main className="min-h-0 min-w-0 flex-1 overflow-y-auto p-6">
              <Switch>
                <Route path="/" component={HarnessPage} />
                <Route path="/graph" component={GraphPage} />
                <Route path="/documents" component={DocumentsPage} />
                <Route path="/metrics">
                  <MetricsPage />
                </Route>
                <Route>
                  <p className="p-6 text-muted-foreground">Страница не найдена.</p>
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
