import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import type { ImperativePanelHandle } from "react-resizable-panels";
import { PanelLeft, PanelRight } from "lucide-react";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

// Слот строки состояния: страница кладёт сюда свои показатели (сессия, окно, шаг), App рисует внизу.
export const StatusSlot = createContext<(n: ReactNode) => void>(() => {});
export function useStatus(node: ReactNode, deps: unknown[]) {
  const set = useContext(StatusSlot);
  useEffect(() => set(node), deps); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => set(null), [set]);
}

export const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
export const kbd = (k: string) => (isMac ? k.replace("Ctrl+", "⌘").replace("Alt+", "⌥").replace("Shift+", "⇧") : k);

export function useMedia(q: string) {
  const [m, setM] = useState(() => window.matchMedia(q).matches);
  useEffect(() => {
    const mq = window.matchMedia(q);
    const on = () => setM(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [q]);
  return m;
}

/** Заголовок панели как в Zed/VS Code: мелкий капс слева, действия справа. */
export function PaneHeader({ title, icon, children, className }: { title: ReactNode; icon?: ReactNode; children?: ReactNode; className?: string }) {
  return (
    <div className={cn("flex h-9 shrink-0 items-center gap-2 border-b px-3", className)}>
      {icon}
      <span className="truncate text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{title}</span>
      <div className="ml-auto flex items-center gap-1">{children}</div>
    </div>
  );
}

function IconToggle({ on, onClick, label, hint, children }: { on: boolean; onClick: () => void; label: string; hint: string; children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          aria-label={label}
          aria-pressed={on}
          className={cn("rounded p-1 text-muted-foreground hover:bg-secondary hover:text-foreground", on && "text-foreground")}
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent>
        {label} <span className="ml-1 font-mono text-muted-foreground">{hint}</span>
      </TooltipContent>
    </Tooltip>
  );
}

/**
 * Рабочая область IDE: [боковая панель | центр | инспектор], ширины тянутся мышью и запоминаются (autoSaveId).
 * Ctrl/⌘+B — боковая панель, Ctrl/⌘+Alt+B — инспектор (как в VS Code/Zed). На узком экране — одна колонка.
 */
export function Workspace({
  id,
  title,
  actions,
  sidebar,
  sidebarTitle,
  inspector,
  children,
  footer,
}: {
  id: string;
  title: ReactNode;
  actions?: ReactNode;
  sidebar?: ReactNode;
  sidebarTitle?: ReactNode;
  inspector?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
}) {
  const wide = useMedia("(min-width: 1024px)");
  const left = useRef<ImperativePanelHandle>(null);
  const right = useRef<ImperativePanelHandle>(null);
  const [open, setOpen] = useState({ left: true, right: true });
  const toggle = (p: typeof left) => (p.current?.isCollapsed() ? p.current.expand() : p.current?.collapse());

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.code !== "KeyB") return;
      e.preventDefault();
      toggle(e.altKey ? right : left);
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, []);

  const center = (
    <div className="flex h-full min-w-0 flex-col">
      <PaneHeader title={title}>
        {actions}
        {wide && sidebar && (
          <IconToggle on={open.left} onClick={() => toggle(left)} label="Боковая панель" hint={kbd("Ctrl+B")}>
            <PanelLeft className="h-4 w-4" />
          </IconToggle>
        )}
        {wide && inspector && (
          <IconToggle on={open.right} onClick={() => toggle(right)} label="Инспектор" hint={kbd("Ctrl+Alt+B")}>
            <PanelRight className="h-4 w-4" />
          </IconToggle>
        )}
      </PaneHeader>
      {children}
      {footer}
    </div>
  );

  if (!wide)
    return (
      <div className="h-full overflow-y-auto">
        <div className="flex h-[calc(100dvh-1.5rem)] flex-col">{center}</div>
        {inspector && <div className="border-t">{inspector}</div>}
      </div>
    );

  return (
    <ResizablePanelGroup direction="horizontal" autoSaveId={`ws-${id}`}>
      {sidebar && (
        <>
          <ResizablePanel
            ref={left}
            order={1}
            defaultSize={18}
            minSize={12}
            maxSize={32}
            collapsible
            collapsedSize={0}
            onCollapse={() => setOpen((o) => ({ ...o, left: false }))}
            onExpand={() => setOpen((o) => ({ ...o, left: true }))}
            className="bg-sidebar/40"
          >
            <div className="flex h-full flex-col">
              <PaneHeader title={sidebarTitle} />
              <div className="min-h-0 flex-1 overflow-y-auto">{sidebar}</div>
            </div>
          </ResizablePanel>
          <ResizableHandle />
        </>
      )}
      <ResizablePanel order={2} minSize={30} defaultSize={sidebar ? 48 : 62}>
        {center}
      </ResizablePanel>
      {inspector && (
        <>
          <ResizableHandle />
          <ResizablePanel
            ref={right}
            order={3}
            defaultSize={34}
            minSize={20}
            maxSize={60}
            collapsible
            collapsedSize={0}
            onCollapse={() => setOpen((o) => ({ ...o, right: false }))}
            onExpand={() => setOpen((o) => ({ ...o, right: true }))}
            className="bg-sidebar/20"
          >
            {inspector}
          </ResizablePanel>
        </>
      )}
    </ResizablePanelGroup>
  );
}

/** Строка списка в боковой панели (вопросы, история, документы). */
export function ListItem({ active, onClick, icon, title, meta, testid }: { active?: boolean; onClick?: () => void; icon?: ReactNode; title: ReactNode; meta?: ReactNode; testid?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      data-testid={testid}
      className={cn(
        "flex w-full items-start gap-2 border-l-2 border-transparent px-3 py-1.5 text-left hover:bg-secondary/60",
        active && "border-primary bg-secondary text-foreground",
      )}
    >
      {icon && <span className="mt-0.5 shrink-0">{icon}</span>}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[13px]">{title}</span>
        {meta && <span className="block truncate font-mono text-[10.5px] text-muted-foreground">{meta}</span>}
      </span>
    </button>
  );
}

export function SectionLabel({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex items-center justify-between px-3 pb-1 pt-3 text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground">
      <span>{children}</span>
      {action}
    </div>
  );
}
