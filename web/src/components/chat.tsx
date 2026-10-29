import { useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowUp, Bot, CircleStop, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

// Раскладка чата: слева лента сообщений + прибитое к низу поле ввода, справа липкая панель.
export function ChatLayout({ children, composer, aside }: { children: ReactNode; composer: ReactNode; aside: ReactNode }) {
  return (
    <div className="grid gap-6 lg:grid-cols-12">
      <section className="flex min-w-0 lg:min-h-[calc(100dvh-6.5rem)] flex-col lg:col-span-7">
        <div className="flex-1 space-y-6 pb-6">{children}</div>
        <div className="sticky bottom-0 z-30 -mx-1 bg-gradient-to-t from-background via-background to-background/0 px-1 pb-4 pt-6">{composer}</div>
      </section>
      <aside className="min-w-0 space-y-4 lg:sticky lg:top-20 lg:col-span-5 lg:max-h-[calc(100dvh-6rem)] lg:self-start lg:overflow-y-auto lg:pr-1">{aside}</aside>
    </div>
  );
}

export function Composer({
  busy,
  onSend,
  onStop,
  placeholder,
  examples,
  toolbar,
  testid = "input-query",
}: {
  busy: boolean;
  onSend: (q: string) => void;
  onStop?: () => void;
  placeholder: string;
  examples?: string[];
  toolbar?: ReactNode;
  testid?: string;
}) {
  const [q, setQ] = useState("");
  const send = (text: string) => {
    if (!text.trim() || busy) return;
    onSend(text.trim());
    setQ("");
  };
  return (
    <div className="space-y-2">
      {examples && (
        <div className="flex flex-wrap gap-1.5">
          {examples.map((ex) => (
            <button
              key={ex}
              type="button"
              disabled={busy}
              onClick={() => send(ex)}
              className="hover-elevate rounded-full border bg-card px-3 py-1 text-xs text-muted-foreground disabled:opacity-50"
              data-testid="button-example"
            >
              {ex}
            </button>
          ))}
        </div>
      )}
      {toolbar}
      <form
        className="flex items-end gap-2 rounded-2xl border bg-card p-2 shadow-lg focus-within:border-primary/50"
        onSubmit={(e) => {
          e.preventDefault();
          send(q);
        }}
      >
        <Textarea
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(q);
            }
          }}
          rows={1}
          placeholder={placeholder}
          className="max-h-40 min-h-[44px] resize-none border-0 bg-transparent text-base shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
          data-testid={testid}
        />
        {busy && onStop ? (
          <Button type="button" size="icon" variant="outline" className="h-10 w-10 shrink-0 rounded-xl" onClick={onStop} data-testid="button-stop" aria-label="Остановить">
            <CircleStop className="h-4 w-4" />
          </Button>
        ) : (
          <Button type="submit" size="icon" className="h-10 w-10 shrink-0 rounded-xl" disabled={busy || !q.trim()} data-testid="button-run" aria-label="Отправить">
            <ArrowUp className="h-4 w-4" />
          </Button>
        )}
      </form>
      <p className="px-1 text-[11px] text-muted-foreground">Enter — отправить, Shift+Enter — новая строка</p>
    </div>
  );
}

export function UserMsg({ text, active, onClick }: { text: string; active?: boolean; onClick?: () => void }) {
  return (
    <div className="flex justify-end gap-3">
      <button
        type="button"
        onClick={onClick}
        className={cn(
          "max-w-[85%] rounded-2xl rounded-tr-sm bg-primary/15 px-4 py-2.5 text-left text-sm leading-relaxed ring-1 ring-primary/20",
          active && "ring-2 ring-primary/60",
        )}
        data-testid="msg-user"
        title="Показать метрики этого запроса справа"
      >
        {text}
      </button>
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary text-muted-foreground">
        <User className="h-4 w-4" />
      </span>
    </div>
  );
}

export function BotMsg({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary ring-1 ring-primary/30">
        <Bot className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1 space-y-3">{children}</div>
    </div>
  );
}

// Прокрутка вниз при новых сообщениях, но только если пользователь и так был внизу — не отнимаем чтение.
export function useStickToBottom(dep: unknown) {
  const near = useRef(true);
  useEffect(() => {
    const on = () => {
      near.current = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
    };
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);
  useEffect(() => {
    if (near.current) window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" });
  }, [dep]);
}
