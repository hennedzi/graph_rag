import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { ArrowUp, Sparkles, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

// Центр рабочей области: лента со своей прокруткой; поле ввода «плавает» над лентой, без жёсткой границы.
export function ChatPane({ children, composer, scrollRef }: { children: ReactNode; composer: ReactNode; scrollRef: RefObject<HTMLDivElement> }) {
  return (
    <>
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-8 px-4 pb-8 pt-6 md:px-6">{children}</div>
      </div>
      <div className="relative shrink-0 px-4 pb-3 md:px-6">
        <div className="pointer-events-none absolute inset-x-0 -top-8 h-8 bg-gradient-to-t from-background to-transparent" />
        <div className="mx-auto max-w-3xl">{composer}</div>
      </div>
    </>
  );
}

/** Подсказки как Suggestions из AI Elements: «пилюли» в одну строку с горизонтальной прокруткой. */
export function Suggestions({ items, onPick, disabled }: { items: string[]; onPick: (s: string) => void; disabled?: boolean }) {
  return (
    <div className="-mx-1 flex gap-2 overflow-x-auto px-1 pb-1 [mask-image:linear-gradient(to_right,black_88%,transparent)] [scrollbar-width:none]">
      {items.map((ex) => (
        <button
          key={ex}
          type="button"
          disabled={disabled}
          onClick={() => onPick(ex)}
          className="shrink-0 rounded-full border bg-card px-3.5 py-1.5 text-xs text-muted-foreground shadow-xs transition-colors hover:border-primary/40 hover:bg-accent hover:text-foreground disabled:opacity-50"
          data-testid="button-example"
        >
          {ex}
        </button>
      ))}
    </div>
  );
}

/**
 * Поле ввода как PromptInput из AI Elements: карточка, сверху авто-растущий textarea,
 * снизу панель — слева инструменты страницы (tools), справа подсказка клавиш и круглая кнопка отправки.
 */
export function Composer({
  busy,
  onSend,
  onStop,
  placeholder,
  examples,
  tools,
  testid = "input-query",
}: {
  busy: boolean;
  onSend: (q: string) => void;
  onStop?: () => void;
  placeholder: string;
  examples?: string[];
  tools?: ReactNode;
  testid?: string;
}) {
  const [q, setQ] = useState("");
  const ta = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [q]);
  const send = (text: string) => {
    if (!text.trim() || busy) return;
    onSend(text.trim());
    setQ("");
  };
  return (
    <div className="space-y-2.5">
      {examples && <Suggestions items={examples} onPick={send} disabled={busy} />}
      <form
        className="rounded-2xl border bg-card shadow-sm transition-[border-color,box-shadow] focus-within:border-primary/50 focus-within:shadow-md focus-within:ring-4 focus-within:ring-primary/10"
        onSubmit={(e) => {
          e.preventDefault();
          send(q);
        }}
        onClick={() => ta.current?.focus()}
      >
        <Textarea
          ref={ta}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send(q);
            }
          }}
          rows={1}
          placeholder={placeholder}
          className="min-h-[52px] resize-none border-0 bg-transparent px-4 pb-1 pt-3.5 text-[14.5px] leading-relaxed shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
          data-testid={testid}
        />
        <div className="flex items-center gap-1.5 px-2.5 pb-2.5" onClick={(e) => e.stopPropagation()}>
          <div className="flex min-w-0 flex-1 items-center gap-1.5">{tools}</div>
          <span className="hidden whitespace-nowrap font-mono text-[10.5px] text-muted-foreground/80 sm:inline">↵ отправить · ⇧↵ строка</span>
          {busy && onStop ? (
            <Button type="button" size="icon" variant="outline" className="h-8 w-8 shrink-0 rounded-full" onClick={onStop} data-testid="button-stop" aria-label="Остановить">
              <Square className="h-3 w-3 fill-current" />
            </Button>
          ) : (
            <Button type="submit" size="icon" className="h-8 w-8 shrink-0 rounded-full" disabled={busy || !q.trim()} data-testid="button-run" aria-label="Отправить">
              <ArrowUp className="h-4 w-4" />
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}

/** Кнопка-«чип» для панели поля ввода. */
export function ComposerChip({ children, onClick, disabled, title, testid }: { children: ReactNode; onClick?: () => void; disabled?: boolean; title?: string; testid?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      data-testid={testid}
      className="flex h-7 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-xs text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
    >
      {children}
    </button>
  );
}

// Вопрос — пузырь справа (Message from="user"), ответ агента — без пузыря, на всю ширину колонки.
export function UserMsg({ text, active, onClick }: { text: string; active?: boolean; onClick?: () => void }) {
  return (
    <div className="flex justify-end">
      <button
        type="button"
        onClick={onClick}
        className={cn(
          "max-w-[85%] rounded-2xl rounded-br-md bg-secondary px-4 py-2.5 text-left text-sm leading-relaxed transition-colors hover:bg-secondary/80",
          active && "ring-2 ring-primary/30",
        )}
        data-testid="msg-user"
        title="Показать этот запрос в инспекторе"
      >
        <span className="whitespace-pre-wrap">{text}</span>
      </button>
    </div>
  );
}

export function BotMsg({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border bg-card shadow-xs">
        <Sparkles className="h-3.5 w-3.5 text-primary" />
      </span>
      <div className="min-w-0 flex-1 space-y-3 pt-0.5">{children}</div>
    </div>
  );
}

// Прокрутка вниз при новых сообщениях, но только если пользователь и так был внизу — не отнимаем чтение.
export function useStickToBottom(dep: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  const near = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const on = () => {
      near.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 160;
    };
    el.addEventListener("scroll", on, { passive: true });
    return () => el.removeEventListener("scroll", on);
  }, []);
  useEffect(() => {
    if (near.current) ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" });
  }, [dep]);
  return ref;
}
