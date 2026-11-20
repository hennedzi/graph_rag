"""CLI-клиент к Graph RAG API.

Полноэкранный терминальный клиент (curses, stdlib) со стримингом ответа,
статус-баром, лентой диалога и хоткеями. Если stdout не TTY или --no-tui —
построчный REPL (для pipes и логов).

Сервер:  uv run uvicorn api:app --port 8200
Клиент:  uv run cli.py
         uv run cli.py "Кто был учителем Эйлера?"      # один вопрос
         uv run cli.py --no-tui                         # построчный REPL
         uv run cli.py --doc README.md --no-stream      # один документ, без стрима

TUI:  F1 помощь · F2 новая сессия · F3 runs · F4 документ · F5 стрим · F10 выход
"""

import argparse
import curses
import json
import locale
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("RAG_URL", "http://127.0.0.1:8200")

RESET = "\033[0m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

PAIR = {"user": 1, "answer": 2, "dim": 3, "error": 4, "cmd": 5, "source": 3}


def _color(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if COLOR else text


def truncate(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def iter_sse(lines):
    """SSE: строки 'data: {...}' до пустой строки; комментарии ':' пропускаем."""
    buf: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if buf:
                yield json.loads("\n".join(buf))
                buf = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip())
    if buf:
        yield json.loads("\n".join(buf))


def request(url: str, method: str = "GET", payload: dict | None = None, stream: bool = False, timeout: int = 600):
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream" if stream else "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def api_error(e: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(e.read().decode("utf-8", "replace"))
        return str(body.get("detail") or body)
    except Exception:
        return f"HTTP {e.code}"


def cmd_runs(base: str, args: list[str]):
    n = int(args[0]) if args and args[0].isdigit() else 10
    rows = json.loads(request(f"{base}/runs?limit={n}").read())
    if not rows:
        print("запусков нет")
        return
    for r in reversed(rows):  # API отдаёт старые → новые, показываем свежие сверху
        steps = f"{r['steps']} ш." if r.get("steps") is not None else "—"
        print(f"{r['id']}  {truncate(r.get('outcome', '?'), 18):18} {steps:6} {truncate(r.get('answer', ''), 60)}")


def cmd_run(base: str, args: list[str]):
    if not args:
        print("использование: /run <id>")
        return
    r = json.loads(request(f"{base}/runs/{args[0]}").read())
    print(f"Запрос: {r.get('query', '')}")
    print(f"Итог:   {r.get('outcome', '?')} — {r.get('outcome_note', '')}")
    print()
    print(r.get("answer", ""))


def cmd_stats(base: str):
    print(json.dumps(json.loads(request(f"{base}/stats").read()), ensure_ascii=False, indent=2))


def cmd_help():
    print("Команды:")
    print("  /new             новая сессия (сбросить диалог)")
    print("  /doc [имя]       ограничить поиск документом; без имени — снять")
    print("  /mode [ask|harness]  режим ответа (по умолчанию harness)")
    print("  /runs [n]        последние запуски")
    print("  /run <id>        показать запуск по id")
    print("  /stats           счётчики базы")
    print("  /help            эта справка")
    print("  /exit            выход")


# ─── Построчный режим ──────────────────────────────────────────
def ask(base: str, query: str, session: str | None, doc: str | None, mode: str, stream: bool, quiet: bool) -> str | None:
    if mode == "ask":
        try:
            r = json.loads(request(f"{base}/ask", "POST", {"query": query}).read())
        except urllib.error.HTTPError as e:
            print(_color(RED, "ошибка: " + api_error(e)))
            return session
        print(r.get("answer", ""))
        return session

    payload = {"query": query, "session_id": session, "doc": doc}
    state = {"sid": session, "streamed": False, "started": False}

    def handle(ev: dict):
        t = ev.get("type")
        if t == "start":
            state["sid"] = ev.get("session_id") or state["sid"]
        elif t == "thought" and not quiet:
            print(_color(DIM, f"  · {truncate(ev.get('text', ''), 200)}"))
        elif t == "call" and not quiet:
            args = json.dumps(ev.get("args", {}), ensure_ascii=False)
            print(_color(CYAN, f"  → {ev.get('tool')} {truncate(args, 160)}"))
        elif t == "result":
            if quiet:
                return
            status = ev.get("status", "ok")
            if status != "ok":
                print(_color(RED, f"  ← {ev.get('tool')}: {truncate(ev.get('out', ''), 300)}"))
            else:
                print(_color(DIM, f"  ← {ev.get('tool')} ({ev.get('chars', 0)} симв.)"))
        elif t == "fast_path_taken" and not quiet:
            print(_color(DIM, f"  ⚡ быстрый путь (score {ev.get('top_score')})"))
        elif t == "token":
            if not state["started"]:
                print()
                state["started"] = True
            sys.stdout.write(ev.get("text", ""))
            sys.stdout.flush()
            state["streamed"] = True

    try:
        if stream:
            with request(f"{base}/harness/stream", "POST", payload, stream=True) as resp:
                for ev in iter_sse(resp):
                    if ev.get("type") == "done":
                        print_result(ev.get("result", {}), state["streamed"])
                        return ev.get("result", {}).get("session_id")
                    if ev.get("type") == "error":
                        if ev.get("session") == "lost":
                            print(_color(RED, "сессия потеряна, начинаю новую"))
                            return None
                        print(_color(RED, "ошибка: " + str(ev.get("detail", ev))))
                        return session
                    handle(ev)
        else:
            r = json.loads(request(f"{base}/harness", "POST", payload).read())
            print_result(r, False)
            return r.get("session_id")
    except KeyboardInterrupt:
        print()
        try:
            request(f"{base}/harness/stop", "POST", {"session_id": state["sid"]}, timeout=10).read()
        except Exception:
            pass
        print(_color(YELLOW, "остановлено"))
        return state["sid"]
    except urllib.error.HTTPError as e:
        print(_color(RED, "ошибка: " + api_error(e)))
    except urllib.error.URLError as e:
        print(_color(RED, f"сервер недоступен ({e.reason}): {base}"))
    return session


def print_result(result: dict, streamed: bool):
    if not streamed:
        print(result.get("answer", ""))
    else:
        print()

    cited = result.get("citations") or {}
    if cited:
        parts = []
        for h, c in cited.items():
            doc = c.get("doc") or c.get("doc_id") or ""
            parts.append(f"{h} {doc}".strip())
        print(_color(DIM, "Источники: " + ", ".join(parts)))

    outcome = result.get("outcome")
    colors = {"answered": GREEN, "partial": YELLOW, "verification_failed": YELLOW,
              "no_evidence": RED, "budget_exceeded": YELLOW, "cancelled": YELLOW}
    if outcome:
        note = result.get("outcome_note") or ""
        print(_color(colors.get(outcome, DIM), f"{outcome} — {note}".strip(" —")))

    stats = []
    if result.get("steps") is not None:
        stats.append(f"{result['steps']} шагов")
    if result.get("tokens_spent") is not None:
        stats.append(f"{result['tokens_spent']} ток.")
    if stats:
        print(_color(DIM, " · ".join(stats)))


def repl(base: str, session: str | None, doc: str | None, mode: str, stream: bool, quiet: bool):
    print(_color(DIM, f"Graph RAG CLI · {base} · /help — команды, /exit — выход"))
    while True:
        badge = " ".join(x for x in (mode if mode != "harness" else "", doc) if x)
        prompt = _color(CYAN, f"rag{(' ' + badge) if badge else ''} › ")
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if not line.startswith("/"):
            session = ask(base, line, session, doc, mode, stream, quiet)
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0]
        args = parts[1].split() if len(parts) > 1 else []
        if cmd in ("/exit", "/quit"):
            break
        elif cmd == "/help":
            cmd_help()
        elif cmd == "/new":
            session = None
            print(_color(DIM, "новая сессия"))
        elif cmd == "/doc":
            doc = args[0] if args else None
            print(_color(DIM, f"область: {doc or 'все документы'}"))
        elif cmd == "/mode":
            if args and args[0] in ("ask", "harness"):
                mode = args[0]
            print(_color(DIM, f"режим: {mode}"))
        elif cmd == "/runs":
            cmd_runs(base, args)
        elif cmd == "/run":
            cmd_run(base, args)
        elif cmd == "/stats":
            cmd_stats(base)
        else:
            print(_color(RED, f"неизвестная команда {cmd}; /help — список"))
    return session


# ─── Полноэкранный режим (curses) ──────────────────────────────
class TUI:
    def __init__(self, stdscr, base: str, doc: str | None, mode: str, stream: bool):
        self.st = stdscr
        self.base = base
        self.doc = doc
        self.mode = mode
        self.stream = stream
        self.session: str | None = None
        self.rows: list[list] = []  # [text, style]
        self.input = ""
        self.status = ""
        self.running = False
        self.done = False
        self.scroll = 0
        self.help = False
        self.doc_mode = False
        self.last_draw = 0.0

        try:
            curses.curs_set(1)
        except curses.error:
            pass
        self.st.keypad(True)
        if curses.has_colors() and COLOR:
            curses.start_color()
            try:
                curses.use_default_colors()
            except curses.error:
                pass
            for i, c in enumerate((curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW,
                                   curses.COLOR_RED, curses.COLOR_MAGENTA), 1):
                curses.init_pair(i, c, -1)
        self.st.timeout(100)

    # — вывод —
    def add(self, text: str, style: str = "dim"):
        for line in (text or "").splitlines() or [""]:
            self.rows.append([line, style])

    def put(self, row: int, col: int, text: str, attr: int, width: int):
        if row < 0 or col < 0 or col >= width:
            return
        try:
            self.st.addnstr(row, col, text, max(0, width - col - 1), attr)
        except curses.error:
            pass

    def draw(self):
        h, w = self.st.getmaxyx()
        if h < 5 or w < 20:
            return
        self.st.erase()

        left = " Graph RAG "
        right = " ".join(x for x in (
            self.mode,
            f"doc={self.doc}" if self.doc else "",
            f"sess={self.session[:6]}" if self.session else "",
            "↑" if self.scroll else "",
            self.status,
        ) if x)
        self.put(0, 0, left.ljust(w - 1), curses.A_REVERSE | curses.color_pair(0), w)
        self.put(0, max(len(left), w - len(right) - 1), right, curses.A_REVERSE | curses.color_pair(0), w)

        if self.help:
            self.draw_help(h, w)
            self.st.refresh()
            return

        lines: list[tuple[str, str]] = []
        for text, style in self.rows:
            wrapped = textwrap.wrap(text, max(10, w - 1), break_long_words=True,
                                    replace_whitespace=False) or [""]
            lines.extend((ln, style) for ln in wrapped)

        body_h = h - 3
        start = max(0, len(lines) - body_h - self.scroll)
        for i, (ln, style) in enumerate(lines[start:start + body_h]):
            self.put(1 + i, 0, ln, curses.color_pair(PAIR.get(style, 0)), w)

        prompt = "doc › " if self.doc_mode else ("работаю… " if self.running else "rag › ")
        buf = "" if self.running else self.input
        self.put(h - 2, 0, (prompt + buf)[:w - 1], curses.A_BOLD, w)
        self.put(h - 1, 0, " F1 помощь  F2 новая  F3 runs  F4 doc  F5 стрим  F10 выход ", curses.A_REVERSE, w)

        try:
            curses.curs_set(0 if self.running else 1)
            if not self.running:
                self.st.move(h - 2, min(len(prompt) + len(buf), w - 1))
        except curses.error:
            pass
        self.st.refresh()

    def draw_help(self, h: int, w: int):
        lines = [
            "Graph RAG CLI — полноэкранный клиент",
            "",
            "Enter      отправить вопрос",
            "F1         эта справка",
            "F2         новая сессия (сбросить диалог)",
            "F3         последние запуски (/runs)",
            "F4         выбрать документ для поиска",
            "F5         стриминг ответа вкл/выкл",
            "F10        выход",
            "PgUp/PgDn  листать ленту",
            "",
            "Ctrl+C     очистить строку или остановить запрос",
            "Esc        закрыть справку",
        ]
        bh, bw = min(len(lines) + 4, h - 2), min(64, w - 2)
        y, x = max(0, (h - bh) // 2), max(0, (w - bw) // 2)
        win = curses.newwin(bh, bw, y, x)
        win.erase()
        win.border()
        for i, ln in enumerate(lines):
            try:
                win.addnstr(2 + i, 2, ln, bw - 4)
            except curses.error:
                pass
        win.refresh()

    # — ввод —
    def loop(self):
        while not self.done:
            self.draw()
            try:
                ch = self.st.get_wch()
            except curses.error:
                continue
            if isinstance(ch, str):
                self.on_char(ch)
            else:
                self.on_key(ch)

    def on_char(self, ch: str):
        if self.help:
            if ch in ("\x1b", "\n", "\r", "q", "Q"):
                self.help = False
            return
        if ch in ("\n", "\r"):
            self.submit()
        elif ch in ("\x7f", "\b"):
            self.input = self.input[:-1]
        elif ch == "\x03":  # Ctrl+C
            self.input = ""
        elif ch == "\x1b":
            self.doc_mode = False
            self.input = ""
        elif ch.isprintable() and len(self.input) < 500:
            self.input += ch

    def on_key(self, ch: int):
        if ch == curses.KEY_F1:
            self.help = not self.help
        elif ch == curses.KEY_F2:
            self.session = None
            self.status = "новая сессия"
        elif ch == curses.KEY_F3:
            self.show_runs(10)
        elif ch == curses.KEY_F4:
            self.doc_mode = True
            self.input = ""
        elif ch == curses.KEY_F5:
            self.stream = not self.stream
            self.status = f"стрим: {'вкл' if self.stream else 'выкл'}"
        elif ch == curses.KEY_F10:
            self.done = True
        elif ch == curses.KEY_PPAGE:
            self.scroll += 10
        elif ch == curses.KEY_NPAGE:
            self.scroll = max(0, self.scroll - 10)
        elif ch == curses.KEY_UP:
            self.scroll += 1
        elif ch == curses.KEY_DOWN:
            self.scroll = max(0, self.scroll - 1)
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            self.input = self.input[:-1]
        elif ch in (curses.KEY_ENTER, 10, 13):
            self.submit()

    def submit(self):
        if self.doc_mode:
            self.doc = self.input.strip() or None
            self.input = ""
            self.doc_mode = False
            self.status = f"область: {self.doc or 'все документы'}"
            return
        line = self.input.strip()
        self.input = ""
        if not line:
            return
        if line.startswith("/"):
            self.command(line)
        else:
            self.ask(line)

    def command(self, line: str):
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        args = parts[1].split() if len(parts) > 1 else []
        if cmd in ("/exit", "/quit"):
            self.done = True
        elif cmd == "/help":
            self.help = True
        elif cmd == "/new":
            self.session = None
            self.status = "новая сессия"
        elif cmd == "/doc":
            self.doc = args[0] if args else None
            self.status = f"область: {self.doc or 'все документы'}"
        elif cmd == "/mode":
            if args and args[0] in ("ask", "harness"):
                self.mode = args[0]
            self.status = f"режим: {self.mode}"
        elif cmd == "/runs":
            self.show_runs(int(args[0]) if args and args[0].isdigit() else 10)
        elif cmd == "/run":
            self.show_run(args[0] if args else "")
        elif cmd == "/stats":
            self.show_stats()
        else:
            self.add(f"неизвестная команда {cmd}; F1 — справка", "error")

    # — запросы —
    def ask(self, q: str):
        self.running = True
        self.status = ""
        self.add(f"› {q}", "user")
        self.scroll = 0
        self.draw()

        idx = None
        buf: list[str] = []
        streamed = False
        try:
            if self.mode == "ask":
                r = json.loads(request(f"{self.base}/ask", "POST", {"query": q}).read())
                self.add(r.get("answer", ""), "answer")
                self.add_sources(r.get("sources") or [])
            elif self.stream:
                payload = {"query": q, "session_id": self.session, "doc": self.doc}
                with request(f"{self.base}/harness/stream", "POST", payload, stream=True) as resp:
                    for ev in iter_sse(resp):
                        t = ev.get("type")
                        if t == "start":
                            self.session = ev.get("session_id") or self.session
                        elif t == "thought":
                            self.add(f"· {truncate(ev.get('text', ''), 200)}", "dim")
                        elif t == "call":
                            args = json.dumps(ev.get("args", {}), ensure_ascii=False)
                            self.add(f"→ {ev.get('tool')} {truncate(args, 140)}", "cmd")
                        elif t == "result":
                            if ev.get("status", "ok") != "ok":
                                self.add(f"← {ev.get('tool')}: {truncate(ev.get('out', ''), 300)}", "error")
                            else:
                                self.add(f"← {ev.get('tool')} ({ev.get('chars', 0)} симв.)", "dim")
                        elif t == "fast_path_taken":
                            self.add(f"⚡ быстрый путь (score {ev.get('top_score')})", "dim")
                        elif t == "token":
                            if idx is None:
                                self.rows.append(["", "answer"])
                                idx = len(self.rows) - 1
                            buf.append(ev.get("text", ""))
                            self.rows[idx][0] = "".join(buf)
                            streamed = True
                            now = time.monotonic()
                            if now - self.last_draw > 0.05:
                                self.last_draw = now
                                self.draw()
                        elif t == "done":
                            self.finish(ev.get("result", {}), streamed, idx)
                            return
                        elif t == "error":
                            self.add(f"ошибка: {ev.get('detail', ev)}", "error")
                            if ev.get("session") == "lost":
                                self.session = None
                            return
                        self.draw()
            else:
                r = json.loads(request(f"{self.base}/harness", "POST",
                                       {"query": q, "session_id": self.session, "doc": self.doc}).read())
                self.finish(r, False, None)
        except KeyboardInterrupt:
            try:
                request(f"{self.base}/harness/stop", "POST", {"session_id": self.session}, timeout=10).read()
            except Exception:
                pass
            self.add("остановлено", "dim")
        except urllib.error.HTTPError as e:
            self.add(f"ошибка: {api_error(e)}", "error")
        except urllib.error.URLError as e:
            self.add(f"сервер недоступен ({e.reason})", "error")
        finally:
            self.running = False
            self.draw()

    def finish(self, r: dict, streamed: bool, idx):
        if not streamed:
            self.add(r.get("answer", ""), "answer")
        self.session = r.get("session_id") or self.session

        cited = r.get("citations") or {}
        if cited:
            parts = []
            for h, c in cited.items():
                doc = c.get("doc") or c.get("doc_id") or ""
                parts.append(f"{h} {doc}".strip())
            self.add("Источники: " + ", ".join(parts), "source")

        outcome = r.get("outcome")
        if outcome:
            note = r.get("outcome_note") or ""
            style = "answer" if outcome == "answered" else "dim"
            self.add(f"{outcome} — {note}".strip(" —"), style)

        stats = []
        if r.get("steps") is not None:
            stats.append(f"{r['steps']} шагов")
        if r.get("tokens_spent") is not None:
            stats.append(f"{r['tokens_spent']} ток.")
        if stats:
            self.add(" · ".join(stats), "dim")

    def add_sources(self, sources: list):
        if not sources:
            return
        parts = []
        for p in sources[:8]:
            parts.append(f"{p.get('doc') or p.get('doc_id', '?')}")
        self.add("Источники: " + ", ".join(parts), "source")

    def show_runs(self, n: int):
        self.add(f"runs (последние {n}):", "cmd")
        try:
            rows = json.loads(request(f"{self.base}/runs?limit={n}").read())
        except Exception as e:
            self.add(f"ошибка: {e}", "error")
            return
        for r in reversed(rows):
            steps = f"{r['steps']} ш." if r.get("steps") is not None else "—"
            self.add(f"{r['id']}  {truncate(r.get('outcome', '?'), 18):18} {steps:6} "
                     f"{truncate(r.get('answer', ''), 60)}", "dim")

    def show_run(self, rid: str):
        if not rid:
            self.add("использование: /run <id>", "error")
            return
        try:
            r = json.loads(request(f"{self.base}/runs/{rid}").read())
        except Exception as e:
            self.add(f"ошибка: {e}", "error")
            return
        self.add(f"Запрос: {r.get('query', '')}", "user")
        self.add(f"Итог: {r.get('outcome', '?')} — {r.get('outcome_note', '')}", "dim")
        self.add(r.get("answer", ""), "answer")

    def show_stats(self):
        try:
            s = json.loads(request(f"{self.base}/stats").read())
        except Exception as e:
            self.add(f"ошибка: {e}", "error")
            return
        self.add(json.dumps(s, ensure_ascii=False, indent=2), "dim")


def tui_main(stdscr, base: str, doc: str | None, mode: str, stream: bool):
    TUI(stdscr, base, doc, mode, stream).loop()


def main():
    global COLOR
    p = argparse.ArgumentParser(description="CLI к Graph RAG API")
    p.add_argument("query", nargs="*", help="вопрос одним сообщением (без REPL)")
    p.add_argument("--url", default=DEFAULT_URL, help=f"адрес API (по умолчанию {DEFAULT_URL})")
    p.add_argument("--doc", help="ограничить поиск одним документом")
    p.add_argument("--mode", choices=["harness", "ask"], default="harness", help="harness — агент, ask — один проход")
    p.add_argument("--no-stream", action="store_true", help="не стримить: один POST /harness")
    p.add_argument("--no-tui", action="store_true", help="построчный REPL вместо полноэкранного")
    p.add_argument("-q", "--quiet", action="store_true", help="не печатать шаги агента")
    p.add_argument("--no-color", action="store_true", help="выключить ANSI-цвета")
    args = p.parse_args()

    COLOR = COLOR and not args.no_color
    base = args.url.rstrip("/")
    stream = not args.no_stream

    if not args.no_tui and sys.stdout.isatty():
        try:
            locale.setlocale(locale.LC_ALL, "")
        except locale.Error:
            pass
        try:
            curses.wrapper(tui_main, base, args.doc, args.mode, stream)
            return
        except KeyboardInterrupt:
            return
        except Exception as e:  # TUI не должен ломать клиент: падаем в построчный режим
            print(f"TUI недоступен ({e}); переключаюсь в построчный режим", file=sys.stderr)

    if args.query:
        ask(base, " ".join(args.query), None, args.doc, args.mode, stream, args.quiet)
        return
    try:
        repl(base, None, args.doc, args.mode, stream, args.quiet)
    except urllib.error.URLError as e:
        print(_color(RED, f"сервер недоступен ({e.reason}): {base}"))
        sys.exit(1)


if __name__ == "__main__":
    main()