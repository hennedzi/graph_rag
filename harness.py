"""
Harness: LLM-агент поверх графа с атомарными инструментами.

Почему экономится контекст:
  • инструменты отдают короткие строки, а не JSON и не полные тексты;
  • пассаж в контексте — это хэндл [P3] + превью; полный текст только через read_passage;
  • повторное чтение не дублирует текст («уже прочитан выше»);
  • общий бюджет символов: при превышении агенту остаётся только answer.
Диалог: session_id продолжает сессию — история, хэндлы P# и прочитанное сохраняются,
бюджет общий на весь диалог, поэтому уточняющий вопрос часто отвечается вообще без новых поисков.
Почему качество выше, чем у /ask:
  • агент сам выбирает путь: сущность → связи → нужные пассажи (multi-hop);
  • ответ проверяется верификатором по прочитанным источникам, и при неподтверждённых
    утверждениях агент получает обратную связь и один шанс исправиться.

  uv run harness.py "Кто был учителем Леонарда Эйлера?"
  uv run harness.py            # прогон по QUERIES (при остановленном сервере: Milvus Lite)
Нужен LLM-сервер с tool calling: vLLM --enable-auto-tool-choice --tool-call-parser hermes,
или llama.cpp --jinja.
"""

import asyncio
import json
import logging
import re
import sys
import time
import urllib.request
import uuid

import openai
from prometheus_client import Counter, Histogram
from pymilvus.exceptions import MilvusException

import core
from core import cfg, log, normalize, timeit

MAX_STEPS = cfg.getint("harness", "max_steps")
# Бюджет — реальные токены окна LLM (usage.prompt_tokens от llama.cpp), а не символы ответов инструментов:
# в окно входят системный промпт, схемы инструментов и вся история диалога. Держите ≤ n_ctx llama-server (-c).
BUDGET = cfg.getint("harness", "budget_tokens", fallback=32000)
CHARS_PER_TOKEN = 3  # shortcut: оценка только для текста, добавленного после последнего вызова LLM; точное число — в следующем usage
PREVIEW = cfg.getint("harness", "preview_chars")
FIX_ROUNDS = cfg.getint("harness", "fix_rounds")
TOOL_CHOICE = {"choice": cfg.get("harness", "tool_choice", fallback="required")}  # required: без ответов мимо answer
LOG_CHARS = cfg.getint("harness", "log_chars", fallback=400)  # 0 — не писать ход мыслей в лог
hlog = logging.getLogger("graph_rag.harness")

TOOL_CALLS = Counter("harness_tool_calls_total", "Вызовы инструментов", ["tool", "status"])
OUTCOME = Counter("harness_answers_total", "Итог сессии", ["outcome"])
STEPS = Histogram("harness_steps", "Шагов LLM на вопрос", buckets=(1, 2, 3, 4, 5, 6, 8, 10, 15))
CONTEXT = Histogram("harness_context_chars", "Символов отдано инструментами",
                    buckets=(1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000, 32000))
CONTEXT_TOKENS = Histogram("harness_context_tokens", "Токенов в окне LLM в конце вопроса",
                           buckets=(2000, 4000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 128000))
SPENT_TOKENS = Histogram("harness_tokens_spent", "Токенов (prompt+completion) потрачено на вопрос",
                         buckets=(5000, 10000, 20000, 40000, 80000, 160000, 320000))
UNSUPPORTED = Counter("harness_unsupported_claims_total", "Неподтверждённые утверждения в финальном ответе")
FOLLOWUPS = Counter("harness_followups_total", "Вопросы, продолжившие сессию")

SYSTEM = f"""Ты — исследователь базы знаний (граф: сущности, отношения, пассажи). Отвечай СТРОГО по прочитанным пассажам.

Стратегия — маленькие шаги, экономь контекст:
1. find_entities — найди точное имя сущности из вопроса. Даже для общих вопросов (про проект, документ, термин)
   начни с find_entities по ключевым словам: граф строится из любых документов, не только биографий.
2. neighbors — связи сущности, отсортированы по близости к вопросу (сразу видны хэндлы [P#]).
   hops=2 — ТОЛЬКО если в вопросе есть посредник («сын учителя Эйлера», «коллега брата»);
   на прямые вопросы («что открыл Эйлер») хватает hops=1.
   Нашлась подходящая сущность — сначала neighbors, а не чтение: хэндлы [P#] дают только neighbors/search_passages/query_db.
3. search_passages — если связей нет или вопрос не про конкретную сущность.
   query_db — точные условия: документ, подстрока, «все отношения, где …», подсчёт (count).
4. read_passage — до 5 пассажей за вызов; бери те, на которые указывают нужные связи
   (связи отсортированы по близости к вопросу — верхние важнее). Не читай всё подряд.
5. answer — как только прочитанного хватает, сразу отвечай; не ищи повторно то, что уже прочитал.
   В уточняющем вопросе сначала проверь прочитанное выше — хэндлы [P#] из прошлых ответов действуют;
   если там нет ответа на НОВЫЙ вопрос — ищи заново, инструменты работают и в продолжении диалога.
   Финал ВСЕГДА через инструмент answer: список claims, один факт в каждом, sources — хэндлы
   ПРОЧИТАННЫХ пассажей, где этот факт написан. Обычный текст без markdown.
Окно контекста общее на весь диалог (история уточняющих вопросов тоже его занимает); когда остаток мал, инструменты об этом скажут.

Правила ответа:
- только то, что явно написано в прочитанных пассажах, без цепочек выводов;
- не объединяй факты о разных людях; отвечай только про того, о ком спросили, и только на вопрос;
- если ответа нет — answer с no_answer=true."""


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


QUERY_DB = """Прямой запрос к Milvus по булеву фильтру (только чтение).
kind: passage | entity | relation.
filter — выражение Milvus по полям:
  text — текст строки (у entity: имя в нижнем регистре, ё→е; у relation: «субъект предикат объект»);
  doc — имя документа (passage); src, dst — сущности отношения (нижний регистр, ё→е);
  passage_ids — ARRAY пассажей, где встречается сущность/отношение (можно писать хэндлы "P3").
Операторы: == != > < >= <=, and or not, in ["a","b"], like "%подстрока%" (регистр важен),
  text_match(text, "слова через пробел") — любое слово, со стеммингом,
  array_contains(passage_ids, "P3"), array_contains_any(passage_ids, ["P1","P2"]), array_length(passage_ids) >= 2.
query — если задан: гибридный поиск ВНУТРИ фильтра (лучшие по смыслу); иначе — первые k строк.
count=true — вернуть только число строк.
Примеры: kind=relation filter='src like "%бернулли%"';  kind=passage filter='doc == "euler.md"' query="Петербург";
  kind=relation filter='array_contains(passage_ids, "P2")';  kind=entity filter='array_length(passage_ids) >= 3' count=true.
При ошибке синтаксиса придёт сообщение Milvus с позицией — исправь фильтр."""

_str, _int = {"type": "string"}, {"type": "integer", "minimum": 1, "maximum": 10}
_kind = {"type": "string", "enum": ["passage", "entity", "relation"]}
TOOLS = [
    _fn("find_entities", "Найти сущности по имени/описанию. Возвращает нормализованные имена.", {"query": _str, "k": _int}, ["query"]),
    _fn("neighbors", "Связи сущности 'субъект предикат объект → [P#]', ближайшие к вопросу сверху. hops=2 — плюс связи соседей.",
        {"entity": _str, "hops": {"type": "integer", "minimum": 1, "maximum": 2}}, ["entity"]),
    _fn("query_db", QUERY_DB, {"kind": _kind, "filter": _str, "query": _str, "count": {"type": "boolean"},
                               "k": {"type": "integer", "minimum": 1, "maximum": 20}}, ["kind"]),
    _fn("search_passages", "Гибридный поиск пассажей с реранкером. Возвращает [P#], score и превью.", {"query": _str, "k": _int}, ["query"]),
    _fn("read_passage", "Полный текст пассажей по хэндлам, не больше 5 за вызов.", {"ids": {"type": "array", "items": _str}}, ["ids"]),
    _fn("answer", "Финальный ответ: атомарные утверждения, у каждого — хэндлы прочитанных пассажей. "
                  "Если в пассажах ответа нет — no_answer=true и пустой claims.",
        {"claims": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string", "description": "один факт, обычный текст без markdown"},
            "sources": {"type": "array", "items": _str, "minItems": 1, "description": "хэндлы, например [\"P1\"]"}},
            "required": ["text", "sources"]}},
         "no_answer": {"type": "boolean"}}, ["claims"]),
]


# ─── Состояние сессии: хэндлы P# ↔ id в Milvus ─────────────────
def new_state(query: str = "") -> dict:
    return {"query": query, "qvec": None, "pid": {}, "handle": {}, "text": {}, "read": set(), "chars": 0, "ctx": 0, "pending": 0, "spent": 0, "budget": BUDGET,
            "trace": [], "thoughts": [], "timings": {}, "doc": {}, "edges": {}, "emit": None}


def handle(st: dict, pid: str) -> str:
    if pid not in st["handle"]:
        st["handle"][pid] = h = f"P{len(st['handle']) + 1}"
        st["pid"][h] = pid
    return st["handle"][pid]


def preview(text: str) -> str:
    return text if len(text) <= PREVIEW else text[:PREVIEW].rsplit(" ", 1)[0] + "…"


async def _entities(query: str, k: int) -> list[dict]:
    return core.hybrid(query, (await core.embed([query]))[0], "entity", k)


async def qvec(st) -> list[float]:
    """Вектор исходного вопроса: им ранжируются связи графа (идея relation retrieval из Graph RAG Milvus)."""
    if st["qvec"] is None:
        st["qvec"] = (await core.embed([st["query"]]))[0]
    return st["qvec"]


FILTER_WORDS = {"and", "or", "not", "in", "like", "true", "false", "text_match", "array_contains",
                "array_contains_any", "array_contains_all", "array_length", "id", "text", "doc", "src", "dst",
                "passage_ids"}
_STRING = r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''


def check_filter(st, expr: str) -> str:
    """
    Фильтр пишет LLM, поэтому белый список: только поля схемы и read-only операторы
    (векторы dense/sparse и type недоступны: type задаёт kind). Хэндлы "P3" → реальные id.
    Синтаксис проверяет сам Milvus — его ошибка с позицией уходит агенту на исправление.
    """
    expr = (expr or "").strip()
    if len(expr) > 500:
        raise ValueError("фильтр длиннее 500 символов")
    bare = re.sub(_STRING, '""', expr)
    if bad := sorted({w for w in re.findall(r"[A-Za-z_]\w*", bare) if w.lower() not in FILTER_WORDS}):
        raise ValueError(f"недопустимые слова вне кавычек: {', '.join(bad)}. Поля: id text doc src dst passage_ids")
    return re.sub(r'"(P\d+)"', lambda m: json.dumps(st["pid"].get(m[1], m[1])), expr)


def links(st, r: dict) -> str:
    return " ".join(f"[{handle(st, p)}]" for p in r["passage_ids"][:3])


def rel_line(st, r: dict) -> str:
    """Строка связи для агента; заодно запоминаем ребро — из них UI рисует исследованный подграф."""
    line = f"{r['text']} → {links(st, r)}"
    st["edges"].setdefault(r["id"], {"src": r["src"], "dst": r["dst"], "text": r["text"],
                                     "handles": [handle(st, p) for p in r["passage_ids"][:3]]})
    return line


def search_rows(vec: list[float], expr: str, limit: int, fields: list[str]) -> list[dict]:
    res = core.milvus.search(core.COLLECTION, [vec], anns_field="dense", filter=expr, limit=limit,
                             output_fields=fields, search_params={"metric_type": "IP", "params": {"ef": 64}})
    return [h["entity"] | {"id": h["id"]} for h in (res[0] if res else [])]


# ─── Атомарные инструменты: (state, **args) → короткая строка ──
async def find_entities(st, query: str, k: int = 5) -> str:
    hits = await _entities(query, min(k, 10))
    if not hits:
        return "ничего не найдено"
    return "\n".join(f"{h['text']} (пассажей: {len(h['passage_ids'])})" for h in hits) + \
        "\n→ neighbors(entity) покажет связи подходящей сущности с хэндлами [P#]"


async def neighbors(st, entity: str, hops: int = 1) -> str:
    async def rels(ents: list[str], limit: int):
        lit = json.dumps(ents, ensure_ascii=False)
        return search_rows(await qvec(st), f'type == "relation" and (src in {lit} or dst in {lit})', limit,
                           ["text", "src", "dst", "passage_ids"])
    e = normalize(entity)
    rows = await rels([e], 20)
    if not rows:  # агент передал «Эйлер» вместо «леонард эйлер» — берём ближайшую сущность
        top = await _entities(entity, 1)
        e = top[0]["text"] if top else e
        rows = await rels([e], 20) if top else []
    if not rows:
        return f"у «{e}» нет связей"
    out = f"сущность: {e}\n" + "\n".join(rel_line(st, r) for r in rows)
    if hops >= 2:  # степень 2 из Graph RAG Milvus, но через фильтр src/dst in [...] вместо матриц смежности
        seen = {r["id"] for r in rows}
        ends = sorted({x for r in rows for x in (r["src"], r["dst"])} - {e})
        far = [r for r in await rels(ends, 15 + len(seen)) if r["id"] not in seen][:15]
        out += "\nчерез соседей:\n" + ("\n".join(rel_line(st, r) for r in far) or "нет")
    return out


async def query_db(st, kind: str, filter: str = "", query: str = "", count: bool = False, k: int = 10) -> str:
    if kind not in ("passage", "entity", "relation"):
        raise ValueError("kind: passage | entity | relation")
    where = check_filter(st, filter)
    expr = f'type == "{kind}"' + (f" and ({where})" if where else "")
    if count:
        return f"строк: {core.milvus.query(core.COLLECTION, filter=expr, output_fields=['count(*)'])[0]['count(*)']}"
    k = max(1, min(int(k), 20))
    rows = core.hybrid(query, (await core.embed([query]))[0], kind, k, where) if query else \
        list(core.milvus.query(core.COLLECTION, filter=expr, output_fields=core.FIELDS, limit=k))
    if not rows:
        return "ничего не найдено"
    if kind == "passage":
        return passage_lines(st, rows)
    if kind == "entity":
        return "\n".join(f"{r['text']} (пассажей: {len(r['passage_ids'])})" for r in rows)
    return "\n".join(rel_line(st, r) for r in rows)


def passage_lines(st, rows: list[dict]) -> str:
    """Уже прочитанные пассажи — одной меткой без превью: агент видит, что нового нет, и не ищет по кругу."""
    lines = []
    for r in rows:
        st["text"][r["id"]], st["doc"][r["id"]] = r["text"], r["doc"]
        h = handle(st, r["id"])
        score = f" {r['score']:.2f}" if "score" in r else ""
        lines.append(f"[{h}] уже прочитан" if h in st["read"] else f"[{h}]{score} {r['doc']}: {preview(r['text'])}")
    if all(line.endswith("уже прочитан") for line in lines):
        lines.append("Новых пассажей нет. Если прочитанного хватает для ответа — вызови answer.")
    return "\n".join(lines)


async def search_passages(st, query: str, k: int = 3) -> str:
    """
    Порог реранкера подобран на биографиях; на технических текстах (README, таблицы, код) честные совпадения
    часто ниже него. Агенту лучше увидеть 3 слабых превью с пометкой, чем «ничего» и поиск по кругу.
    """
    _, _, ps = await core.retrieve(query, st["timings"], top_n=min(k, 10), min_score=float("-inf"))
    strong = [p for p in ps if p["score"] >= core.MIN_RERANK_SCORE]
    if strong:
        return passage_lines(st, strong)
    if not ps:
        return "ничего не найдено"
    return f"выше порога {core.MIN_RERANK_SCORE} нет; слабые совпадения — проверь превью:\n" + passage_lines(st, ps[:3])


async def read_passage(st, ids: list[str]) -> str:
    asked = list(dict.fromkeys(h.strip("[] ").upper() for h in ids))
    hs, rest = asked[:5], asked[5:]  # 5 за вызов: агент выбирает нужные, а не читает всё подряд
    unknown = [h for h in hs if h not in st["pid"]]
    missing = [st["pid"][h] for h in hs if h in st["pid"] and st["pid"][h] not in st["text"]]
    for r in core.get_rows(missing, ["text", "doc"]):
        st["text"][r["id"]], st["doc"][r["id"]] = r["text"], r["doc"]
    out = []
    for h in hs:
        if h in unknown:
            out.append(f"[{h}] неизвестный хэндл: это не P# из выдачи; для имени сущности вызови neighbors")
        elif h in st["read"]:
            out.append(f"[{h}] уже прочитан выше")
        else:
            st["read"].add(h)
            out.append(f"[{h}] {st['text'].get(st['pid'][h], '(не найден)')}")
    if rest:
        out.append(f"не прочитаны (не больше 5 за вызов): {' '.join(rest)} — запроси, только если они нужны для ответа")
    return "\n\n".join(out)


IMPL = {"find_entities": find_entities, "neighbors": neighbors, "query_db": query_db,
        "search_passages": search_passages, "read_passage": read_passage}


def render(st, args: dict) -> tuple[str, list[dict], list[str]]:
    """
    claims → текст «факт [P1] [P3]» + детерминированные ошибки (без LLM): факт без источника,
    неизвестный хэндл. Старый формат {"text": ...} тоже принимаем — модель может проигнорировать схему.
    """
    if args.get("no_answer"):
        return core.NO_CONTEXT, [], []
    if not isinstance(args.get("claims"), list):
        text = str(args.get("text", "")).strip()
        return text, [], [] if text else ["пустой ответ: передай claims"]
    claims, errors = [], []
    for c in args["claims"]:
        c = c if isinstance(c, dict) else {"text": str(c)}
        fact = re.sub(r"[*`]+", "", str(c.get("text", ""))).strip().lstrip("-•# ").strip()
        srcs = list(dict.fromkeys(str(x).strip("[] ").upper() for x in (c.get("sources") or [])))
        if not fact:
            continue
        if not srcs:
            errors.append(f"нет sources: «{fact}»")
        elif bad := [x for x in srcs if x not in st["pid"]]:
            errors.append(f"неизвестные хэндлы {' '.join(bad)}: «{fact}»")
        claims.append({"text": fact, "sources": srcs})
    if not claims and not errors:
        errors.append("пустой claims: дай утверждения или no_answer=true")
    return "\n".join(f"{c['text']} " + " ".join(f"[{x}]" for x in c["sources"]) for c in claims), claims, errors


async def chat_tools(msgs: list[dict], usage: dict | None = None):
    """tool_choice=required, а если сервер его не поддерживает — один раз падаем на auto и запоминаем."""
    try:
        return await core.chat(msgs, usage=usage, tools=TOOLS, tool_choice=TOOL_CHOICE["choice"])
    except openai.BadRequestError as e:
        if TOOL_CHOICE["choice"] == "auto":
            raise
        log.warning("LLM-сервер не принял tool_choice=%s (%s), дальше auto", TOOL_CHOICE["choice"], e)
        TOOL_CHOICE["choice"] = "auto"
        return await core.chat(msgs, usage=usage, tools=TOOLS, tool_choice="auto")


async def check(st, text: str) -> list[dict]:
    """Верификация по процитированным пассажам (подгружаем, даже если агент их не читал)."""
    cited = [h for h in st["pid"] if f"[{h}]" in text]
    await read_passage(st, [h for h in cited if h not in st["read"]])
    return await core.verify(text, {h: st["text"].get(st["pid"][h], "") for h in cited})


# ─── Ход мыслей в лог ──────────────────────────────────────────
def thought(msg) -> str:
    """llama.cpp/vLLM отдают рассуждение в reasoning_content; иначе оно внутри <think> в content."""
    r = getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning_content")
    if not r and (m := re.search(r"<think>(.*?)(?:</think>|$)", msg.content or "", re.S)):
        r = m[1]
    return (r or "").strip()


def short(text, n: int = LOG_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


async def emit(st, type: str, **ev):
    """Событие для живого UI (/harness/stream); без подписчика — no-op."""
    if st["emit"]:
        await st["emit"]({"type": type, **ev})


def say(sid: str, fmt: str, *args):
    if LOG_CHARS:
        hlog.info(f"[%s] {fmt}", sid, *args)


# ─── Сессии диалога ────────────────────────────────────────────
# shortcut: сессии в памяти процесса (теряются при рестарте, не делятся между воркерами); для нескольких
# воркеров — хранить msgs+state в Redis/SQLite по session_id.
SESSIONS: dict[str, dict] = {}
SESSION_TTL, MAX_SESSIONS = 3600, 100


class SessionError(Exception):
    """Сессию нельзя продолжить: не найдена/истекла или по ней уже идёт вопрос."""


def open_session(session_id: str | None, query: str) -> tuple[str, dict]:
    now = time.monotonic()
    for k in [k for k, v in SESSIONS.items() if now - v["used"] > SESSION_TTL and not v["busy"]]:
        del SESSIONS[k]
    if session_id:
        if not (ses := SESSIONS.get(session_id)):
            raise SessionError("сессия не найдена или истекла — начните новый диалог")
        if ses["busy"]:
            raise SessionError("по этой сессии уже идёт вопрос")
        FOLLOWUPS.inc()
    else:
        if len(SESSIONS) >= MAX_SESSIONS:  # вытесняем самую старую свободную
            if idle := sorted((v["used"], k) for k, v in SESSIONS.items() if not v["busy"]):
                del SESSIONS[idle[0][1]]
        session_id = uuid.uuid4().hex[:12]
        SESSIONS[session_id] = ses = {"st": new_state(), "msgs": [{"role": "system", "content": SYSTEM}], "turns": 0}
    ses.update(busy=True, used=now)
    st = ses["st"]
    # пер-вопросные поля сбрасываем; хэндлы, тексты, прочитанное, рёбра и бюджет — общие на диалог
    st.update(query=query, qvec=None, trace=[], thoughts=[], timings={}, spent=0)
    hint = f"\n(уже прочитаны: {' '.join(sorted(st['read'], key=lambda h: int(h[1:])))})" if st["read"] else ""
    ses["msgs"].append({"role": "user", "content": query + hint})
    ses["turns"] += 1
    return session_id, ses


LIMIT: dict = {}


async def budget() -> int:
    """
    budget_tokens, но не больше реального окна llama-server (/props → n_ctx минус запас под ответ):
    иначе при -c меньше бюджета сервер вернёт ошибку переполнения раньше, чем агент получит предупреждение.
    """
    if "v" not in LIMIT:
        LIMIT["v"] = BUDGET
        url = str(core.llm.base_url).rstrip("/").removesuffix("/v1") + "/props"
        try:
            # stdlib, а не httpx: openai 3.x уже не тянет httpx, новая зависимость ради одного GET не нужна
            props = json.loads(await asyncio.to_thread(lambda: urllib.request.urlopen(url, timeout=3).read()))
            if n := (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx"):
                LIMIT["v"] = min(BUDGET, int(n) - 2048)
                log.info("harness: окно llama-server n_ctx=%s, бюджет %s токенов", n, LIMIT["v"])
        except (OSError, ValueError, AttributeError):  # не llama.cpp или нет /props — верим конфигу
            log.info("harness: n_ctx не узнать по %s, бюджет из конфига %s токенов", url, BUDGET)
    return LIMIT["v"]


def ctx_now(st) -> int:
    """Токены в окне: точные на момент последнего вызова LLM + оценка того, что добавлено после."""
    return st["ctx"] + st["pending"] // CHARS_PER_TOKEN


async def llm_step(st, msgs: list[dict], tools: bool = True):
    u: dict = {}
    msg = await (chat_tools(msgs, u) if tools else core.chat(msgs, usage=u))
    if u:
        st["ctx"], st["pending"] = u["prompt"], 0
        st["spent"] += u["prompt"] + u["completion"]
    else:  # сервер не вернул usage — оцениваем всё окно по символам
        st["ctx"], st["pending"] = sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or []))
                                       for m in msgs) // CHARS_PER_TOKEN, 0
    # ответ модели сам попадёт в окно (рассуждения reasoning_content — нет: в историю их не кладём)
    st["pending"] += len(msg.content or "") + sum(len(c.function.arguments or "") for c in msg.tool_calls or [])
    return msg


def close_calls(msgs: list[dict], calls, done: set):
    """У каждого tool_call должен быть ответ, иначе следующий ход диалога LLM-сервер может отвергнуть."""
    for c in calls:
        if c.id not in done:
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": "пропущено: ответ уже принят"})


# ─── Цикл агента ───────────────────────────────────────────────
async def run(query: str, on_event=None, session_id: str | None = None) -> dict:
    session_id, ses = open_session(session_id, query)
    try:
        return await _run(query, on_event, session_id, ses)
    except BaseException:  # история могла оборваться посреди tool_calls — такую сессию не продолжаем
        SESSIONS.pop(session_id, None)
        raise
    finally:
        ses["st"]["emit"] = None
        ses.update(busy=False, used=time.monotonic())


async def _run(query: str, on_event, session_id: str, ses: dict) -> dict:
    st, msgs, t0, sid = ses["st"], ses["msgs"], time.perf_counter(), session_id[:6]
    chars0, st["pending"] = st["chars"], st["pending"] + len(msgs[-1]["content"])
    st["budget"] = B = await budget()
    st["emit"] = on_event
    say(sid, "вопрос #%d: %s", ses["turns"], query)
    await emit(st, "start", query=query, budget=B, max_steps=MAX_STEPS, session_id=session_id,
               turn=ses["turns"], chars=st["chars"], ctx_tokens=ctx_now(st), edges=list(st["edges"].values()))
    answer, unsupported, fixes, steps, claims = None, [], 0, 0, []

    while answer is None and steps < MAX_STEPS:
        steps += 1
        with timeit("harness_llm", st["timings"]):
            msg = await llm_step(st, msgs)
        calls = msg.tool_calls or []
        if t := thought(msg):
            st["thoughts"].append({"step": steps, "text": t})
            say(sid, "шаг %d, думает: %s", steps, short(t))
            await emit(st, "thought", step=steps, text=t)
        if said := core.strip_think(msg.content):
            say(sid, "шаг %d, пишет: %s", steps, short(said))
        msgs.append({"role": "assistant", "content": msg.content or "",
                     **({"tool_calls": [c.model_dump() for c in calls]} if calls else {})})
        if not calls:  # модель ответила текстом без инструмента — считаем это ответом
            answer = core.strip_think(msg.content)
            unsupported = await check(st, answer)
            break
        done: set = set()
        for c in calls:
            done.add(c.id)
            name, args = c.function.name, core.parse_json(c.function.arguments or "{}", {})
            say(sid, "→ %s %s", name, short(json.dumps(args, ensure_ascii=False)))
            await emit(st, "call", step=steps, id=c.id, tool=name, args=args)
            t1, status, edges0 = time.perf_counter(), "ok", len(st["edges"])
            if name == "answer":
                text, parsed, errors = render(st, args)
                if errors:  # ловится кодом бесплатно, LLM-верификатор не зовём
                    out = "Ответ отклонён:\n" + "\n".join(f"- {e}" for e in errors) + "\nИсправь и снова вызови answer."
                    TOOL_CALLS.labels(name, "bad_answer").inc()
                    say(sid, "← answer отклонён кодом: %s", short("; ".join(errors)))
                    await emit(st, "result", step=steps, id=c.id, tool=name, status="bad_answer", out=out,
                               chars=len(out), ms=0, total_chars=st["chars"], ctx_tokens=ctx_now(st), edges=[])
                    st["trace"].append({"tool": name, "args": args, "chars": len(out)})
                    msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
                    st["pending"] += len(out)
                    continue
                with timeit("harness_verify", st["timings"]):
                    unsupported = await check(st, text)
                if unsupported and fixes < FIX_ROUNDS:
                    fixes, status = fixes + 1, "unsupported"
                    out = ("Ответ отклонён, не подтверждены источниками:\n"
                           + "\n".join(f"- {u.get('claim')}" for u in unsupported)
                           + "\nУбери эти утверждения или прочитай источник, затем снова вызови answer.")
                else:
                    answer, claims = text, parsed
                    out = "принято"
            elif name not in IMPL:
                out, status = f"нет инструмента {name}", "unknown_tool"
            elif ctx_now(st) > B:
                out, status = "Бюджет окна исчерпан. Вызови answer по уже прочитанному.", "budget"
            else:
                try:
                    with timeit(f"tool_{name}", st["timings"]):
                        out = await IMPL[name](st, **args)
                    TOOL_CALLS.labels(name, "ok").inc()
                except TypeError as e:  # модель передала не те аргументы — пусть исправится
                    out, status = f"ошибка аргументов: {e}", "bad_args"
                    TOOL_CALLS.labels(name, "bad_args").inc()
                except (ValueError, MilvusException) as e:  # плохой фильтр: текст ошибки — подсказка агенту
                    out, status = f"ошибка запроса: {getattr(e, 'message', None) or e}"[:600], "bad_query"
                    TOOL_CALLS.labels(name, "bad_query").inc()
                st["chars"] += len(out)
                if (left := B - ctx_now(st) - len(out) // CHARS_PER_TOKEN) < B // 4:  # заранее, а не после
                    out += f"\n(окно: осталось ~{max(left, 0)} из {B} токенов — пора отвечать)"
            say(sid, "← %s (%d симв.): %s", name, len(out), short(out))
            await emit(st, "result", step=steps, id=c.id, tool=name, status=status, out=out[:3000], chars=len(out),
                       ms=round((time.perf_counter() - t1) * 1000, 1), total_chars=st["chars"],
                       ctx_tokens=ctx_now(st) + len(out) // CHARS_PER_TOKEN,
                       edges=list(st["edges"].values())[edges0:])
            st["trace"].append({"tool": name, "args": args, "chars": len(out)})
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
            st["pending"] += len(out)
            if answer is not None:
                close_calls(msgs, calls, done)
                break

    if answer is None:  # шаги кончились — просим ответ без инструментов
        msgs.append({"role": "user", "content": "Шаги закончились. Дай финальный ответ по прочитанным пассажам со ссылками [P#]."})
        answer = core.strip_think((await llm_step(st, msgs, tools=False)).content)
        msgs.append({"role": "assistant", "content": answer})
        unsupported = await check(st, answer)
        OUTCOME.labels("max_steps").inc()
    else:
        OUTCOME.labels("no_context" if answer.startswith(core.NO_CONTEXT[:30])
                       else "unsupported" if unsupported else "answered").inc()

    say(sid, "итог за %d шагов, окно %d/%d токенов, потрачено %d токенов, неподтверждённых %d: %s",
        steps, ctx_now(st), B, st["spent"], len(unsupported), short(answer))
    STEPS.observe(steps)
    CONTEXT.observe(st["chars"] - chars0)
    CONTEXT_TOKENS.observe(ctx_now(st))
    SPENT_TOKENS.observe(st["spent"])
    UNSUPPORTED.inc(len(unsupported))
    st["timings"]["total"] = round((time.perf_counter() - t0) * 1000, 1)
    cited = {h: {"doc_id": st["pid"][h], "doc": st["doc"].get(st["pid"][h], ""),
                 "text": st["text"].get(st["pid"][h], "")} for h in st["pid"] if f"[{h}]" in answer}
    result = {"query": query, "answer": answer, "claims": claims, "citations": cited, "unsupported": unsupported,
              "steps": steps, "fix_rounds": fixes, "context_chars": st["chars"], "turn_chars": st["chars"] - chars0, "trace": st["trace"], "thoughts": st["thoughts"],
              "graph": list(st["edges"].values()), "session_id": session_id, "turn": ses["turns"],
              "context_tokens": ctx_now(st), "tokens_spent": st["spent"], "budget": B,
              "budget_left": max(B - ctx_now(st), 0)}
    core.save_run(mode="harness", **result, timings_ms={k: v for k, v in st["timings"].items() if k != "total"})
    return result | {"timings_ms": st["timings"]}


QUERIES = [
    "Кто был учителем Леонарда Эйлера?",
    "Какой вклад в математику внёс Якоб Бернулли?",
    "Как связаны Иоганн Бернулли и Леонард Эйлер?",
    "Что такое семья Бернулли?",
    "Где работал Даниил Бернулли?",
    "Какой закон сформулировал Даниил Бернулли?",
    "Кто пригласил Эйлера в Петербург?",
    "Что такое вариационное исчисление?",
]


async def _main(queries):
    for q in queries:
        r = await run(q)
        log.info("\n❓ %s\n💬 %s\n   шагов=%s контекст=%s симв. неподтв.=%s путь=%s", q, r["answer"], r["steps"],
                 r["context_chars"], len(r["unsupported"]), " → ".join(t["tool"] for t in r["trace"]))


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:] or QUERIES))
