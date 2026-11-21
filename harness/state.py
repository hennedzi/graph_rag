"""
Состояние сессии harness: хэндлы, бюджет, события, сессии диалога.
"""
import json
import logging
import re
import time
import copy
import uuid
from enum import Enum
import core
from config import CONFIG

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    TOOL_EXECUTING = "tool_executing"
    VERIFYING = "verifying"
    ANSWERING = "answering"
    DONE = "done"
    ERROR = "error"
from core import log, normalize, timeit

PREVIEW = CONFIG.harness.preview_chars
LOG_CHARS = CONFIG.harness.log_chars
MAX_STEPS = CONFIG.harness.max_steps
BUDGET = CONFIG.harness.budget_tokens
FAST_PATH_THRESHOLD = CONFIG.harness.fast_path_threshold
FIX_ROUNDS = CONFIG.harness.fix_rounds
TOOL_CHOICE = {"choice": CONFIG.harness.tool_choice}
CHARS_PER_TOKEN = CONFIG.harness.chars_per_token

hlog = logging.getLogger("graph_rag.harness")

FILTER_WORDS = {"and", "or", "not", "in", "like", "true", "false", "text_match", "array_contains",
                "array_contains_any", "array_contains_all", "array_length", "id", "text", "doc", "src", "dst",
                "passage_ids"}
_STRING = r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''

SESSIONS: dict[str, dict] = {}
SESSION_TTL, MAX_SESSIONS = 3600, 100


class SessionError(Exception):
    """Сессию нельзя продолжить: не найдена/истекла или по ней уже идёт вопрос."""


def new_state(query: str = "") -> dict:
    return {"query": query, "qvec": None, "pid": {}, "handle": {}, "text": {}, "read": set(), "chars": 0, "ctx": 0, "pending": 0, "spent": 0, "budget": BUDGET,
            "trace": [], "thoughts": [], "timings": {}, "doc": {}, "edges": {}, "emit": None,
            "run_id": "", "seq": 0, "events": [], "cancel": False, "scope": None, "verify_error": False}


def handle(st: dict, pid: str) -> str:
    if pid not in st["handle"]:
        st["handle"][pid] = h = f"P{len(st['handle']) + 1}"
        st["pid"][h] = pid
    return st["handle"][pid]


def preview(text: str) -> str:
    return text if len(text) <= PREVIEW else text[:PREVIEW].rsplit(" ", 1)[0] + "…"


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


def remember(st, r: dict):
    """Ребро в исследованный подграф — из них UI рисует граф."""
    st["edges"].setdefault(r["id"], {"src": r["src"], "dst": r["dst"], "text": r["text"],
                                     "handles": [handle(st, p) for p in r["passage_ids"][:3]]})


def rel_line(st, r: dict) -> str:
    """Строка связи для агента; заодно запоминаем ребро."""
    line = f"{r['text']} → {links(st, r)}"
    remember(st, r)
    return line


def scope_where(st) -> str:
    # shortcut: область поиска фильтрует пассажи (у них есть doc); сущности и рёбра общие на базу —
    # поля doc у них нет, фильтрации графа по документу пока нет, FUTURE §7/D.
    return f"doc == {json.dumps(st['scope'])}" if st.get("scope") else ""


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


def graph_part(st, ents: list[dict], rels: list[dict]) -> str:
    """
    retrieve и так берёт кандидатов через граф (passage_ids найденных сущностей и связей), но раньше это
    выбрасывалось: агент видел «обычный поиск» и не шёл по связям, а UI — пустой граф. Теперь видно обоим.
    """
    lines = [rel_line(st, r) for r in rels[:3]]
    names = " · ".join(e["text"] for e in ents[:3])
    return (f"граф: сущности {names} — для их связей neighbors/find_paths\n" if names else "") + "".join(f"{x}\n" for x in lines)


async def qvec(st) -> list[float]:
    """Вектор исходного вопроса: им ранжируются связи графа (идея relation retrieval из Graph RAG Milvus)."""
    if st["qvec"] is None:
        st["qvec"] = (await core.embed([st["query"]]))[0]
    return st["qvec"]


async def _entities(query: str, k: int) -> list[dict]:
    return core.hybrid(query, (await core.embed([query]))[0], "entity", k)


def search_rows(vec: list[float], expr: str, limit: int, fields: list[str]) -> list[dict]:
    res = core.milvus.search(core.COLLECTION, [vec], anns_field="dense", filter=expr, limit=limit,
                             output_fields=fields, search_params={"metric_type": "IP", "params": {"ef": 64}})
    return [h["entity"] | {"id": h["id"]} for h in (res[0] if res else [])]


EVENT_SCHEMA_VERSION = 1

async def emit(st, type: str, **ev):
    """Событие получает seq и попадает в журнал запуска (st["events"] → run JSON); живой UI получает то же самое."""
    st["seq"] += 1
    if "duration_ms" not in ev:
        ev["duration_ms"] = ev.get("ms", 0)
    ev = {"v": EVENT_SCHEMA_VERSION, "seq": st["seq"], "type": type, **ev}
    st["events"].append(ev)
    if st["emit"]:
        await st["emit"](ev)


def say(sid: str, fmt: str, *args):
    if LOG_CHARS:
        hlog.info(f"[%s] {fmt}", sid, *args)


def short(text, n: int = LOG_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


def open_session(session_id: str | None, query: str, scope: str | None = None, branch_from: int | None = None) -> tuple[str, dict]:
    now = time.monotonic()
    for k in [k for k, v in SESSIONS.items() if now - v["used"] > SESSION_TTL and not v["busy"]]:
        del SESSIONS[k]
    if branch_from is not None and session_id:
        src = SESSIONS.get(session_id)
        if not src:
            raise SessionError("сессия не найдена или истекла — начните новый диалог")
        if src["busy"]:
            raise SessionError("по этой сессии уже идёт вопрос")
        if len(SESSIONS) >= MAX_SESSIONS:
            if idle := sorted((v["used"], k) for k, v in SESSIONS.items() if not v["busy"]):
                del SESSIONS[idle[0][1]]
        session_id = uuid.uuid4().hex[:12]
        ses = copy.deepcopy(src)
        SESSIONS[session_id] = ses
        ses.update(busy=True, used=time.monotonic())
        ses["st"]["branch_from"] = branch_from
        st = ses["st"]
        st.update(query=query, qvec=None, trace=[], thoughts=[], timings={}, spent=0,
                  run_id=uuid.uuid4().hex[:12], seq=0, events=[], cancel=False, verify_error=False, scope=scope)
        hint = f"\n(уже прочитаны: {' '.join(sorted(st['read'], key=lambda h: int(h[1:])))})" if st["read"] else ""
        ses["msgs"].append({"role": "user", "content": query + hint})
        ses["turns"] += 1
        return session_id, ses

    if session_id:
        if not (ses := SESSIONS.get(session_id)):
            raise SessionError("сессия не найдена или истекла — начните новый диалог")
        if ses["busy"]:
            raise SessionError("по этой сессии уже идёт вопрос")
        from .prompts import FOLLOWUPS
        FOLLOWUPS.inc()
    else:
        if len(SESSIONS) >= MAX_SESSIONS:
            if idle := sorted((v["used"], k) for k, v in SESSIONS.items() if not v["busy"]):
                del SESSIONS[idle[0][1]]
        session_id = uuid.uuid4().hex[:12]
        from .prompts import SYSTEM
        SESSIONS[session_id] = ses = {"st": new_state(), "msgs": [{"role": "system", "content": SYSTEM}], "turns": 0}
    ses.update(busy=True, used=now)
    st = ses["st"]
    st.update(query=query, qvec=None, trace=[], thoughts=[], timings={}, spent=0,
              run_id=uuid.uuid4().hex[:12], seq=0, events=[], cancel=False, verify_error=False, scope=scope)
    hint = f"\n(уже прочитаны: {' '.join(sorted(st['read'], key=lambda h: int(h[1:])))})" if st["read"] else ""
    ses["msgs"].append({"role": "user", "content": query + hint})
    ses["turns"] += 1
    return session_id, ses


def cancel_run(session_id: str) -> bool:
    """Остановка запуска: текущий вызов провайдера дорабатывает, новых шагов не будет."""
    ses = SESSIONS.get(session_id)
    if not ses or not ses["busy"]:
        return False
    ses["st"]["cancel"] = True
    return True