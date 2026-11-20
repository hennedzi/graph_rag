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
import graph
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
OFF_TOPIC = Counter("harness_offtopic_claims_total", "Верные, но не по вопросу утверждения, выброшенные из ответа")

SYSTEM = f"""Ты — исследователь базы знаний (граф: сущности, отношения, пассажи). Отвечай СТРОГО по прочитанным пассажам.

Стратегия — маленькие шаги, экономь контекст:
1. find_entities — найди точное имя сущности из вопроса. Даже для общих вопросов (про проект, документ, термин)
   начни с find_entities по ключевым словам: граф строится из любых документов, не только биографий.
2. neighbors — связи сущности, отсортированы по близости к вопросу (сразу видны хэндлы [P#]).
   hops=2 — ТОЛЬКО если в вопросе есть посредник («сын учителя Эйлера», «коллега брата»);
   на прямые вопросы («что открыл Эйлер») хватает hops=1.
   Нашлась подходящая сущность — сначала neighbors, а не чтение: хэндлы [P#] дают только neighbors/search_passages/query_db.
   rank_by_ppr — Personalized PageRank от сущностей вопроса по ВСЕМУ графу: самые связанные с ними сущности
   и пассажи. Для широких и многоходовых вопросов («какой вклад», «кто повлиял», «что объединяет») — вместо hops=2.
   find_paths — цепочки связей между ДВУМЯ сущностями (до 3 хопов) с хэндлами [P#]: «как связаны X и Y»,
   «через кого X связан с Y». Цепочка — подсказка, какие пассажи читать; факты всё равно бери из пассажей.
   В вопросе уже подсказаны возможные сущности графа — если они по теме, find_entities не нужен.
3. search_passages — если связей нет или вопрос не про конкретную сущность; он тоже показывает найденные
   сущности и связи графа — по ним можно продолжить neighbors.
   query_db — точные условия: документ, подстрока, «все отношения, где …», подсчёт (count).
4. read_passage — до 5 пассажей за вызов; бери те, на которые указывают нужные связи
   (связи отсортированы по близости к вопросу — верхние важнее). Не читай всё подряд.
5. answer — как только прочитанного хватает, сразу отвечай; не ищи повторно то, что уже прочитал.
   В уточняющем вопросе сначала проверь прочитанное выше — хэндлы [P#] из прошлых ответов действуют;
   если там нет ответа на НОВЫЙ вопрос — ищи заново, инструменты работают и в продолжении диалога.
   Если к реплике приписано «(вопрос целиком: …)» — отвечай на этот вопрос, а не на буквальные слова реплики.
   Финал ВСЕГДА через инструмент answer: список claims, один факт в каждом, sources — хэндлы
   ПРОЧИТАННЫХ пассажей, где этот факт написан. Обычный текст без markdown.
Реплика без вопроса к базе (благодарность, приветствие, «понятно», мнение, просьба без предмета) —
сразу answer(reply="короткий ответ без фактов"), без поиска. На вопрос о фактах reply не принимается.
Окно контекста общее на весь диалог (история уточняющих вопросов тоже его занимает); когда остаток мал, инструменты об этом скажут.

Правила ответа:
- только то, что явно написано в прочитанных пассажах, без цепочек выводов;
- не объединяй факты о разных людях; отвечай только про того, о ком спросили, и только на вопрос;
- каждое утверждение должно отвечать именно на вопрос; верные, но посторонние факты, найденные попутно
  (интерфейс, порты, команды — когда спрашивали не про них), не включай: проверка их отбросит;
- если в прочитанном нет ответа на САМ вопрос — answer с no_answer=true, даже если нашлось что-то по соседней теме."""


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
    _fn("rank_by_ppr", "Personalized PageRank от seed-сущностей: топ связанных сущностей и пассажей по всему графу, "
                       "плюс связи seed с топом. Для многоходовых и обзорных вопросов.",
        {"entities": {"type": "array", "items": _str, "minItems": 1, "maxItems": 5}, "k": _int}, ["entities"]),
    _fn("find_paths", "Цепочки связей source → … → target (до max_hops хопов) с хэндлами [P#], короткие и через "
                      "узкие вершины сверху. Для «как связаны X и Y».",
        {"source": _str, "target": _str, "max_hops": {"type": "integer", "minimum": 1, "maximum": 4}}, ["source", "target"]),
    _fn("query_db", QUERY_DB, {"kind": _kind, "filter": _str, "query": _str, "count": {"type": "boolean"},
                               "k": {"type": "integer", "minimum": 1, "maximum": 20}}, ["kind"]),
    _fn("search_passages", "Гибридный поиск пассажей с реранкером. Возвращает [P#], score и превью.", {"query": _str, "k": _int}, ["query"]),
    _fn("read_passage", "Полный текст пассажей по хэндлам, не больше 5 за вызов.", {"ids": {"type": "array", "items": _str}}, ["ids"]),
    _fn("answer", "Финальный ответ: атомарные утверждения, у каждого — хэндлы прочитанных пассажей. "
                  "Если в пассажах ответа нет — no_answer=true и пустой claims. Реплика не про факты — reply.",
        {"claims": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string", "description": "один факт, обычный текст без markdown"},
            "sources": {"type": "array", "items": _str, "minItems": 1, "description": "хэндлы, например [\"P1\"]"}},
            "required": ["text", "sources"]}},
         "no_answer": {"type": "boolean"},
         "reply": {"type": "string", "description": "ответ на реплику без вопроса к базе (спасибо, привет); без фактов"}},
        []),
]


# ─── Состояние сессии: хэндлы P# ↔ id в Milvus ─────────────────
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


def remember(st, r: dict):
    """Ребро в исследованный подграф — из них UI рисует граф."""
    st["edges"].setdefault(r["id"], {"src": r["src"], "dst": r["dst"], "text": r["text"],
                                     "handles": [handle(st, p) for p in r["passage_ids"][:3]]})


def rel_line(st, r: dict) -> str:
    """Строка связи для агента; заодно запоминаем ребро."""
    line = f"{r['text']} → {links(st, r)}"
    remember(st, r)
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
        "\n→ neighbors(entity) — связи сущности с хэндлами [P#]; rank_by_ppr([..]) — шире по графу; find_paths(a, b) — как связаны две"


async def neighbors(st, entity: str, hops: int = 1) -> str:
    async def rels(ents: list[str], limit: int):
        lit = json.dumps(ents, ensure_ascii=False)
        return search_rows(await qvec(st), f'type == "relation" and (src in {lit} or dst in {lit})', limit,
                           ["text", "src", "dst", "passage_ids"])
    e, g = normalize(entity), graph.get()
    e = g["alias"].get(e, e)
    rows = await rels(g["group"].get(e, [e]), 20)  # все написания сущности: «эйлер» и «леонард эйлер»
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


def scope_where(st) -> str:
    # shortcut: область поиска фильтрует пассажи (у них есть doc); сущности и рёбра общие на базу —
    # поля doc у них нет, фильтрации графа по документу пока нет, FUTURE §7/D.
    return f"doc == {json.dumps(st['scope'])}" if st.get("scope") else ""


async def query_db(st, kind: str, filter: str = "", query: str = "", count: bool = False, k: int = 10) -> str:
    if kind not in ("passage", "entity", "relation"):
        raise ValueError("kind: passage | entity | relation")
    where = check_filter(st, filter)
    if kind == "passage" and (sw := scope_where(st)):
        where = sw + (f" and ({where})" if where else "")
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


def graph_part(st, ents: list[dict], rels: list[dict]) -> str:
    """
    retrieve и так берёт кандидатов через граф (passage_ids найденных сущностей и связей), но раньше это
    выбрасывалось: агент видел «обычный поиск» и не шёл по связям, а UI — пустой граф. Теперь видно обоим.
    """
    lines = [rel_line(st, r) for r in rels[:3]]
    names = " · ".join(e["text"] for e in ents[:3])
    return (f"граф: сущности {names} — для их связей neighbors/find_paths\n" if names else "") + "".join(f"{x}\n" for x in lines)


async def search_passages(st, query: str, k: int = 3) -> str:
    """
    Порог реранкера подобран на биографиях; на технических текстах (README, таблицы, код) честные совпадения
    часто ниже него. Агенту лучше увидеть 3 слабых превью с пометкой, чем «ничего» и поиск по кругу.
    """
    ents, rels, ps = await core.retrieve(query, st["timings"], top_n=min(k, 10), min_score=float("-inf"))
    if st.get("scope"):  # область поиска: пассажи только из выбранного документа
        ps = [p for p in ps if p["doc"] == st["scope"]]
    head = graph_part(st, ents, rels)
    if st.get("scope"):
        head = f"(область: пассажи только из {st['scope']}; граф и сущности — по всей базе)\n" + head
    strong = [p for p in ps if p["score"] >= core.MIN_RERANK_SCORE]
    if strong:
        return head + passage_lines(st, strong)
    if not ps:
        return head + "пассажей не найдено"
    return head + f"выше порога {core.MIN_RERANK_SCORE} нет; слабые совпадения — проверь превью:\n" + passage_lines(st, ps[:3])


async def seed_hint(st) -> str:
    """
    Кандидаты-сущности вопроса кладутся прямо в вопрос: маленькая модель на «как работает X» сама выбирает
    search_passages, а с готовыми именами сразу идёт neighbors/find_paths — и экономит шаг find_entities.
    """
    hits = core.hybrid(st["query"], await qvec(st), "entity", 3)
    return " · ".join(f"{h['text']} ({len(h['passage_ids'])} пасс.)" for h in hits)


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


async def _seeds(names: list[str]) -> tuple[dict, list[str]]:
    g = graph.get()
    found = [await graph.resolve_any(g, n) for n in names[:5]]
    return g, list(dict.fromkeys(e for e in found if e))


async def rank_by_ppr(st, entities: list[str], k: int = 8) -> str:
    g, seeds = await _seeds([entities] if isinstance(entities, str) else entities)
    if not seeds:
        return "ни одной из сущностей нет в графе — сначала find_entities"
    k = max(1, min(int(k), 10))
    r = graph.ppr(g["adj"], dict.fromkeys(seeds, 1.0))
    top = sorted((e for e in r if e not in seeds), key=r.get, reverse=True)[:k]
    if not top:
        return f"seed: {', '.join(seeds)}\nу seed нет связей в графе"
    ps = sorted(graph.passage_scores(r, g["ent_passages"]).items(), key=lambda x: -x[1])[:5]
    texts = {p["id"]: p for p in core.get_rows([p for p, _ in ps], ["text", "doc"])}
    rows = [texts[p] | {"score": s * 100} for p, s in ps if p in texts]
    near = [graph.relation_between(g, s, e) for e in top for s in seeds if frozenset((s, e)) in g["rels"]]
    out = (f"seed: {', '.join(seeds)}\nсущности по PPR (% массы): "
           + " · ".join(f"{e} {r[e] * 100:.1f}" for e in top)
           + "\nпассажи по массе сущностей (%):\n" + passage_lines(st, rows))
    for e in top:  # связи внутри топа тоже в подграф UI: видно, как вершины связаны между собой
        for x in top + seeds:
            if x != e and frozenset((x, e)) in g["rels"]:
                remember(st, graph.relation_between(g, x, e))
    return out + ("\nсвязи seed с топом:\n" + "\n".join(rel_line(st, x) for x in near[:8]) if near else "")


def path_line(st, g: dict, path: list[str]) -> str:
    out, hs = path[0], []
    for a, b in zip(path, path[1:]):
        r = graph.relation_between(g, a, b)
        p = graph.predicate(r)
        fwd = g["alias"].get(r["src"], r["src"]) == a
        out += f" —{p}→ {b}" if fwd else f" ←{p}— {b}"
        remember(st, r)
        hs += [handle(st, x) for x in r["passage_ids"][:2]]
    return out + "  " + " ".join(f"[{h}]" for h in dict.fromkeys(hs))


async def find_paths(st, source: str, target: str, max_hops: int = 3) -> str:
    g, ends = await _seeds([source, target])
    if len(ends) < 2:
        return "одна из сущностей не найдена в графе (или это одна и та же сущность) — уточни через find_entities"
    hops = max(1, min(int(max_hops), 4))
    found = graph.paths(g["adj"], ends[0], ends[1], hops)
    if not found:
        return f"путей «{ends[0]}» → «{ends[1]}» до {hops} хопов нет; попробуй rank_by_ppr или search_passages"
    return f"пути {ends[0]} → {ends[1]}:\n" + "\n".join(path_line(st, g, p) for p in found)


IMPL = {"find_entities": find_entities, "neighbors": neighbors, "rank_by_ppr": rank_by_ppr, "find_paths": find_paths,
        "query_db": query_db,
        "search_passages": search_passages, "read_passage": read_passage}


def render(st, args: dict) -> tuple[str, list[dict], list[str]]:
    """
    claims → текст «факт [P1] [P3]» + детерминированные ошибки (без LLM): факт без источника,
    неизвестный хэндл. Старый формат {"text": ...} тоже принимаем — модель может проигнорировать схему.
    """
    if args.get("no_answer"):
        return core.NO_CONTEXT, [], []
    if (reply := str(args.get("reply") or "").strip()) and not args.get("claims"):
        # reply не верифицируется, поэтому пускаем его только на не-вопрос и без ссылок/длинных фактов
        if is_question(st["query"]):
            return reply, [], ["это вопрос к базе: reply нельзя — найди пассажи и дай claims или no_answer=true"]
        if "[P" in reply or len(reply) > 300:
            return reply, [], ["reply — короткая реплика без фактов и ссылок; факты — только через claims"]
        return reply, [], []
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
    return fmt(claims), claims, errors


def lines_to_claims(st, text: str) -> list[dict]:
    """Ответ текстом → claims: строка без известного хэндла [P#] — не факт из источника, отбрасываем."""
    out = []
    for line in text.splitlines():
        srcs = [h for h in dict.fromkeys(re.findall(r"\[(P\d+)\]", line)) if h in st["pid"]]
        fact = re.sub(r"\s*\[P\d+\]", "", re.sub(r"[*`]+", "", line)).strip().lstrip("-•#0123456789.) ").strip()
        if srcs and fact:
            out.append({"text": fact, "sources": srcs})
    return out


def fmt(claims: list[dict]) -> str:
    return "\n".join(f"{c['text']} " + " ".join(f"[{x}]" for x in c["sources"]) for c in claims)


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
    """Верификация по процитированным пассажам (подгружаем, даже если агент их не читал).
    None от verify = проверка не выполнена: это не «всё подтверждено» и не «в документах нет»."""
    cited = [h for h in st["pid"] if f"[{h}]" in text]
    await read_passage(st, [h for h in cited if h not in st["read"]])
    res = await core.verify(text, {h: st["text"].get(st["pid"][h], "") for h in cited})
    if res is None:
        st["verify_error"] = True
        return []
    return res


RELEVANT_PROMPT = """Вопрос: {q}

Утверждения ответа:
{claims}

Для каждого утверждения реши: относится ли оно к вопросу — отвечает на него, его часть или поясняет ответ.
off_topic=true ТОЛЬКО если утверждение про другую тему: например, на «как работает поиск» — раскладка
интерфейса, порт сервера, поля событий. Сомневаешься — off_topic=false.
Верни СТРОГО JSON-массив: [{{"n": номер, "off_topic": true|false}}]
Только JSON, без пояснений и markdown.

JSON:"""


async def off_topic(st, claims: list[dict]) -> list[dict]:
    """Верификатор проверяет только «написано ли в источнике», поэтому верный, но посторонний факт проходил как ответ."""
    if not claims:
        return []
    listing = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(claims, 1))
    raw = await core.ask_llm(RELEVANT_PROMPT.format(q=st["query"], claims=listing))
    checks = core.parse_json(raw, None)
    if not isinstance(checks, list):  # сломанный JSON не должен выбрасывать ответ
        log.warning("Проверка релевантности вернула невалидный JSON: %s", short(raw))
        return []
    # выбрасываем только явное off_topic=true: ошибка судьи не должна стирать верный ответ
    drop = {int(c["n"]) for c in checks if isinstance(c, dict) and c.get("off_topic") is True and str(c.get("n")).isdigit()}
    if drop:
        say(st.get("sid", ""), "не по вопросу «%s»: %s", st["query"], short(raw))
    return [c for i, c in enumerate(claims, 1) if i in drop]


RESOLVE_PROMPT = """Перепиши реплику пользователя в самостоятельный вопрос к базе знаний.

Прошлые вопросы пользователя (последний внизу):
{questions}

Последний ответ ассистента:
{answer}

Новая реплика: «{q}»

Правила:
- реплика ссылается на прошлое («это», «он», «а где», «что это значит») — подставь, о чём речь;
- реплика — возражение или жалоба на ответ («это неправда», «это не ответ на мой вопрос», «не то») —
  верни дословно тот прошлый вопрос, на который ответ не устроил;
- реплика уточняет предмет («я про X», «имею в виду X») — перепиши ПОСЛЕДНИЙ вопрос с этим предметом
  (например, «что ещё умеет приложение?» + «я про Graph RAG» → «что ещё умеет приложение Graph RAG?»);
- реплика без вопроса (спасибо, оценка, «теперь лучше», согласие) — верни её дословно;
- реплика уже самостоятельный вопрос — верни её без изменений.
Верни только вопрос одной строкой, без пояснений."""


async def resolve(ses: dict, query: str) -> str:
    """
    «И что это значит», «ЭТО НЕ ПРАВДА» раньше уходили в поиск дословно: сущности и ранжирование связей
    шли по словам «запоминаются», «error», и агент цитировал случайные пассажи. Ищем по полному вопросу.
    """
    if not (past := ses.get("questions")):
        return query
    u: dict = {}
    prompt = RESOLVE_PROMPT.format(questions="\n".join(past[-5:]), answer=ses.get("prev", ""), q=query)
    out = core.strip_think((await core.chat([{"role": "user", "content": prompt}], usage=u)).content)
    ses["st"]["spent"] += u.get("prompt", 0) + u.get("completion", 0)
    line = next((x.strip(" «»\"'") for x in out.splitlines() if x.strip()), "")
    return line if 0 < len(line) <= 500 else query


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
    """Событие получает seq и попадает в журнал запуска (st["events"] → run JSON); живой UI получает то же самое."""
    st["seq"] += 1
    if "duration_ms" not in ev:
        ev["duration_ms"] = ev.get("ms", 0)
    ev = {"seq": st["seq"], "type": type, **ev}
    st["events"].append(ev)
    if st["emit"]:
        await st["emit"](ev)


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


def open_session(session_id: str | None, query: str, scope: str | None = None) -> tuple[str, dict]:
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
    st.update(query=query, qvec=None, trace=[], thoughts=[], timings={}, spent=0,
              run_id=uuid.uuid4().hex[:12], seq=0, events=[], cancel=False, verify_error=False, scope=scope)
    hint = f"\n(уже прочитаны: {' '.join(sorted(st['read'], key=lambda h: int(h[1:])))})" if st["read"] else ""
    ses["msgs"].append({"role": "user", "content": query + hint})
    ses["turns"] += 1
    return session_id, ses


def cancel_run(session_id: str) -> bool:
    """Остановка запуска: текущий вызов провайдера дорабатывает (прервать его нельзя), новых шагов не будет."""
    ses = SESSIONS.get(session_id)
    if not ses or not ses["busy"]:
        return False
    ses["st"]["cancel"] = True
    return True


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


# ─── Реплики без поиска ────────────────────────────────────────
# «Спасибо» и «опиши проще» не должны запускать поиск: первое отвечается без LLM, второе — одним вызовом
# по прошлому ответу (с проверкой ссылок верификатором), а не 5–10 шагами агента.
CHAT_WORDS = {
    "thanks": {"спасибо", "спс", "благодарю", "мерси", "thanks", "thank", "thx"},
    "hello": {"привет", "здравствуй", "здравствуйте", "добрый", "день", "вечер", "утро", "hi", "hello", "хай"},
    "bye": {"пока", "до", "свидания", "встречи", "bye"},
    "ok": {"ок", "окей", "ok", "okay", "понятно", "ясно", "понял", "поняла", "хорошо", "отлично", "супер", "класс",
           "круто", "здорово", "ага", "угу", "да", "нет", "good", "great", "cool", "большое", "огромное", "тебе", "вам",
           "все", "ну", "и", "а", "еще", "раз", "очень", "так", "теперь", "лучше", "намного", "гораздо"},
}
CHAT_REPLY = {
    "thanks": "Пожалуйста. Задайте следующий вопрос или попросите пересказать ответ проще или короче.",
    "hello": "Здравствуйте. Спросите что-нибудь по базе знаний — отвечу со ссылками на источники.",
    "bye": "До встречи.",
    "ok": "Хорошо. Можно задать следующий вопрос или попросить ответ «проще» или «короче».",
}
REWRITE = re.compile(r"\b(?:попроще|проще|покороче|короче|кратко|кратче|вкратце|в двух словах|своими словами|понятнее|"
                     r"перефразир\w*|переформулир\w*|упрост\w*|списком|по пунктам|одним предложением|"
                     r"по-английски|на английском)\b")
# «короче» должно реально сокращать (FUTURE §7); «проще» честно не измерить — принимаем пересказ с источниками
SHORTEN = re.compile(r"\b(?:покороче|короче|кратко|кратче|вкратце|в двух словах|одним предложением)\b")
FILLER = {"опиши", "объясни", "расскажи", "скажи", "сделай", "напиши", "перескажи", "изложи", "сформулируй", "ответь",
          "можно", "можешь", "попробуй", "давай", "пожалуйста", "а", "ну", "и", "это", "то", "же", "так", "ли", "еще",
          "ответ", "мне", "чуть", "немного", "его", "как", "нибудь", "по", "сказать", "надо", "нужно", "лучше", "ты"}
QWORDS = {"кто", "что", "где", "когда", "почему", "зачем", "как", "какой", "какая", "какое", "какие", "каков", "сколько",
          "чей", "чем", "откуда", "куда", "ли", "who", "what", "where", "when", "why", "how", "which"}


def words(text: str) -> list[str]:
    return re.findall(r"[a-zа-я][a-zа-я-]*", text.lower().replace("ё", "е"))


def is_question(text: str) -> bool:
    return "?" in text or bool(set(words(text)) & QWORDS)


def route(query: str, has_answer: bool) -> str:
    """chat — реплика без вопроса (ответ шаблоном); rewrite — переформулировать прошлый ответ; agent — искать."""
    w = words(query)
    if len(w) <= 6 and all(x in CHAT_WORDS["thanks"] | CHAT_WORDS["hello"] | CHAT_WORDS["bye"] | CHAT_WORDS["ok"] for x in w):
        return "chat"
    joined = " ".join(w)
    if has_answer and REWRITE.search(joined) and not set(REWRITE.sub(" ", joined).split()) - FILLER:
        return "rewrite"
    return "agent"


def chat_reply(query: str) -> str:
    w = set(words(query))
    return next((CHAT_REPLY[k] for k in ("thanks", "hello", "bye") if w & CHAT_WORDS[k] - {"до", "день", "вечер", "утро"}),
                CHAT_REPLY["ok"] if w else "Задайте вопрос по базе знаний.")


REWRITE_PROMPT = """Пользователь просит переформулировать твой прошлый ответ: «{q}».
Правила: только факты из прошлого ответа, ничего не добавляй и не додумывай; после каждого факта оставь
его ссылки [P#] ровно как в оригинале; обычный текст без markdown.

Прошлый ответ:
{a}

Новый ответ:"""


async def quick(query: str, kind: str, st: dict, ses: dict) -> tuple[str, list[dict], str]:
    """Ответ без цикла агента. Возвращает (ответ, неподтверждённые, итог для метрики)."""
    if kind == "chat":
        return chat_reply(query), [], "chat"
    last, u = ses["last"], {}
    with timeit("harness_rewrite", st["timings"]):
        text = core.strip_think((await core.chat([{"role": "user", "content": REWRITE_PROMPT.format(q=query, a=last)}],
                                                 usage=u)).content)
    st["spent"] += u.get("prompt", 0) + u.get("completion", 0)
    with timeit("harness_verify", st["timings"]):
        bad = await check(st, text) if re.search(r"\[P\d+\]", text) else [{"claim": "в пересказе пропали ссылки [P#]"}]
    if bad or st["verify_error"]:  # пересказ не проверен или не прошёл проверку — честнее отдать прошлый проверенный ответ
        return f"Пересказать без потери точности не вышло, прошлый ответ:\n{last}", [], "rewrite_fallback"
    if SHORTEN.search(" ".join(words(query))) and len(text) >= len(last):
        return f"Пересказ не стал короче, прошлый ответ:\n{last}", [], "rewrite_fallback"
    return text, [], "rewrite"


# ─── Цикл агента ───────────────────────────────────────────────
def outcome_of(answer: str, claims: list[dict], unsupported: list, cancelled: bool,
               steps_exhausted: bool, verify_error: bool) -> tuple[str, str]:
    """Машинная причина завершения + сообщение из реального результата, не из рассказа модели (FUTURE §4)."""
    if cancelled:
        return "cancelled", "остановлено пользователем; накопленные материалы сохранены"
    if verify_error:
        return "verification_failed", "проверяющая модель не ответила — опора на источники не проверена"
    if not claims:
        return ("budget_exceeded", "достигнут лимит шагов — ответа не нашлось") if steps_exhausted \
            else ("no_evidence", "в прочитанных документах нет ответа на вопрос")
    if unsupported:
        return "partial", f"ответ частичный: {len(unsupported)} утверждений не подтверждены источниками"
    return "answered", "утверждения ответа подтверждены источниками"


async def run(query: str, on_event=None, session_id: str | None = None, doc: str | None = None) -> dict:
    session_id, ses = open_session(session_id, query, scope=doc)
    try:
        if (kind := route(query, bool(ses.get("last")))) != "agent":
            return await _quick(query, kind, on_event, session_id, ses)
        return await _run(query, on_event, session_id, ses)
    except BaseException:  # история могла оборваться посреди tool_calls — такую сессию не продолжаем
        SESSIONS.pop(session_id, None)
        raise
    finally:
        ses["st"]["emit"] = None
        ses.update(busy=False, used=time.monotonic())


async def _run(query: str, on_event, session_id: str, ses: dict) -> dict:
    st, msgs, t0, sid = ses["st"], ses["msgs"], time.perf_counter(), session_id[:6]
    st["sid"] = sid
    with timeit("harness_resolve", st["timings"]):
        st["query"] = await resolve(ses, query)
    if normalize(st["query"]) != normalize(query):
        retry = st["query"] in ses.get("questions", [])
        msgs[-1]["content"] += f"\n(вопрос целиком: {st['query']}" + (
            "; прошлый ответ пользователя не устроил — найди ответ именно на этот вопрос, "
            "а если его нет в документах — no_answer=true)" if retry else ")")
    with timeit("harness_seed", st["timings"]):
        seeds = await seed_hint(st)
    if seeds:
        msgs[-1]["content"] += f"\n(возможные сущности вопроса в графе: {seeds} — если подходят, начни с neighbors, find_paths или rank_by_ppr)"
    chars0, st["pending"] = st["chars"], st["pending"] + len(msgs[-1]["content"])
    st["budget"] = B = await budget()
    st["emit"] = on_event
    say(sid, "вопрос #%d: %s", ses["turns"], query)
    await emit(st, "start", query=query, budget=B, max_steps=MAX_STEPS, session_id=session_id,
               turn=ses["turns"], chars=st["chars"], ctx_tokens=ctx_now(st), edges=list(st["edges"].values()),
               run_id=st["run_id"], scope=st["scope"])
    if seeds:
        st["thoughts"].append({"step": 0, "text": f"Сущности вопроса в графе: {seeds}"})
        await emit(st, "thought", step=0, text=f"Сущности вопроса в графе: {seeds}")
    answer, unsupported, fixes, steps, claims = None, [], 0, 0, []

    while answer is None and steps < MAX_STEPS:
        if st["cancel"]:  # остановка: новых вызовов модели и инструментов не будет
            break
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
        batch_start = time.perf_counter()
        for c in calls:
            done.add(c.id)
            name, args = c.function.name, core.parse_json(c.function.arguments or "{}", {})
            say(sid, "→ %s %s", name, short(json.dumps(args, ensure_ascii=False)))
            await emit(st, "call", step=steps, id=c.id, tool=name, args=args, started_at=batch_start)
            t1, status, edges0 = batch_start, "ok", len(st["edges"])
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
                    unsupported, off = await asyncio.gather(check(st, text), off_topic(st, parsed))
                if (unsupported or off) and fixes < FIX_ROUNDS:
                    fixes, status = fixes + 1, "unsupported" if unsupported else "off_topic"
                    out = ("Ответ отклонён:\n"
                           + "".join(f"- не подтверждено источниками: {u.get('claim')}\n" for u in unsupported)
                           + "".join(f"- не отвечает на вопрос «{st['query']}»: {c['text']}\n" for c in off)
                           + "Убери эти утверждения. Если без них ответа на вопрос нет — найди именно его "
                             "или вызови answer(no_answer=true).")
                else:
                    if off:  # последний шанс исправиться прошёл — посторонние факты выбрасываем сами
                        OFF_TOPIC.inc(len(off))
                        parsed = [c for c in parsed if c not in off]
                        text = fmt(parsed) or (f"{core.NO_CONTEXT} На вопрос «{st['query']}» в документах ответа "
                                               "не нашлось — только факты не по теме.")
                        unsupported = unsupported if parsed else []
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
                       edges=list(st["edges"].values())[edges0:], started_at=batch_start)
            st["trace"].append({"tool": name, "args": args, "chars": len(out)})
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
            st["pending"] += len(out)
            if answer is not None:
                close_calls(msgs, calls, done)
                break

    if answer is None and st["cancel"]:
        answer = "Запуск остановлен на шаге {0}. Прочитанные материалы и подграф — в инспекторе.".format(steps)
    elif answer is None:  # шаги кончились — просим ответ без инструментов, но проверяем так же, как answer
        msgs.append({"role": "user", "content": f"Шаги закончились. Ответь на вопрос «{st['query']}» по прочитанным "
                     "пассажам: один факт в строке, в конце строки ссылки [P#], без markdown. "
                     f"Если ответа на этот вопрос в прочитанном нет — ответь: {core.NO_CONTEXT}"})
        raw = core.strip_think((await llm_step(st, msgs, tools=False)).content)
        msgs.append({"role": "assistant", "content": raw})
        claims = lines_to_claims(st, raw)
        with timeit("harness_verify", st["timings"]):
            unsupported, off = await asyncio.gather(check(st, fmt(claims)), off_topic(st, claims))
        OFF_TOPIC.inc(len(off))
        claims = [c for c in claims if c not in off]
        answer = fmt(claims) or core.NO_CONTEXT
        unsupported = unsupported if claims else []
    outcome, note = outcome_of(answer, claims, unsupported, st["cancel"], steps >= MAX_STEPS, st["verify_error"])
    OUTCOME.labels(outcome).inc()

    say(sid, "итог за %d шагов, окно %d/%d токенов, потрачено %d токенов, неподтверждённых %d: %s",
        steps, ctx_now(st), B, st["spent"], len(unsupported), short(answer))
    ses["prev"] = answer
    if st["query"] not in (qs := ses.setdefault("questions", [])):
        qs.append(st["query"])
    if claims and not unsupported:
        ses["last"] = answer  # для «опиши проще»: пересказывать можно только проверенный ответ со ссылками
    STEPS.observe(steps)
    CONTEXT.observe(st["chars"] - chars0)
    CONTEXT_TOKENS.observe(ctx_now(st))
    SPENT_TOKENS.observe(st["spent"])
    UNSUPPORTED.inc(len(unsupported))
    return finish(st, ses, session_id, query, answer, claims, unsupported, steps, fixes, chars0, t0, "agent", outcome, note)


async def _quick(query: str, kind: str, on_event, session_id: str, ses: dict) -> dict:
    st, t0, chars0 = ses["st"], time.perf_counter(), ses["st"]["chars"]
    st["budget"], st["emit"] = await budget(), on_event
    await emit(st, "start", query=query, budget=st["budget"], max_steps=0, session_id=session_id, route=kind,
               turn=ses["turns"], chars=st["chars"], ctx_tokens=ctx_now(st), edges=list(st["edges"].values()),
               run_id=st["run_id"], scope=st["scope"])
    answer, unsupported, kind = await quick(query, kind, st, ses)
    ses["msgs"].append({"role": "assistant", "content": answer})  # агент увидит реплику в следующем вопросе
    st["pending"] += len(ses["msgs"][-2]["content"]) + len(answer)
    outcome = "verification_failed" if kind == "rewrite_fallback" else "answered"
    note = {"chat": "реплика без поиска", "rewrite": "пересказ прошлого ответа без нового поиска",
            "rewrite_fallback": "пересказ не прошёл проверку — показан прошлый проверенный ответ"}.get(kind, "")
    OUTCOME.labels(outcome).inc()
    say(session_id[:6], "без поиска (%s): %s", kind, short(answer))
    return finish(st, ses, session_id, query, answer, [], unsupported, 0, 0, chars0, t0, kind, outcome, note)


def finish(st, ses, session_id, query, answer, claims, unsupported, steps, fixes, chars0, t0, kind, outcome, note) -> dict:
    B = st["budget"]
    st["timings"]["total"] = round((time.perf_counter() - t0) * 1000, 1)
    cited = {h: {"doc_id": st["pid"][h], "doc": st["doc"].get(st["pid"][h], ""),
                 "text": st["text"].get(st["pid"][h], "")} for h in st["pid"] if f"[{h}]" in answer}
    result = {"query": query, "question": st["query"], "answer": answer, "claims": claims, "citations": cited, "unsupported": unsupported,
              "steps": steps, "fix_rounds": fixes, "context_chars": st["chars"], "turn_chars": st["chars"] - chars0, "trace": st["trace"], "thoughts": st["thoughts"],
              "graph": list(st["edges"].values()), "session_id": session_id, "turn": ses["turns"],
              "context_tokens": ctx_now(st), "tokens_spent": st["spent"], "budget": B,
              "budget_left": max(B - ctx_now(st), 0), "route": kind,
              "run_id": st["run_id"], "scope": st["scope"], "outcome": outcome, "outcome_note": note,
              "read": {h: {"doc": st["doc"].get(st["pid"].get(h, ""), ""),
                          "text": st["text"].get(st["pid"].get(h, ""), "")}
                       for h in sorted(st["read"], key=lambda x: int(x[1:]))},
              # живой стрим и сохранённый run показывают одни и те же шаги; edges уже есть в graph — не дублируем
              "events": [{k: v for k, v in e.items() if k != "edges"} for e in st["events"]]}
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
