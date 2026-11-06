"""
Ядро Graph RAG: конфиг, клиенты моделей, Milvus, метрики, поиск, генерация, верификация.

Одна коллекция, три типа строк (type): passage | entity | relation.
  passage  — текст чанка, doc = имя документа
  entity   — нормализованное имя, passage_ids = где встречается
  relation — "s p o", src/dst = сущности, passage_ids = откуда взят факт
id = хэш содержимого → повторная загрузка того же текста идемпотентна.
"""

import configparser
import hashlib
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram
from pymilvus import (AnnSearchRequest, DataType, Function, FunctionType,
                      MilvusClient, RRFRanker)

# ─── Конфиг ────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
if not cfg.read(os.environ.get("RAG_CONFIG", "config.ini"), encoding="utf-8"):
    raise SystemExit("Нет config.ini: cp config.example.ini config.ini и впишите api_key")

LLM_MODEL = cfg["llm"]["model"]
EMB_MODEL, EMB_DIM = cfg["embeddings"]["model"], cfg.getint("embeddings", "dimension")
EMB_BATCH = cfg.getint("embeddings", "batch", fallback=64)
RERANK_MODEL = cfg["reranker"]["model"]
COLLECTION = cfg["milvus"]["collection"]

TOP_ENTITIES      = cfg.getint("search", "top_entities")
TOP_RELATIONS     = cfg.getint("search", "top_relations")
PASSAGE_HITS      = cfg.getint("search", "passage_hits")
RERANK_CANDIDATES = cfg.getint("search", "rerank_candidates")
TOP_PASSAGES      = cfg.getint("search", "top_passages")
MIN_RERANK_SCORE  = cfg.getfloat("search", "min_rerank_score")
RRF_K             = cfg.getint("search", "rrf_k")

RUNS_DIR = Path(cfg.get("logging", "runs_dir", fallback="runs"))
LOG_FILE = Path(cfg.get("logging", "file", fallback="logs/graph_rag.log"))
MAX_LINKS = 4096  # потолок Milvus для ARRAY; сущность в >4096 пассажах хранит первые 4096

NO_CONTEXT = "В предоставленном контексте нет прямого ответа."

# ─── Логи ──────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=cfg.get("logging", "level", fallback="INFO").upper(),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
for noisy in ("pymilvus", "httpx", "httpx2", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger("grpc._server").setLevel(logging.CRITICAL)  # Milvus Lite не реализует AllocTimestamp, трейсбек безвреден
log = logging.getLogger("graph_rag")

# ─── Метрики (общие для всех процессов) ────────────────────────
STAGE = Histogram("rag_stage_seconds", "Длительность этапа", ["stage"],
                  buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120))
LLM_TOKENS = Counter("rag_llm_tokens_total", "Токены LLM", ["kind"])
MODEL_ERRORS = Counter("rag_model_errors_total", "Ошибки внешних моделей", ["kind"])


@contextmanager
def timeit(stage: str, timings: dict):
    """Пишет длительность и в Prometheus, и в запись прогона."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        STAGE.labels(stage).observe(dt)
        timings[stage] = round(timings.get(stage, 0) + dt * 1000, 1)


def save_run(**record) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    record |= {"timestamp": datetime.now().isoformat(),
               "total_ms": round(sum(record.get("timings_ms", {}).values()), 1)}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ─── Чистые помощники ──────────────────────────────────────────
def normalize(name: str) -> str:
    return " ".join(name.lower().replace("ё", "е").strip(" .,;:!?«»\"'()").split())


def uid(kind: str, text: str) -> str:
    return f"{kind[0]}:{hashlib.sha1(text.encode()).hexdigest()[:32]}"


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def parse_json(raw: str, default):
    """Достаёт первый JSON-массив/объект из ответа LLM (markdown, <think>, болтовня вокруг)."""
    raw = strip_think(raw)
    starts = [i for i in (raw.find("["), raw.find("{")) if i >= 0]
    if not starts:
        return default
    start = min(starts)
    end = raw.rfind("]" if raw[start] == "[" else "}")
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return default


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


# ─── Клиенты моделей ───────────────────────────────────────────
llm      = AsyncOpenAI(base_url=cfg["llm"]["url"].rstrip("/"), api_key=cfg["llm"]["api_key"], timeout=180)
embedder = AsyncOpenAI(base_url=cfg["embeddings"]["url"].rstrip("/"), api_key="not-needed", timeout=60)
reranker = AsyncOpenAI(base_url=cfg["reranker"]["url"].rstrip("/"), api_key="not-needed", timeout=60)


async def embed(texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), EMB_BATCH):
        r = await embedder.embeddings.create(input=texts[i:i + EMB_BATCH], model=EMB_MODEL)
        out += [d.embedding for d in r.data]
    return out


async def chat(messages: list[dict], usage: dict | None = None, **kw):
    """usage — необязательный dict, куда кладём prompt/completion токены этого вызова (для бюджета harness)."""
    r = await llm.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.0, **kw)
    if r.usage:
        LLM_TOKENS.labels("prompt").inc(r.usage.prompt_tokens)
        LLM_TOKENS.labels("completion").inc(r.usage.completion_tokens)
        if usage is not None:
            usage.update(prompt=r.usage.prompt_tokens, completion=r.usage.completion_tokens)
    return r.choices[0].message


async def ask_llm(prompt: str) -> str:
    return strip_think((await chat([{"role": "user", "content": prompt}])).content)


async def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    if not documents:
        return []
    data = await reranker.post("/rerank", cast_to=dict, body={
        "model": RERANK_MODEL, "query": query, "documents": documents, "top_n": top_n})
    return [(it["index"], it["relevance_score"]) for it in data["results"]]


# ─── Milvus ────────────────────────────────────────────────────
milvus = MilvusClient(uri=cfg["milvus"]["uri"])
FIELDS = ["text", "type", "doc", "src", "dst", "passage_ids"]


def ensure_collection(reset: bool = False):
    if reset and milvus.has_collection(COLLECTION):
        milvus.drop_collection(COLLECTION)
    if milvus.has_collection(COLLECTION):
        milvus.load_collection(COLLECTION)  # после рестарта коллекция в состоянии released
        return
    s = milvus.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("id", DataType.VARCHAR, is_primary=True, max_length=40)
    s.add_field("dense", DataType.FLOAT_VECTOR, dim=EMB_DIM)
    s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    s.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True, enable_match=True,
                analyzer_params={"tokenizer": "standard",
                                 "filter": ["lowercase", {"type": "stemmer", "language": "russian"}]})
    s.add_field("type", DataType.VARCHAR, max_length=16)
    s.add_field("doc", DataType.VARCHAR, max_length=512)
    s.add_field("src", DataType.VARCHAR, max_length=512)
    s.add_field("dst", DataType.VARCHAR, max_length=512)
    s.add_field("passage_ids", DataType.ARRAY, element_type=DataType.VARCHAR,
                max_capacity=MAX_LINKS, max_length=40)
    s.add_function(Function(name="bm25_fn", function_type=FunctionType.BM25,
                            input_field_names=["text"], output_field_names=["sparse"]))
    idx = milvus.prepare_index_params()
    idx.add_index("dense", index_type="HNSW", metric_type="IP", params={"M": 16, "efConstruction": 200})
    idx.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    milvus.create_collection(COLLECTION, schema=s, index_params=idx)
    log.info("Коллекция %s создана", COLLECTION)


def get_rows(ids: list[str], fields: list[str] = FIELDS, client=None) -> list[dict]:
    # list(): pymilvus 3 отдаёт HybridExtraList, векторы в ней подставляются лениво через __iter__/__getitem__;
    # list.sort() и прочие C-операции их обходят и видят dense=None.
    rows = (client or milvus).query(COLLECTION, filter=f"id in {json.dumps(ids)}", output_fields=fields) if ids else []
    return list(rows)


def counts() -> dict[str, int]:
    return {t: milvus.query(COLLECTION, filter=f'type == "{t}"', output_fields=["count(*)"])[0]["count(*)"]
            for t in ("passage", "entity", "relation")}


# shortcut: синхронный pymilvus блокирует event loop (мс на Lite), при сотнях RPS — AsyncMilvusClient.
def hybrid(query: str, q_dense: list[float], kind: str, limit: int, where: str = "") -> list[dict]:
    """dense + BM25 → RRF по строкам одного типа; where — доп. фильтр Milvus (уже проверенный вызывающим)."""
    expr = f'type == "{kind}"' + (f" and ({where})" if where else "")
    reqs = [AnnSearchRequest([q_dense], "dense", {"metric_type": "IP", "params": {"ef": 64}}, limit * 2, expr=expr),
            AnnSearchRequest([query], "sparse", {"metric_type": "BM25"}, limit * 2, expr=expr)]
    res = milvus.hybrid_search(COLLECTION, reqs, RRFRanker(k=RRF_K), limit=limit, output_fields=FIELDS)
    return [h["entity"] | {"id": h["id"], "score": h["distance"]} for h in (res[0] if res else [])]


async def retrieve(query: str, timings: dict, top_n: int = TOP_PASSAGES, min_score: float | None = None):
    """
    Кандидаты = пассажи, связанные с найденными сущностями и отношениями (граф)
              ∪ пассажи, найденные напрямую гибридным поиском (страховка, если сущность не распознана).
    Дальше: би-энкодер сужает графовых кандидатов → реранкер → порог.
    """
    with timeit("query_embed", timings):
        q = (await embed([query]))[0]
    with timeit("hybrid_search", timings):
        ents = hybrid(query, q, "entity", TOP_ENTITIES)
        rels = hybrid(query, q, "relation", TOP_RELATIONS)
        direct = hybrid(query, q, "passage", PASSAGE_HITS)
    linked_ids = sorted({p for h in ents + rels for p in h["passage_ids"]} - {h["id"] for h in direct})
    with timeit("load_passages", timings):
        linked = get_rows(linked_ids, ["text", "doc", "dense"])
    linked.sort(key=lambda p: -dot(q, p.pop("dense")))
    cands = list({p["id"]: p for p in direct + linked[:RERANK_CANDIDATES]}.values())
    with timeit("rerank", timings):
        ranked = await rerank(query, [c["text"] for c in cands], top_n)
    passages = [{"id": cands[i]["id"], "doc": cands[i]["doc"], "text": cands[i]["text"], "score": float(s)}
                for i, s in ranked if s >= (MIN_RERANK_SCORE if min_score is None else min_score)]
    return ents, rels, passages


# ─── Генерация и верификация ───────────────────────────────────
ANSWER_PROMPT = """Ты — ассистент, отвечающий СТРОГО по предоставленному контексту.

ПРАВИЛА (нарушение любого — ошибка):
1. Используй ТОЛЬКО информацию, которая ЯВНО написана в контексте.
2. НЕ делай логических выводов, которых нет в тексте дословно.
3. Если в контексте нет прямого ответа — напиши: «В предоставленном контексте нет прямого ответа».
4. Не объединяй факты о разных людях/объектах в одно утверждение.
5. Каждое утверждение подкрепляй ссылкой [N] ТОЛЬКО если цитируемый фрагмент содержит это утверждение.
6. Лучше короткий точный ответ, чем длинный с домыслами.
7. Отвечай ТОЛЬКО про того, о ком прямо спрошено. Не добавляй родственников и коллег.
8. Запрещены цепочки выводов вида «X — брат Y, Y — учитель Z, значит X тоже учитель Z».
9. Отвечай ТОЛЬКО на заданный вопрос, без справочной информации вокруг.

ПРИМЕР:
  Контекст:
    [1] Иван Петров родился в Москве.
    [2] Его брат Сергей Петров работал в Киеве.
  Вопрос: Где родился Иван Петров?
  Ответ: Иван Петров родился в Москве [1].

Контекст:
{context}

Вопрос: {query}

Ответ (строго по контексту, только про того, о ком спрошено):"""

VERIFY_PROMPT = """Разбей ответ на АТОМАРНЫЕ утверждения — ровно один факт в каждом.
Для каждого утверждения определи, какой источник [N] его подтверждает.
Верни СТРОГО JSON-массив: [{{"claim": "один факт", "source": "N", "supported": true|false}}]
Только JSON, без пояснений и markdown.

Ответ:
{answer}

Источники:
{sources}

JSON:"""


async def generate(query: str, texts: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
    return await ask_llm(ANSWER_PROMPT.format(context=context, query=query))


async def verify(answer: str, sources: dict[str, str]) -> list[dict]:
    """sources: метка → текст. Возвращает неподтверждённые утверждения."""
    if not answer or not sources or answer.startswith(NO_CONTEXT[:30]):
        return []
    block = "\n".join(f"[{k}] {t}" for k, t in sources.items())
    checks = parse_json(await ask_llm(VERIFY_PROMPT.format(answer=answer, sources=block)), None)
    if not isinstance(checks, list):
        log.warning("Верификатор вернул невалидный JSON")
        MODEL_ERRORS.labels("verify_json").inc()
        return []
    return [c for c in checks if isinstance(c, dict) and c.get("supported") is False]
