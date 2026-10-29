"""
ETL: документ → чанки → триплеты (LLM) → эмбеддинги → Milvus.

Обычно смонтирован в api.py под /etl (Milvus Lite = один процесс).
  uv run etl.py seed                # залить dataset.py (сервер должен быть остановлен)
  uv run etl.py reset               # пересоздать коллекцию (сервер должен быть остановлен)

Форматы: .txt / .md — триплеты извлекает LLM;
         .json — [{"text": ..., "triples": [{"subject","predicate","object"}]}] как в dataset.py.
"""

import asyncio
import json
import re
import sys
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from prometheus_client import Counter, Gauge, make_asgi_app

import core
from core import cfg, log, normalize, timeit, uid

CHUNK_CHARS = cfg.getint("etl", "chunk_chars")
MAX_BYTES = cfg.getint("etl", "max_upload_mb") * 1024 * 1024
MAX_NAME = 200
UPSERT_BATCH = 1000  # ~4 МБ на батч при dim=1024, далеко от лимита gRPC

DOCS = Counter("etl_documents_total", "Документы по итогу обработки", ["status"])
CHUNKS = Counter("etl_chunks_total", "Чанки, прошедшие через ETL")
TRIPLES = Counter("etl_triples_total", "Триплеты по итогу валидации", ["result"])
ROWS = Counter("etl_rows_written_total", "Строки, записанные в Milvus", ["type"])
INFLIGHT = Gauge("etl_jobs_in_progress", "Задачи в работе")

EXTRACT_PROMPT = """Извлеки из текста факты в виде триплетов (субъект, предикат, объект).

ПРАВИЛА:
1. Только факты, ЯВНО написанные в тексте. Ничего не домысливай.
2. Субъект и объект — полное имя сущности в именительном падеже: «Леонард Эйлер», а не «он» или «Эйлера».
3. Местоимения и «его брат» замени на сущность, о которой идёт речь в тексте.
4. Предикат — короткое отношение в нижнем регистре: «учился у», «родился в», «сформулировал».
5. Один триплет — один факт.

Верни СТРОГО JSON-массив без пояснений:
[{"subject": "...", "predicate": "...", "object": "..."}]

Текст:
<<TEXT>>

JSON:"""


# ─── Transform: чистые функции ─────────────────────────────────
def chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Абзацы склеиваются до size; длинный абзац режется по предложениям, длинное предложение — жёстко."""
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if len(para) <= size:
            pieces += [para] if para else []
        else:
            pieces += [s[i:i + size] for s in re.split(r"(?<=[.!?…])\s+", para) for i in range(0, len(s), size)]
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > size:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    return chunks + [cur] if cur else chunks


def clean_triples(raw) -> list[tuple[str, str, str]]:
    """Отсев мусора от LLM: не тот тип, пустые/длинные поля, петли, дубли."""
    out, seen = [], set()
    for t in raw if isinstance(raw, list) else []:
        if not isinstance(t, dict) or not all(isinstance(t.get(k), str) for k in ("subject", "predicate", "object")):
            TRIPLES.labels("rejected_format").inc()
            continue
        s, o, p = normalize(t["subject"]), normalize(t["object"]), " ".join(t["predicate"].lower().split())
        if not (s and p and o) or max(len(s), len(o), len(p)) > MAX_NAME or s == o:
            TRIPLES.labels("rejected_value").inc()
            continue
        if (s, p, o) in seen:
            TRIPLES.labels("duplicate").inc()
            continue
        seen.add((s, p, o))
        out.append((s, p, o))
    TRIPLES.labels("accepted").inc(len(out))
    return out


def build_graph(pids: list[str], triples: list[list[tuple]]):
    """Сущность/отношение → множество пассажей, в которых они упомянуты."""
    ent, rel = defaultdict(set), defaultdict(set)
    for pid, ts in zip(pids, triples):
        for s, p, o in ts:
            ent[s].add(pid)
            ent[o].add(pid)
            rel[(s, p, o)].add(pid)
    return ent, rel


# ─── Extract + Load ────────────────────────────────────────────
_llm_slots = asyncio.Semaphore(cfg.getint("etl", "extract_concurrency"))
# shortcut: lock на процесс делает read-merge-write безопасным при одном воркере, для нескольких — очередь с одним писателем.
_write_lock = asyncio.Lock()


async def triples_of(item: dict) -> list[tuple]:
    if "triples" in item:
        return clean_triples(item["triples"])
    async with _llm_slots:
        raw = await core.ask_llm(EXTRACT_PROMPT.replace("<<TEXT>>", item["text"]))
    triples = core.parse_json(raw, None)
    if triples is None:
        TRIPLES.labels("rejected_json").inc()
        log.warning("Экстрактор вернул невалидный JSON: %.120s", raw)
    return clean_triples(triples)


async def ingest(doc: str, items: list[dict], timings: dict) -> dict:
    texts = [it["text"] for it in items]
    pids = [uid("passage", t) for t in texts]
    CHUNKS.inc(len(texts))

    with timeit("etl_extract", timings):
        triples = await asyncio.gather(*map(triples_of, items))
    ent, rel = build_graph(pids, triples)

    rows = [{"id": pid, "text": t, "type": "passage", "doc": doc, "src": "", "dst": "", "passage_ids": []}
            for pid, t in zip(pids, texts)]
    rows += [{"id": uid("entity", e), "text": e, "type": "entity", "doc": "", "src": "", "dst": "",
              "passage_ids": sorted(ps)} for e, ps in ent.items()]
    rows += [{"id": uid("relation", "|".join(k)), "text": " ".join(k), "type": "relation", "doc": "",
              "src": k[0], "dst": k[2], "passage_ids": sorted(ps)} for k, ps in rel.items()]
    rows = list({r["id"]: r for r in rows}.values())  # одинаковые чанки внутри документа

    async with _write_lock:
        with timeit("etl_merge", timings):
            old = {r["id"]: r for r in core.get_rows([r["id"] for r in rows], ["dense", "passage_ids"])}
            for r in rows:
                if r["id"] in old:
                    r["dense"] = old[r["id"]]["dense"]
                    r["passage_ids"] = sorted({*old[r["id"]]["passage_ids"], *r["passage_ids"]})[:core.MAX_LINKS]
            # неизменившиеся строки не переписываем
            rows = [r for r in rows if r["id"] not in old or r["passage_ids"] != sorted(old[r["id"]]["passage_ids"])]
        fresh = [r for r in rows if "dense" not in r]
        with timeit("etl_embed", timings):
            for r, v in zip(fresh, await core.embed([r["text"] for r in fresh])):
                r["dense"] = v
        with timeit("etl_upsert", timings):
            for i in range(0, len(rows), UPSERT_BATCH):
                core.milvus.upsert(core.COLLECTION, rows[i:i + UPSERT_BATCH])
    for t in ("passage", "entity", "relation"):
        ROWS.labels(t).inc(sum(r["type"] == t for r in rows))

    return {"doc": doc, "chunks": len(texts), "triples": sum(map(len, triples)),
            "entities": len(ent), "relations": len(rel), "rows_written": len(rows), "embedded": len(fresh)}


def parse_upload(name: str, data: bytes) -> list[dict]:
    """Файл → список items для ingest. Ошибки формата → ValueError."""
    text = data.decode("utf-8")
    if Path(name).suffix.lower() == ".json":
        items = json.loads(text)
        if not isinstance(items, list) or not all(isinstance(i, dict) and isinstance(i.get("text"), str) for i in items):
            raise ValueError('ожидается [{"text": str, "triples": [...]}]')
        return [{"text": " ".join(i["text"].split()), **({"triples": i["triples"]} if "triples" in i else {})}
                for i in items if i["text"].strip()]
    if Path(name).suffix.lower() in (".txt", ".md"):
        return [{"text": c} for c in chunk(text)]
    raise ValueError("поддерживаются .txt, .md, .json")


# ─── Задачи ────────────────────────────────────────────────────
# shortcut: статусы в памяти теряются при рестарте (данные целы, повтор загрузки идемпотентен), нужна история — таблица в SQLite.
JOBS: dict[str, dict] = {}


async def run_job(job_id: str, doc: str, items: list[dict]):
    job, timings = JOBS[job_id], {}
    INFLIGHT.inc()
    job["status"] = "running"
    try:
        job |= await ingest(doc, items, timings) | {"status": "done"}
        DOCS.labels("done").inc()
        log.info("ETL %s: %s", doc, {k: job[k] for k in ("chunks", "triples", "rows_written")})
    except Exception as e:  # граница фоновой задачи: любая ошибка → статус failed, а не вечный running
        if isinstance(e, openai.APIError):
            core.MODEL_ERRORS.labels("etl").inc()
        job |= {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        DOCS.labels("failed").inc()
        log.exception("ETL %s упал", doc)
    finally:
        INFLIGHT.dec()
        job["timings_ms"] = timings


# ─── HTTP ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_):
    core.ensure_collection()
    yield


app = FastAPI(title="Graph RAG ETL", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.post("/documents", status_code=202)
async def upload(files: list[UploadFile], background: BackgroundTasks):
    """Принимает файлы, валидирует сразу, обрабатывает в фоне. Возвращает job_id на каждый файл."""
    jobs = []
    for f in files:
        data = await f.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename}: больше {MAX_BYTES // 2**20} МБ")
        try:
            items = parse_upload(f.filename or "", data)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            DOCS.labels("rejected").inc()
            raise HTTPException(422, f"{f.filename}: {e}")
        if not items:
            raise HTTPException(422, f"{f.filename}: пустой документ")
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"job_id": job_id, "doc": f.filename, "status": "queued", "chunks": len(items)}
        background.add_task(run_job, job_id, f.filename[:512], items)
        jobs.append(JOBS[job_id])
    return jobs


@app.get("/jobs")
def job_list():
    return list(JOBS.values())[::-1]


@app.get("/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "нет такой задачи")
    return JOBS[job_id]


@app.get("/stats")
def stats():
    return core.counts()


@app.get("/health")
def health():
    return {"ok": True}


# ─── CLI ───────────────────────────────────────────────────────
async def _seed():
    from dataset import DATASET
    core.ensure_collection()
    timings = {}
    print(await ingest("dataset.py", list(DATASET), timings), timings, core.counts())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "reset":
        core.ensure_collection(reset=True)
    elif cmd == "seed":
        asyncio.run(_seed())
    else:
        sys.exit(__doc__)
    core.milvus.close()
