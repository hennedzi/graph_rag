"""
Основной сервер: поиск, ответ одним проходом, ответ через Harness и ETL в одном процессе.

  uv run uvicorn api:app --port 8200        # один процесс, без --workers: так требует Milvus Lite

  /etl/...       — ETL (etl.py): POST /etl/documents, GET /etl/jobs/{id}, GET /etl/stats
  POST /search   — только retrieval (сущности, отношения, пассажи)
  POST /ask      — retrieval → generation → verification (классический пайплайн)
  POST /harness  — агент с атомарными инструментами (качественнее на multi-hop, экономнее по контексту)
  GET  /metrics  — Prometheus
"""

from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

import core
import etl
import harness
from core import timeit

REQUESTS = Counter("rag_requests_total", "Запросы по итогу", ["endpoint", "outcome"])
UNSUPPORTED = Counter("rag_unsupported_claims_total", "Неподтверждённые утверждения в /ask")
TOP_SCORE = Histogram("rag_top_rerank_score", "Лучший score реранкера",
                      buckets=(-5, -2, 0, 1, 2, 3, 4, 5, 6, 8, 10))  # логиты llama.cpp, не 0..1
PASSAGES = Histogram("rag_passages_used", "Пассажей в контексте генерации", buckets=(0, 1, 2, 3, 5, 8, 13))


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@asynccontextmanager
async def lifespan(_):
    core.ensure_collection()
    yield


app = FastAPI(title="Graph RAG", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())
# shortcut: Milvus Lite = один процесс, поэтому ETL здесь же; на Milvus standalone — отдельный uvicorn etl:app.
app.mount("/etl", etl.app)


@app.exception_handler(openai.APIError)
async def model_down(request: Request, e: openai.APIError):
    core.MODEL_ERRORS.labels(request.url.path.strip("/")).inc()
    REQUESTS.labels(request.url.path.strip("/"), "error").inc()
    core.log.error("Модель недоступна: %s", e)
    return JSONResponse({"detail": f"модель недоступна: {type(e).__name__}"}, status_code=502)


@app.exception_handler(Exception)
async def crashed(request: Request, e: Exception):
    # uvicorn пишет трейсбек только в консоль; здесь он попадает и в logs/graph_rag.log, а клиент получает JSON.
    REQUESTS.labels(request.url.path.strip("/"), "error").inc()
    core.log.exception("Ошибка в %s", request.url.path)
    return JSONResponse({"detail": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/search")
async def search(q: Query):
    timings = {}
    ents, rels, ps = await core.retrieve(q.query, timings)
    REQUESTS.labels("search", "answered" if ps else "no_context").inc()
    return {"entities": [e["text"] for e in ents], "relations": [r["text"] for r in rels],
            "passages": ps, "timings_ms": timings}


@app.post("/ask")
async def ask(q: Query):
    timings = {}
    ents, rels, ps = await core.retrieve(q.query, timings)
    PASSAGES.observe(len(ps))
    TOP_SCORE.observe(ps[0]["score"] if ps else 0)
    if not ps:
        REQUESTS.labels("ask", "no_context").inc()
        return {"answer": core.NO_CONTEXT, "sources": [], "unsupported": [], "timings_ms": timings}

    texts = [p["text"] for p in ps]
    with timeit("generation", timings):
        answer = await core.generate(q.query, texts)
    with timeit("verification", timings):
        unsupported = await core.verify(answer, {str(i): t for i, t in enumerate(texts, 1)})

    UNSUPPORTED.inc(len(unsupported))
    REQUESTS.labels("ask", "unsupported" if unsupported else "answered").inc()
    run = core.save_run(mode="ask", query=q.query, answer=answer, sources=ps, unsupported=unsupported,
                        entities=[e["text"] for e in ents], relations=[r["text"] for r in rels], timings_ms=timings)
    return {"answer": answer, "sources": ps, "unsupported": unsupported, "timings_ms": timings, "run": run.name}


@app.post("/harness")
async def ask_harness(q: Query):
    r = await harness.run(q.query)
    REQUESTS.labels("harness", "unsupported" if r["unsupported"] else "answered").inc()
    return r


@app.get("/stats")
def stats():
    return core.counts()


@app.get("/health")
def health():
    return {"ok": True}
