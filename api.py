"""
Основной сервер: поиск, ответ одним проходом, ответ через Harness и ETL в одном процессе.

  uv run uvicorn api:app --port 8200        # один процесс, без --workers: так требует Milvus Lite

  /etl/...       — ETL (etl.py): POST /etl/documents, GET /etl/jobs/{id}, GET /etl/stats
  POST /search   — только retrieval (сущности, отношения, пассажи)
  POST /ask      — retrieval → generation → verification (классический пайплайн)
  POST /harness  — агент с атомарными инструментами (качественнее на multi-hop, экономнее по контексту)
  POST /harness/stream — тот же агент, шаги событиями SSE (для UI)
  GET  /runs, GET /dashboard — история запусков и снимок метрик в JSON (для UI)
  GET  /metrics  — Prometheus
  /ui/           — веб-интерфейс (собранный web/dist)
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import REGISTRY, Counter, Histogram, make_asgi_app
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


@app.post("/harness/stream")
async def harness_stream(q: Query):
    """События агента (start/thought/call/result/done/error) как Server-Sent Events."""
    events: asyncio.Queue = asyncio.Queue()

    async def work():
        try:
            r = await harness.run(q.query, on_event=events.put)
            REQUESTS.labels("harness", "unsupported" if r["unsupported"] else "answered").inc()
            await events.put({"type": "done", "result": r})
        except Exception as e:  # граница фоновой задачи: ошибку отдаём в поток, а не теряем
            if isinstance(e, openai.APIError):
                core.MODEL_ERRORS.labels("harness").inc()
            REQUESTS.labels("harness", "error").inc()
            core.log.exception("Ошибка в /harness/stream")
            await events.put({"type": "error", "detail": f"{type(e).__name__}: {e}"})
        finally:
            await events.put(None)

    # shortcut: если клиент закрыл вкладку, агент всё равно доработает и сохранит run; отмена — task.cancel() по disconnect.
    task = asyncio.create_task(work())

    async def sse():
        while (ev := await events.get()) is not None:
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        await task

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def run_summary(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    return {"id": path.stem, "mode": r.get("mode"), "query": r.get("query"), "timestamp": r.get("timestamp"),
            "total_ms": r.get("total_ms"), "timings_ms": r.get("timings_ms", {}), "steps": r.get("steps"),
            "context_chars": r.get("context_chars"), "unsupported": len(r.get("unsupported") or []),
            "answer": r.get("answer", ""), "tools": [t["tool"] for t in r.get("trace", [])],
            "top_score": max((s.get("score", 0) for s in r.get("sources") or []), default=None)}


@app.get("/runs")
def runs(limit: int = 200):
    """Последние запуски из runs/*.json, старые → новые (для графиков)."""
    paths = sorted(core.RUNS_DIR.glob("*.json"))[-max(1, min(limit, 1000)):] if core.RUNS_DIR.exists() else []
    out = []
    for p in paths:
        try:
            out.append(run_summary(p))
        except (json.JSONDecodeError, KeyError, TypeError):  # битый/старый файл не роняет дашборд
            core.log.warning("Пропущен run %s", p.name)
    return out


@app.get("/runs/{run_id}")
def run_detail(run_id: str):
    path = core.RUNS_DIR / f"{Path(run_id).name}.json"  # Path().name: без ../ наружу
    if not path.exists():
        return JSONResponse({"detail": "нет такого запуска"}, status_code=404)
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/dashboard")
def dashboard():
    """Снимок наших Prometheus-метрик в JSON: UI рисует графики без отдельного Prometheus."""
    return [{"name": m.name, "type": m.type, "help": m.documentation,
             "samples": [{"name": s.name, "labels": s.labels, "value": s.value} for s in m.samples]}
            for m in REGISTRY.collect() if m.name.startswith(("rag_", "harness_", "etl_"))]


@app.get("/stats")
def stats():
    return core.counts()


@app.get("/health")
def health():
    return {"ok": True}


UI_DIR = Path(__file__).parent / "web" / "dist"
if UI_DIR.exists():  # без сборки фронтенда API работает как раньше
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")
