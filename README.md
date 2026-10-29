# Graph RAG: ETL + API + Harness

Локальный Graph RAG на русском: Qwen (LLM) + BGE-M3 (dense) + BM25 (sparse) + BGE-reranker-v2-m3 + Milvus.

| Файл | Что внутри |
|---|---|
| `core.py` | конфиг, логи, метрики, клиенты моделей, схема Milvus, гибридный поиск, генерация, верификация |
| `etl.py` | ETL (смонтирован в `api.py` под `/etl`): файл → чанки → триплеты (LLM) → эмбеддинги → upsert в Milvus; CLI `seed`/`reset` |
| `api.py` | единый сервер: `/etl/*`, `/search`, `/ask`, `/harness`, `/metrics/` |
| `harness.py` | агент с атомарными инструментами поверх графа; CLI для прогона вопросов |
| `test_core.py` | assert-тесты чистой логики (без моделей) |

## Запуск

1. Модели (как в AI_diplom):
   - LLM: Unsloth Studio, `unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL` на `:8888` (для Harness включите tool calling)
   - эмбеддинги `bge-m3` на `:8001`, реранкер `bge-reranker-v2-m3` на `:8081` и Prometheus на `:9090`: `docker compose up -d`
     (модели скачиваются в `models/hf-cache` при первом запуске)
2. Конфиг: `cp config.example.ini config.ini` и впишите `api_key` из Unsloth Studio (`config.ini` в `.gitignore`).
   Зависимости: `uv sync` (создаст `.venv` по `pyproject.toml` + `uv.lock`)
3. Тесты: `uv run pytest -q test_core.py`. Новая зависимость: `uv add <пакет>`, dev: `uv add --dev <пакет>`
4. База — Milvus Lite (файл `graph_rag.db`). Файл открывает только один процесс, поэтому ETL смонтирован
   в основной сервер, а `uvicorn` запускается без `--workers`. CLI-команды (`etl.py seed/reset`, `harness.py`)
   запускайте при остановленном сервере.
5. Загрузить данные:
   ```bash
   uv run etl.py seed                                       # dataset.py, LLM не нужен; сервер остановлен
   uv run uvicorn api:app --port 8200
   curl -F files=@book.txt -F files=@notes.md localhost:8200/etl/documents   # → [{job_id,...}]
   curl localhost:8200/etl/jobs/<job_id>                    # queued → running → done|failed
   curl localhost:8200/stats                                # {"passage": N, "entity": N, "relation": N}
   ```
6. Спрашивать:
   ```bash
   curl -XPOST localhost:8200/ask     -H 'content-type: application/json' -d '{"query":"Кто был учителем Эйлера?"}'
   curl -XPOST localhost:8200/harness -H 'content-type: application/json' -d '{"query":"Кто был учителем Эйлера?"}'
   ```
7. Метрики: `localhost:8200/metrics/`; Prometheus: `localhost:9090` (поднимается тем же `docker compose up -d`).
8. Веб-интерфейс: `localhost:8200/ui/` (корень `/` редиректит туда). Собранный `web/dist` уже лежит в проекте,
   node нужен только чтобы менять фронтенд (см. «Веб-интерфейс»).

## ETL

- Форматы: `.txt`/`.md` (триплеты извлекает LLM), `.json` в формате `dataset.py` (триплеты готовые).
- Чанкинг: абзацы склеиваются до `chunk_chars`; длинный абзац режется по предложениям.
- Триплеты проходят валидацию: тип, пустые поля, длина, петли `s == o`, дубли; имена нормализуются (`ё→е`, регистр, пунктуация по краям).
- `id = sha1(содержимого)`. Повторная загрузка того же текста ничего не пишет и не тратит эмбеддинги.
- Новые документы дописывают `passage_ids` к уже существующим сущностям и отношениям (read-merge-write под lock). Эмбеддинги считаются только для новых строк.
- Ошибки сразу дают HTTP-ответ: 413 (размер), 422 (формат/кодировка/пустой файл). Ошибки моделей уходят в `status=failed` задачи.

## Поиск (`/ask`)

Кандидаты = пассажи сущностей ∪ пассажи отношений ∪ прямой гибридный поиск по пассажам → би-энкодер → реранкер → порог → генерация → верификация. Прямой поиск по пассажам добавлен как страховка на случай, когда сущность из вопроса не распознана.

## Harness (`/harness`)

Инструменты атомарные, каждый отдаёт несколько коротких строк:

| Инструмент | Что возвращает |
|---|---|
| `find_entities(query)` | нормализованные имена + число пассажей |
| `neighbors(entity, hops=1\|2)` | `s p o → [P1] [P4]`, связи отсортированы по близости к вопросу; `hops=2` добавляет связи соседей (multi-hop из [Graph RAG Milvus](https://milvus.io/docs/ru/graph_rag_with_milvus.md), но через фильтр `src in [...]` вместо матриц смежности); при неточном имени берётся ближайшая сущность |
| `query_db(kind, filter, query, count, k)` | прямой read-only запрос: булев фильтр Milvus (`like`, `in`, `text_match`, `array_contains`, `array_length`…) + опционально гибридный поиск внутри фильтра или `count`. Фильтр пишет сам агент ([подход Milvus](https://milvus.io/docs/ru/generating_milvus_query_filter_expressions.md)), шпаргалка по схеме лежит в описании инструмента. Белый список полей, хэндлы `"P3"` → id, ошибка Milvus с позицией уходит агенту на исправление (`harness_tool_calls_total{status="bad_query"}`) |
| `search_passages(query)` | `[P#] score doc: превью` |
| `read_passage(ids)` | полный текст; повторное чтение возвращает «уже прочитан выше» |
| `answer(text)` | ответ проходит верификацию; неподтверждённые утверждения возвращаются агенту на исправление (`fix_rounds`) |

Финальный ответ — структурированный: `answer(claims=[{text, sources:["P1"]}], no_answer)`. Код до LLM-верификатора отклоняет факт без источника и неизвестный хэндл, убирает markdown; `tool_choice = required` не даёт ответить мимо `answer` (если LLM-сервер его не умеет — один раз падаем на `auto`). В ответе `/harness` есть поле `claims`.

Как экономится контекст: хэндлы `P#` вместо 34-символьных id, превью вместо текстов, полный текст только по запросу, общий бюджет `budget_chars` (16000; при остатке меньше четверти инструменты предупреждают агента). Ответ содержит `trace` (путь агента) и `context_chars`.

## Метрики

| Метрика | Зачем |
|---|---|
| `rag_stage_seconds{stage}` | латентность каждого этапа: embed, hybrid_search, rerank, generation, verification, etl_*, tool_* |
| `rag_llm_tokens_total{kind}` | расход токенов prompt/completion |
| `rag_model_errors_total{kind}` | недоступность моделей, невалидный JSON верификатора |
| `rag_requests_total{endpoint,outcome}` | answered / unsupported / no_context / error |
| `rag_top_rerank_score`, `rag_passages_used` | качество retrieval: если score проседает, значит база не покрывает вопросы |
| `rag_unsupported_claims_total`, `harness_unsupported_claims_total` | галлюцинации |
| `etl_documents_total{status}`, `etl_chunks_total`, `etl_rows_written_total{type}` | объём загрузки |
| `etl_triples_total{result}` | accepted / duplicate / rejected_format / rejected_value / rejected_json — качество экстрактора |
| `etl_jobs_in_progress` | очередь ETL |
| `harness_tool_calls_total{tool,status}`, `harness_steps`, `harness_context_chars`, `harness_answers_total{outcome}` | поведение агента и экономия контекста |

PromQL:
```promql
histogram_quantile(0.95, sum by (le, stage) (rate(rag_stage_seconds_bucket[5m])))      # p95 по этапам
sum(rate(rag_requests_total{outcome="unsupported"}[1h])) / sum(rate(rag_requests_total[1h]))  # доля галлюцинаций
sum by (result) (rate(etl_triples_total[1h]))                                            # качество экстракции
histogram_quantile(0.5, rate(harness_context_chars_bucket[1h]))                          # медиана контекста агента
```

## Веб-интерфейс (`web/`)

React + Vite + shadcn/ui + Recharts + d3-force. FastAPI отдаёт собранный `web/dist` на `/ui/`; API не меняется.

- **Harness** — живой ход агента через `POST /harness/stream` (SSE): мысли, вызовы инструментов с аргументами и
  результатами, бюджет контекста, растущий подграф связей, финальные утверждения с источниками (наведение на `P#` — текст пассажа).
- **Поиск** — `/search` (сущности, связи, пассажи со score реранкера) рядом с `/ask`.
- **Документы** — drag-and-drop в `/etl/documents`, статусы задач (`GET /etl/jobs`), счётчики Milvus.
- **Метрики** — графики из `GET /dashboard` (снимок Prometheus-метрик в JSON) и `GET /runs` (история `runs/*.json`).

Разработка фронтенда (API должен быть запущен на :8200, dev-сервер проксирует запросы):

```
cd web
npm install
npm run dev
```

Пересобрать то, что отдаёт FastAPI: `cd web && npm run build`.

События `/harness/stream` (`data: {json}` на строку): `start`, `thought`, `call`, `result` (status, out, chars, ms,
total_chars, edges), `done` (полный результат как у `/harness`), `error`.

## Что упрощено (ponytail)

Из исходника убраны: обёртки `step/info/ok/warn/banner` (заменены на `logging`), неиспользуемые `min_entity_score`/`min_entities_keep`, ручной парсинг markdown-JSON (теперь один `parse_json`), auto_id (заменён хэшем, без него невозможна идемпотентная дозагрузка).

Сознательно не сделано, при необходимости добавляется отдельно:
- удаление и замена документа (`DELETE /documents/{doc}`): нужно вычищать `passage_ids` у сущностей;
- PDF/DOCX: сейчас только txt/md/json;
- контекст соседних чанков при извлечении триплетов: местоимения на границе чанков теряются;
- статусы задач ETL хранятся в памяти; при рестарте теряются, но повторная загрузка безопасна;
- `/harness/stream`: если закрыть вкладку, агент доработает в фоне и сохранит run (отмены по disconnect нет);
- `/dashboard` показывает счётчики текущего процесса: после рестарта они с нуля, история — в `runs/` и Prometheus;
- один процесс (Milvus Lite + lock на процесс). Переход на Milvus standalone: `uri = http://localhost:19530`,
  схема и код те же, ETL можно снова вынести: `uvicorn etl:app --port 8100`.
