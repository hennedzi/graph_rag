"""
Harness: LLM-агент поверх графа с атомарными инструментами.

Публичный API:
    run(query, on_event=None, session_id=None, doc=None) -> dict
    cancel_run(session_id) -> bool
    open_session(session_id, query, scope=None) -> (session_id, ses)

... (docstring continues)
"""

import asyncio
import json
import logging
import re
import sys
import time
import urllib.request
from types import SimpleNamespace

import core
import graph
from config import CONFIG
from core import log, normalize, timeit

from .prompts import (
    SYSTEM, TOOLS, MAX_STEPS, BUDGET, FAST_PATH_THRESHOLD, FIX_ROUNDS,
    TOOL_CHOICE, CHARS_PER_TOKEN, LOG_CHARS, hlog,
    TOOL_CALLS, OUTCOME, STEPS, CONTEXT, CONTEXT_TOKENS, SPENT_TOKENS,
    UNSUPPORTED, FOLLOWUPS, OFF_TOPIC,
    META_Q, REWRITE, CHAT_WORDS, CHAT_REPLY, SHORTEN, FILLER, QWORDS,
    REWRITE_PROMPT,
)
from .state import (
    new_state, handle, preview, check_filter, links, remember, rel_line,
    scope_where, passage_lines, graph_part, qvec, _entities, search_rows,
    emit, say, short, open_session, cancel_run, SESSIONS, SessionError,
)
from .tools import (
    find_entities, neighbors, query_db, search_passages, seed_hint,
    read_passage, rank_by_ppr, find_paths, IMPL,
)
from .render import (
    clean_prose, render, lines_to_claims, fmt,
    chat_reply, route, is_question,
)
from .verify import (
    chat_tools, check, off_topic, resolve, thought,
    RELEVANT_PROMPT, RESOLVE_PROMPT,
)

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


class BudgetExceeded(Exception):
    pass


def ctx_now(st) -> int:
    """Токены в окне: точные на момент последнего вызова LLM + оценка того, что добавлено после."""
    tokens = st["ctx"] + st["pending"] // CHARS_PER_TOKEN
    if tokens >= st["budget"]:
        raise BudgetExceeded(f"Context budget {st['budget']} exceeded: {tokens} tokens")
    return tokens


async def llm_step(st, msgs: list[dict], tools: bool = True, on_token=None):
    u: dict = {}
    if on_token is not None:  # C5: финальный ответ стримим, инструменты на этом шаге не нужны
        msg = SimpleNamespace(content=await core.chat_stream(msgs, on_token, usage=u), tool_calls=[])
    else:
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
        bad = await check(st, text) if re.search(r"\[P\d+\]", text) else [{"claim": "в пересказе нет ссылок на источники"}]
    if bad or st["verify_error"]:  # пересказ не проверен или не прошёл проверку — честнее отдать прошлый проверенный ответ
        return f"Пересказать без потери точности не вышло, прошлый ответ:\n{last}", [], "rewrite_fallback"
    if SHORTEN.search(" ".join(words(query))) and len(text) >= len(last):
        return f"Пересказ не стал короче, прошлый ответ:\n{last}", [], "rewrite_fallback"
    return text, [], "rewrite"


def words(text: str) -> list[str]:
    return re.findall(r"[a-zа-я][a-zа-я-]*", text.lower().replace("ё", "е"))


import re
from .prompts import META_Q, REWRITE


def outcome_of(answer: str, claims: list[dict], unsupported: list, cancelled: bool,
               steps_exhausted: bool, verify_error: bool) -> tuple[str, str]:
    """Машинная причина завершения + сообщение из реального результата, не из рассказа модели (FUTURE §4)."""
    if cancelled:
        return "cancelled", "остановлено пользователем; накопленные материалы сохранены"
    if verify_error:
        return "verification_failed", "проверяющая модель не ответила — опора на источники не проверена"
    if not claims:
        if answer and not answer.startswith(core.NO_CONTEXT[:30]):
            # модель ответила текстом без ссылок [P#]: отвечать «нет ответа» было бы ложью
            return "verification_failed", "ответ без ссылок на источники — проверка не выполнена"
        return ("budget_exceeded", "достигнут лимит шагов — ответа не нашлось") if steps_exhausted \
            else ("no_evidence", "в прочитанных документах нет ответа на вопрос")
    if unsupported:
        return "partial", f"ответ частичный: {len(unsupported)} утверждений не подтверждены источниками"
    return "answered", "утверждения ответа подтверждены источниками"


async def run(query: str, on_event=None, session_id: str | None = None, doc: str | None = None, branch_from: int | None = None) -> dict:
    session_id, ses = open_session(session_id, query, scope=doc, branch_from=branch_from)
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


def citations(st, answer: str, sources: list[str] | None = None) -> dict:
    if sources is None:
        sources = [h for h in st["pid"] if f"[{h}]" in answer]
    return {h: {"doc_id": st["pid"][h], "doc": st["doc"].get(st["pid"][h], ""),
                "text": st["text"].get(st["pid"][h], "")} for h in sources}


async def emit_answer(st, answer: str, cited: dict):
    """Печатаем ответ в читаемом темпе: виден процесс генерации, а не «резкий» ответ. Стрим не дублируем."""
    if not st.get("streamed"):
        n = len(answer)
        step = 6                                        # символов за тик — мелкими порциями, как набор текста
        ticks = max(1, (n + step - 1) // step)
        total = min(6.0, max(0.6, n / 220))             # ~220 симв/с, но не дольше 6 с на весь ответ
        nap = total / ticks
        for i in range(0, n, step):
            await emit(st, "token", text=answer[i:i + step])
            if nap > 0:
                await asyncio.sleep(nap)
    await emit(st, "answer_finished", text=answer, citations=cited)


async def final_text(st, msgs: list[dict], instruction: str) -> tuple[str, list[dict], list[dict], list[str]]:
    """C5: финальный ответ текстом — токены стримим в ленту, затем проверяем как answer.
    Возвращает: (text, claims, unsupported, sources)"""
    msgs.append({"role": "user", "content": instruction})
    st["streamed"] = True  # этот ответ уже ушёл токенами — повторно чанковать не нужно
    raw = clean_prose(core.strip_think((await llm_step(st, msgs, tools=False, on_token=lambda t: emit(st, "token", text=t))).content))
    msgs.append({"role": "assistant", "content": raw})
    claims = lines_to_claims(st, raw)
    # Извлекаем источники из claims для верификации
    sources = list(dict.fromkeys(h for c in claims for h in c.get("sources", [])))
    with timeit("harness_verify", st["timings"]):
        unsupported, off = await asyncio.gather(check(st, raw, sources), off_topic(st, claims))
    OFF_TOPIC.inc(len(off))
    claims = [c for c in claims if c not in off]
    if not claims:  # ни одного факта с источником — связного ответа из источников не вышло
        return core.NO_CONTEXT, [], unsupported or [{"claim": "ответ без ссылок на источники"}], []
    return raw, claims, unsupported, sources


async def _run(query: str, on_event, session_id: str, ses: dict) -> dict:
    st, msgs, t0, sid = ses["st"], ses["msgs"], time.perf_counter(), session_id[:6]
    st["sid"] = sid
    st["streamed"] = False  # на каждый вопрос свой стрим: не наследуем флаг прошлого хода
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
    await emit(st, "state_change", state="thinking")
    if seeds:
        st["thoughts"].append({"step": 0, "text": f"Сущности вопроса в графе: {seeds}"})
        await emit(st, "thought", step=0, text=f"Сущности вопроса в графе: {seeds}")
    answer, unsupported, fixes, steps, claims, answer_sources = None, [], 0, 0, [], []

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
        if not calls:  # модель ответила текстом: связный ответ, проверяем как answer
            answer = clean_prose(core.strip_think(msg.content))
            claims = lines_to_claims(st, answer)
            # Извлекаем источники из claims для верификации
            answer_sources = list(dict.fromkeys(h for c in claims for h in c.get("sources", [])))
            if claims:
                with timeit("harness_verify", st["timings"]):
                    unsupported, off = await asyncio.gather(check(st, answer, answer_sources), off_topic(st, claims))
                OFF_TOPIC.inc(len(off))
                claims = [c for c in claims if c not in off]
                unsupported = unsupported if claims else (unsupported or [{"claim": "нет ссылок на источники"}])
            else:  # короткая реплика без фактов — проверять нечего
                unsupported = await check(st, answer, answer_sources)
            break
        done: set = set()
        batch_start = time.perf_counter()
        for c in calls:
            done.add(c.id)
            name, args = c.function.name, core.parse_json(c.function.arguments or "{}", {})
            say(sid, "→ %s %s", name, short(json.dumps(args, ensure_ascii=False)))
            await emit(st, "call", step=steps, id=c.id, tool=name, args=args, started_at=batch_start)

        # ── C2: параллельный dispatch ──
        # Модель генерирует все tool_calls в одном ходе ДО получения результатов, поэтому ни один вызов
        # не может использовать результат другого — все независимы. Выполняем через asyncio.gather.
        # answer — терминальный, обрабатываем отдельно и только после параллельных вызовов.
        async def _tool(c):
            name, args = c.function.name, core.parse_json(c.function.arguments or "{}", {})
            t0 = time.perf_counter()
            edges_before = len(st["edges"])
            status = "ok"
            if ctx_now(st) > B:
                out, status = "Бюджет окна исчерпан. Вызови answer по уже прочитанному.", "budget"
            elif name not in IMPL:
                out, status = f"нет инструмента {name}", "unknown_tool"
            else:
                try:
                    with timeit(f"tool_{name}", st["timings"]):
                        out = await IMPL[name](st, **args)
                    TOOL_CALLS.labels(name, "ok").inc()
                except TypeError as e:
                    out, status = f"ошибка аргументов: {e}", "bad_args"
                    TOOL_CALLS.labels(name, "bad_args").inc()
                except (ValueError, Exception) as e:
                    out, status = f"ошибка запроса: {getattr(e, 'message', None) or e}"[:600], "bad_query"
                    TOOL_CALLS.labels(name, "bad_query").inc()
            st["chars"] += len(out)
            if (left := B - ctx_now(st) - len(out) // CHARS_PER_TOKEN) < B // 4:
                out += f"\n(окно: осталось ~{max(left, 0)} из {B} токенов — пора отвечать)"

            added = list(st["edges"].values())[edges_before:]
            return c, out, status, round((time.perf_counter() - t0) * 1000, 1), added

        tool_calls = [c for c in calls if c.function.name != "answer"]
        ans_calls = [c for c in calls if c.function.name == "answer"]

        if tool_calls:
            for c, out, status, ms, added in await asyncio.gather(*[_tool(c) for c in tool_calls]):
                say(sid, "← %s (%d симв.): %s", c.function.name, len(out), short(out))
                await emit(st, "result", step=steps, id=c.id, tool=c.function.name, status=status, out=out[:3000],
                           chars=len(out), ms=ms, total_chars=st["chars"],
                           ctx_tokens=ctx_now(st) + len(out) // CHARS_PER_TOKEN,
                           edges=added, speculative=st.get("last_speculative", False) if c.function.name == "search_passages" else False, started_at=batch_start)
                st["trace"].append({"tool": c.function.name,
                                    "args": core.parse_json(c.function.arguments or "{}", {}),
                                    "chars": len(out)})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
                st["pending"] += len(out)

        if (not ans_calls and tool_calls and not st.get("fast_path_used")
                and any(c.function.name == "search_passages" for c in tool_calls)
                and st.get("last_search_top", 0.0) >= FAST_PATH_THRESHOLD):
            # C4+C5: реранкер уверен — один стриминговый вызов модели, без графа и второго круга.
            st["fast_path_used"] = True
            await emit(st, "fast_path_taken", top_score=st["last_search_top"])
            answer, claims, unsupported, fast_path_sources = await final_text(
                st, msgs, "Быстрый путь: реранкер уверен. Ответь на вопрос по прочитанным пассажам СВЯЗНЫМ ТЕКСТОМ "
                          "(4–8 предложений, абзацы), без markdown. "
                          f"Если ответа на этот вопрос в прочитанном нет — ответь: {core.NO_CONTEXT}")
            answer_sources = fast_path_sources
            break

        for c in ans_calls:
            name, args = c.function.name, core.parse_json(c.function.arguments or "{}", {})
            t1, status, edges0 = batch_start, "ok", len(st["edges"])
            if name == "answer":
                text, parsed, errors, sources = render(st, args)
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
                    unsupported, off = await asyncio.gather(check(st, text, sources), off_topic(st, parsed))
                if (unsupported or off) and fixes < FIX_ROUNDS:
                    fixes, status = fixes + 1, "unsupported" if unsupported else "off_topic"
                    out = ("Ответ отклонён:\n"
                           + "".join(f"- не подтверждено источниками: {u.get('claim')}\n" for u in unsupported)
                           + "".join(f"- не отвечает на вопрос «{st['query']}»: {c['text']}\n" for c in off)
                           + "Убери эти утверждения. Если без них ответа на вопрос нет — найди именно его "
                             "или вызови answer(no_answer=true).")
                else:
                    if off:  # последний шанс исправиться прошёл — посторонние утверждения просто не считаем ответом
                        OFF_TOPIC.inc(len(off))
                        parsed = [c for c in parsed if c not in off]
                        unsupported = unsupported if parsed else []
                    answer, claims, answer_sources = text, parsed, sources
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
                except TypeError as e:
                    out, status = f"ошибка аргументов: {e}", "bad_args"
                    TOOL_CALLS.labels(name, "bad_args").inc()
                except (ValueError, Exception) as e:
                    out, status = f"ошибка запроса: {getattr(e, 'message', None) or e}"[:600], "bad_query"
                    TOOL_CALLS.labels(name, "bad_query").inc()
                st["chars"] += len(out)
                if (left := B - ctx_now(st) - len(out) // CHARS_PER_TOKEN) < B // 4:
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
        answer_sources = []
    elif answer is None:  # шаги кончились — просим ответ без инструментов, проверяем так же, как answer
        answer, claims, unsupported, answer_sources = await final_text(
            st, msgs, f"Шаги закончились. Ответь на вопрос «{st['query']}» по прочитанным пассажам СВЯЗНЫМ ТЕКСТОМ "
                       "(4–8 предложений, абзацы), ставь sources в аргументе answer, без markdown. "
                       f"Если ответа на этот вопрос в прочитанном нет — ответь: {core.NO_CONTEXT}")
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
    await emit_answer(st, answer, citations(st, answer, answer_sources))
    return finish(st, ses, session_id, query, answer, claims, unsupported, steps, fixes, chars0, t0, "agent", outcome, note, answer_sources)


async def _quick(query: str, kind: str, on_event, session_id: str, ses: dict) -> dict:
    st, t0, chars0 = ses["st"], time.perf_counter(), ses["st"]["chars"]
    st["streamed"] = False
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
    await emit_answer(st, answer, citations(st, answer, []))
    return finish(st, ses, session_id, query, answer, [], unsupported, 0, 0, chars0, t0, kind, outcome, note, [])


def finish(st, ses, session_id, query, answer, claims, unsupported, steps, fixes, chars0, t0, kind, outcome, note, sources: list[str] | None = None) -> dict:
    B = st["budget"]
    st["timings"]["total"] = round((time.perf_counter() - t0) * 1000, 1)
    cited = citations(st, answer, sources)
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