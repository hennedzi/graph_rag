"""Самопроверка чистой логики без моделей: uv run pytest -q test_core.py"""
import time
from pymilvus import DataType, MilvusClient

import core
from core import normalize, parse_json, uid
from etl import build_graph, chunk, clean_triples


def test_normalize():
    assert normalize("  Пётр   ВЕЛИКИЙ, ") == "петр великий"


def test_parse_json():
    assert parse_json('<think>хм [1]</think>```json\n[{"a": 1}]\n```', None) == [{"a": 1}]
    assert parse_json("Вот: {\"x\": [1, 2]} готово", None) == {"x": [1, 2]}
    assert parse_json("нет json", []) == []
    assert parse_json("[битый", None) is None


def test_uid_stable():
    assert uid("entity", "эйлер") == uid("entity", "эйлер") != uid("relation", "эйлер")
    assert len(uid("passage", "x" * 10_000)) <= 40


def test_chunk():
    assert chunk("") == []
    assert chunk("а.\n\nб.", size=100) == ["а.\nб."]
    long = "Слово. " * 100
    parts = chunk(long, size=50)
    assert all(len(p) <= 50 for p in parts) and "".join(parts).count("Слово") == 100
    assert all(len(p) <= 10 for p in chunk("x" * 35, size=10))


def test_clean_triples():
    raw = [{"subject": "Эйлер", "predicate": "Учился  у", "object": "Иоганн Бернулли"},
           {"subject": "эйлер", "predicate": "учился у", "object": "иоганн бернулли"},  # дубль
           {"subject": "x", "predicate": "p", "object": "X"},                           # петля
           {"subject": "", "predicate": "p", "object": "y"},                            # пусто
           {"subject": 1, "predicate": "p", "object": "y"}, "мусор"]
    assert clean_triples(raw) == [("эйлер", "учился у", "иоганн бернулли")]
    assert clean_triples(None) == []


def test_build_graph():
    ent, rel = build_graph(["p1", "p2"], [[("a", "r", "b")], [("a", "r", "b"), ("b", "q", "c")]])
    assert ent == {"a": {"p1", "p2"}, "b": {"p1", "p2"}, "c": {"p2"}}
    assert rel[("a", "r", "b")] == {"p1", "p2"}


def test_get_rows_vectors_survive_sort(tmp_path):
    # Регрессия: dense=None после linked.sort() в retrieve (ленивые векторы pymilvus 3).
    c = MilvusClient(str(tmp_path / "t.db"))
    schema = c.create_schema()
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=2)
    idx = c.prepare_index_params()
    idx.add_index("dense", index_type="FLAT", metric_type="IP")
    c.create_collection(core.COLLECTION, schema=schema, index_params=idx)
    c.insert(core.COLLECTION, [{"id": "a", "dense": [1.0, 0.0]}, {"id": "b", "dense": [0.0, 1.0]}])
    rows = core.get_rows(["a", "b"], ["dense"], client=c)
    rows.sort(key=lambda r: -r["dense"][1])
    assert [r["id"] for r in rows] == ["b", "a"]
    c.close()


def test_check_filter():
    import pytest
    from harness import check_filter, new_state
    st = new_state()
    st["pid"]["P1"] = "p:abc"
    assert check_filter(st, ' src like "%бернулли%" and not (dst == "x") ') == 'src like "%бернулли%" and not (dst == "x")'
    assert check_filter(st, 'array_contains(passage_ids, "P1")') == 'array_contains(passage_ids, "p:abc")'
    assert check_filter(st, 'text == "dense or sparse"') == 'text == "dense or sparse"'  # слова в кавычках не проверяются
    assert check_filter(st, "") == ""
    for bad in ('dense != 1', 'type == "passage"', 'src == "a" or exec("x")', "x" * 501):
        with pytest.raises(ValueError):
            check_filter(st, bad)


def test_thought():
    from types import SimpleNamespace as NS
    from harness import thought
    assert thought(NS(reasoning_content="план", content="", model_extra={})) == "план"
    assert thought(NS(content="<think> ищу связи </think>ответ", model_extra={})) == "ищу связи"
    assert thought(NS(content="<think>оборвалось", model_extra={})) == "оборвалось"
    assert thought(NS(content="просто ответ", model_extra={})) == ""


def test_read_passage_reports_skipped():
    import asyncio
    from harness import new_state, read_passage
    st = new_state()
    for i in range(1, 13):
        st["pid"][f"P{i}"] = f"p:{i}"
        st["text"][f"p:{i}"] = f"текст {i}"
    out = asyncio.run(read_passage(st, [f"P{i}" for i in range(1, 13)]))
    assert "[P10] текст 10" in out and "текст 11" not in out and "не прочитаны (не больше 10 за вызов): P11 P12" in out
    assert "уже прочитан выше" in asyncio.run(read_passage(st, ["P1"]))


def test_passage_lines_marks_read():
    from harness import new_state, passage_lines
    st = new_state()
    rows = [{"id": "p:1", "doc": "d", "text": "один"}, {"id": "p:2", "doc": "d", "text": "два", "score": 3.5}]
    out = passage_lines(st, rows)
    assert out == "[P1] d: один\n[P2] 3.50 d: два"
    st["read"].add("P1")
    assert passage_lines(st, rows[:1]) == "[P1] уже прочитан\nНовых пассажей нет. Если прочитанного хватает для ответа — вызови answer."


def test_render_answer():
    from harness import new_state, render
    st = new_state()
    st["pid"] |= {"P1": "p:1", "P2": "p:2"}
    st["read"] |= {"P1", "P2"}

    # Новый формат: текст БЕЗ [P#], источники в отдельном аргументе sources
    text, claims, errors, sources = render(st, {"text": "Эйлер учился в Базеле.", "sources": ["P1", "P2"]})
    assert (text, errors, sources) == ("Эйлер учился в Базеле.", [], ["P1", "P2"])
    assert claims[0]["sources"] == ["P1", "P2"]

    # Обратная совместимость: текст С [P#] тоже работает
    text, claims, errors, sources = render(st, {"text": "**Эйлер** учился в Базеле [P1] [P2]."})
    assert (text, errors, sources) == ("Эйлер учился в Базеле [P1] [P2].", [], ["P1", "P2"])
    assert claims[0]["sources"] == ["P1", "P2"]

    # кириллическая П в ссылках нормализуется к латинской P (модель путает алфавиты)
    text, claims, errors, sources = render(st, {"text": "Эйлер учился у Бернулли [П1] [п2]."})
    assert (text, errors, sources) == ("Эйлер учился у Бернулли [P1] [P2].", [], ["P1", "P2"])
    assert claims[0]["sources"] == ["P1", "P2"]

    # кириллическая П вне хэндлов не портится
    text, claims, errors, sources = render(st, {"text": "Профессор читает лекцию [P1]."})
    assert text == "Профессор читает лекцию [P1]." and errors == [] and sources == ["P1"]

    # Ошибки: неизвестные хэндлы
    assert "неизвестные хэндлы P9" in render(st, {"text": "факт [P9]"})[2][0]
    # Ошибки: источники не прочитаны
    st2 = new_state()
    st2["pid"]["P1"] = "p:1"  # P1 известен, но не прочитан
    assert "источники не прочитаны: P1" in render(st2, {"text": "факт [P1]"})[2][0]
    # Текст без источников (и без [P#]) теперь допустим, если переданы sources
    text, claims, errors, sources = render(st, {"text": "Эйлер учился в Базеле.", "sources": ["P1"]})
    assert errors == [] and sources == ["P1"]
    # Но если нет ни sources, ни [P#] — это ошибка только при верификации (claims пустые)
    text, claims, errors, sources = render(st, {"text": "Эйлер учился в Базеле."})
    assert sources == [] and claims == []  # claims пустые, верификатор не вызовется

    assert render(st, {"no_answer": True})[0].startswith("В предоставленном контексте нет")
    assert render(st, {"claims": [{"text": "факт", "sources": ["P1"]}]})[0] == "факт [P1]"  # старый формат → текст
    assert render(st, {"text": "   "})[2] == ["пустой ответ: передай text"]


def test_rel_line_records_edge():
    from harness import new_state, rel_line
    st = new_state()
    r = {"id": "r1", "text": "a учил b", "src": "a", "dst": "b", "passage_ids": ["p1"]}
    assert rel_line(st, r) == "a учил b → [P1]"
    rel_line(st, r)  # повтор не дублирует ребро
    assert list(st["edges"].values()) == [{"src": "a", "dst": "b", "text": "a учил b", "handles": ["P1"]}]


def test_emit_without_listener_is_noop():
    import asyncio
    from harness import emit, new_state
    st, got = new_state(), []
    asyncio.run(emit(st, "start", query="q"))
    assert st["events"][0]["seq"] == 1  # событие попало в журнал запуска и без подписчика

    async def put(ev):
        got.append(ev)
    st["emit"] = put
    asyncio.run(emit(st, "call", tool="neighbors"))
    assert got == [{"v": 1, "seq": 2, "type": "call", "tool": "neighbors", "duration_ms": 0}]  # живой UI и журнал получают одно и то же


def test_run_summary(tmp_path):
    import json
    from api import run_summary
    p = tmp_path / "20260101_000000_000000.json"
    p.write_text(json.dumps({"mode": "ask", "query": "q", "total_ms": 5, "sources": [{"score": 2.5}, {"score": 4.0}],
                             "unsupported": [{"claim": "x"}]}), encoding="utf-8")
    s = run_summary(p)
    assert (s["id"], s["top_score"], s["unsupported"], s["tools"]) == (p.stem, 4.0, 1, [])


def test_session_keeps_dialog_state():
    import pytest
    from harness import SESSIONS, SessionError, close_calls, open_session
    sid, ses = open_session(None, "Кто учитель Эйлера?")
    ses["st"]["read"].update({"P2", "P10"})
    ses["st"]["chars"] = 1234
    with pytest.raises(SessionError):  # пока идёт вопрос, второй параллельно нельзя
        open_session(sid, "ещё")
    ses["busy"] = False
    sid2, ses2 = open_session(sid, "А где он работал?")
    assert sid2 == sid and ses2["turns"] == 2 and ses2["st"]["chars"] == 1234
    assert ses2["msgs"][-1]["content"].endswith("(уже прочитаны: P2 P10)")
    assert ses2["st"]["query"] == "А где он работал?" and ses2["st"]["trace"] == []
    with pytest.raises(SessionError):
        open_session("нет-такой", "q")

    class C:
        def __init__(self, i): self.id = i
    msgs = []
    close_calls(msgs, [C("a"), C("b")], {"a"})
    assert [m["tool_call_id"] for m in msgs] == ["b"]
    SESSIONS.pop(sid)


def test_llm_step_counts_real_tokens(monkeypatch):
    import asyncio
    import harness
    from types import SimpleNamespace as NS

    async def fake_chat(msgs, usage=None, **kw):
        usage.update(prompt=5000, completion=700)  # 700 — с рассуждениями, в окно они не попадают
        return NS(content="ok", tool_calls=[NS(function=NS(arguments='{"q":"x"}'))])
    monkeypatch.setattr(harness.core, "chat", fake_chat)
    st = harness.new_state("q")
    st["pending"] = 999
    asyncio.run(harness.llm_step(st, [{"role": "user", "content": "q"}]))
    assert st["ctx"] == 5000 and st["spent"] == 5700
    assert st["pending"] == len("ok") + len('{"q":"x"}')  # старая оценка сброшена точным usage
    st["pending"] += 3000  # ответ инструмента после вызова
    assert harness.ctx_now(st) == 5000 + (2 + 9 + 3000) // harness.CHARS_PER_TOKEN


def test_canonical_aliases():
    from graph import canonical
    names = {"эйлер", "леонард эйлер", "бернулли", "якоб бернулли", "иоганн бернулли", "якоб ii бернулли", "1738"}
    assert canonical(names) == {"эйлер": "леонард эйлер"}  # «бернулли» неоднозначно, «якоб бернулли» ≠ «якоб ii»


def test_ppr_power_iteration():
    from graph import ppr
    # цепочка b—a—c—d, рестарт в b: масса убывает с расстоянием, сумма = 1, r — неподвижная точка уравнения
    adj = {"a": {"b": 1, "c": 1}, "b": {"a": 1}, "c": {"a": 1, "d": 1}, "d": {"c": 1}, "x": {}}
    r = ppr(adj, {"b": 1.0}, iters=500, tol=1e-13)
    assert r["a"] > r["c"] > r["d"] > 0 and r["b"] > r["c"] and "x" not in r
    assert abs(sum(r.values()) - 1) < 1e-9
    for u in r:  # r(u) = α·v(u) + (1−α)·Σ r(w)/deg(w) по соседям w
        assert abs(r[u] - (0.15 * (u == "b") + 0.85 * sum(r[w] / len(adj[w]) for w in adj[u]))) < 1e-9
    assert ppr(adj, {"нет": 1.0}) == {}
    iso = ppr(adj, {"x": 1.0})  # висячая вершина: вся масса возвращается в seed
    assert abs(iso["x"] - 1) < 1e-9


def test_passage_scores_split_mass():
    from graph import passage_scores
    s = passage_scores({"хаб": 0.5, "узкая": 0.2}, {"хаб": {"p1", "p2", "p3", "p4", "p5"}, "узкая": {"p1"}})
    assert abs(s["p1"] - 0.3) < 1e-9 and abs(s["p2"] - 0.1) < 1e-9  # хаб делит массу на 5 пассажей


def test_paths_rank_short_and_narrow():
    from graph import paths
    adj = {"a": {"hub": 1, "m": 1, "b": 1}, "hub": {"a": 1, "b": 1, "x": 1, "y": 1, "z": 1}, "m": {"a": 1, "b": 1},
           "b": {"a": 1, "hub": 1, "m": 1}, "x": {"hub": 1}, "y": {"hub": 1}, "z": {"hub": 1}}
    assert paths(adj, "a", "b", 2) == [["a", "b"], ["a", "m", "b"], ["a", "hub", "b"]]  # через хаб — последним
    assert all(len(set(p)) == len(p) for p in paths(adj, "a", "b", 4))  # только простые пути
    assert paths(adj, "a", "a") == [] and paths(adj, "a", "нет") == []


def test_build_graph_merges_aliases():
    from graph import build
    ents = [{"text": "эйлер", "passage_ids": ["p1"]}, {"text": "леонард эйлер", "passage_ids": ["p2"]},
            {"text": "базель", "passage_ids": ["p1"]}, {"text": "одинокая", "passage_ids": ["p3"]}]
    rels = [{"id": "r1", "text": "эйлер родился в базель", "src": "эйлер", "dst": "базель", "passage_ids": ["p1"]},
            {"id": "r2", "text": "леонард эйлер работал в петербург", "src": "леонард эйлер", "dst": "петербург",
             "passage_ids": ["p2"]}]
    g = build(ents, rels)
    assert set(g["adj"]["леонард эйлер"]) == {"базель", "петербург"} and "эйлер" not in g["adj"]
    assert g["group"]["леонард эйлер"] == ["леонард эйлер", "эйлер"] and g["ent_passages"]["леонард эйлер"] == {"p1", "p2"}
    assert g["stats"] | {} == {"nodes": 4, "edges": 2, "avg_degree": 1.0, "components": 2, "largest_component": 3, "aliases": 1}
    assert g["stats_raw"]["components"] == 3  # до склейки граф фрагментирован: эйлер и леонард эйлер — разные


def test_ppr_and_paths_tools(monkeypatch):
    import asyncio
    import graph
    import harness
    rels = [{"id": "r1", "text": "иоганн бернулли учитель леонард эйлер", "src": "иоганн бернулли", "dst": "леонард эйлер", "passage_ids": ["p1"]},
            {"id": "r2", "text": "леонард эйлер работал в петербург", "src": "леонард эйлер", "dst": "петербург", "passage_ids": ["p2"]}]
    g = graph.build([{"text": "иоганн бернулли", "passage_ids": ["p1"]}], rels)
    monkeypatch.setattr(graph, "get", lambda: g)
    monkeypatch.setattr(harness.core, "get_rows", lambda ids, fields=None: [{"id": i, "text": f"текст {i}", "doc": "d"} for i in ids])
    st = harness.new_state("q")
    out = asyncio.run(harness.find_paths(st, "Иоганн Бернулли", "петербург"))
    assert "иоганн бернулли —учитель→ леонард эйлер —работал в→ петербург  [P1] [P2]" in out
    assert len(st["edges"]) == 2
    out = asyncio.run(harness.rank_by_ppr(st, ["иоганн бернулли"], k=2))
    assert out.startswith("seed: иоганн бернулли\nсущности по PPR") and "леонард эйлер" in out.split("\n")[1]
    assert "иоганн бернулли учитель леонард эйлер → [P1]" in out


def test_eval_hit():
    from eval import hit
    assert hit("Учителем был Иоганн Бернулли [P1]", ["иоганн"]) and not hit("Якоб", ["иоганн"])
    assert hit("В предоставленном контексте нет прямого ответа.", []) and not hit("Иоганн", [])


def test_route():
    from harness import chat_reply, route
    for q in ["Спасибо!", "Теперь лучше, спасибо", "спасибо большое", "Привет", "ок", "понятно, спасибо", "...", "Добрый день"]:
        assert route(q, True) == "chat", q
    for q in ["Опиши проще", "а можно короче?", "объясни попроще, пожалуйста", "перескажи своими словами"]:
        assert route(q, True) == "rewrite", q
        assert route(q, False) == "agent", q  # нечего пересказывать — пусть агент ищет
    for q in ["Кто учитель Эйлера?", "Эйлер", "кратко расскажи про Лагранжа", "Да, а где он родился?"]:
        assert route(q, True) == "agent", q
    assert chat_reply("спасибо").startswith("Пожалуйста") and chat_reply("Добрый день").startswith("Здравствуйте")


def test_answer_must_fit_question(monkeypatch):
    import asyncio
    import harness
    from types import SimpleNamespace as NS

    async def fake_ask(prompt):
        return '<think>…</think>[{"n": 1, "off_topic": true}, {"n": 2, "off_topic": false}]'
    monkeypatch.setattr(harness.core, "ask_llm", fake_ask)
    st = harness.new_state("Как работает гибридный поиск?")
    claims = [{"text": "Панели тянутся мышью", "sources": ["P1"]}, {"text": "BM25 + dense + реранкер", "sources": ["P2"]}]
    assert asyncio.run(harness.off_topic(st, claims)) == [claims[0]]  # верный, но посторонний факт

    async def broken_ask(prompt):
        return "Оба утверждения [1] и [2] по теме"
    monkeypatch.setattr(harness.core, "ask_llm", broken_ask)
    assert asyncio.run(harness.off_topic(st, claims)) == []  # судья сломался — ответ не трогаем
    st["pid"] = {"P1": "p1", "P2": "p2"}
    got = harness.lines_to_claims(st, "* **Гибридный** поиск: BM25 + dense [P2]\nбез ссылки\nчужой хэндл [P9]")
    assert got == [{"text": "Гибридный поиск: BM25 + dense", "sources": ["P2"]}]

    async def fake_chat(msgs, usage=None, **kw):
        return NS(content="«Как работает гибридный поиск?»\n")
    monkeypatch.setattr(harness.core, "chat", fake_chat)
    ses = {"st": st, "questions": ["Как работает гибридный поиск?"], "prev": "Панели тянутся мышью [P1]"}
    assert asyncio.run(harness.resolve(ses, "Это не ответ на мой вопрос!")) == "Как работает гибридный поиск?"
    assert asyncio.run(harness.resolve({"st": st}, "Кто такой Эйлер?")) == "Кто такой Эйлер?"  # первый вопрос — без LLM


def test_reply_guard():
    from harness import new_state, render
    st = new_state("Кто учитель Эйлера?")
    assert render(st, {"reply": "Иоганн Бернулли"})[2]  # вопрос к базе: reply без источников нельзя
    assert render(st, {"reply": "Иоганн Бернулли", "no_answer": True})[0].startswith("В предоставленном контексте нет")
    st = new_state("Я студент")
    assert render(st, {"reply": "Хорошо, спрашивайте."}) == ("Хорошо, спрашивайте.", [], [], [])
    assert render(st, {"reply": "Очень короткий ответ выходит", "no_answer": True}) == ("Очень короткий ответ выходит", [], [], [])
    assert render(st, {"reply": "Эйлер родился в Базеле [P1]"})[2]

    # вопрос о самом ассистенте — это не вопрос к базе, reply разрешён
    st = new_state("Что ты умеешь?")
    assert render(st, {"reply": "Ищу по базе знаний и отвечаю со ссылками."}) == ("Ищу по базе знаний и отвечаю со ссылками.", [], [], [])

    # reply с хэндлом [P#] — это факт: принимаем как текст и проверяем как обычно
    st = new_state("README.md")
    st["pid"]["P1"] = "p:1"
    st["read"].add("P1")
    text, claims, errors, sources = render(st, {"reply": "Кратко о проекте [P1]."})
    assert claims and claims[0]["sources"] == ["P1"] and not errors and sources == ["P1"]


def test_unlink():
    from etl import unlink
    rows = [{"id": "e1", "passage_ids": ["a", "b"]}, {"id": "e2", "passage_ids": ["a"]}]
    keep, gone = unlink(rows, {"a"})
    assert keep == [{"id": "e1", "passage_ids": ["b"]}] and gone == ["e2"]


def test_graph_part():
    from harness import graph_part, new_state
    st = new_state("q")
    rel = {"id": "r1", "src": "иоганн бернулли", "dst": "леонард эйлер", "text": "иоганн бернулли учитель леонард эйлер", "passage_ids": ["p1"]}
    out = graph_part(st, [{"text": "леонард эйлер"}], [rel])
    assert "сущности леонард эйлер" in out and "[P1]" in out and "r1" in st["edges"]  # ребро попало в граф UI
    assert graph_part(st, [], []) == ""


def test_event_duration_ms():
    import asyncio
    from harness import new_state, emit
    st = new_state("q")
    st["run_id"] = "r"
    asyncio.run(emit(st, "start", query="q"))
    assert st["events"][0]["duration_ms"] >= 0
    assert all("duration_ms" in e for e in st["events"])


def test_parallel_dispatch(monkeypatch):
    """C2: два независимых tool_call в одном ходу выполняются параллельно — одинаковый started_at."""
    import asyncio
    import harness
    from types import SimpleNamespace as NS

    def _ns(**kw):
        n = NS(**kw)
        n.model_dump = lambda: kw
        return n

    def _call(cid, name, args):
        return NS(id=cid, function=NS(name=name, arguments=args), model_dump=lambda: {"id": cid, "function": {"name": name, "arguments": args}})

    started = {}

    async def fake_find(st, query, k=5):
        started["find"] = time.monotonic()
        await asyncio.sleep(0)  # дать шанс другому инструменту стартовать
        return "сущность X"

    async def fake_query(st, kind, filter="", query="", count=False, k=10):
        started["db"] = time.monotonic()
        await asyncio.sleep(0)
        return "строки: 3"

    monkeypatch.setitem(harness.IMPL, "find_entities", fake_find)
    monkeypatch.setitem(harness.IMPL, "query_db", fake_query)
    async def _noop_resolve(ses, query):
        return query

    async def _noop_budget():
        return 32000

    async def _noop_seed(st):
        return ""

    monkeypatch.setattr(harness, "resolve", _noop_resolve)
    monkeypatch.setattr(harness, "budget", _noop_budget)
    monkeypatch.setattr(harness, "seed_hint", _noop_seed)
    monkeypatch.setattr(harness.core, "embed", lambda texts, *a, **k: [[0.0] * 1024 for _ in texts])

    calls = [_call("c1", "find_entities", '{"query": "X"}'),
             _call("c2", "query_db", '{"kind": "passage"}')]
    msg = NS(content="", tool_calls=calls, model_extra={})
    plain = NS(content="ответ", model_extra={}, tool_calls=[])

    async def fake_chat(msgs, usage=None, **kw):
        if not hasattr(fake_chat, "n"):
            fake_chat.n = 0
        fake_chat.n += 1
        return msg if fake_chat.n == 1 else plain

    async def _noop_verify(answer, sources):
        return []

    monkeypatch.setattr(harness.core, "chat", fake_chat)
    monkeypatch.setattr(harness.core, "verify", _noop_verify)

    st = harness.new_state("вопрос")
    st["run_id"] = "r"
    res = asyncio.run(harness.run("вопрос"))
    results = [e for e in res["events"] if e["type"] == "result"]
    started_at = {e["tool"]: e["started_at"] for e in results}
    assert len(started_at) == 2, started_at
    vals = list(started_at.values())
    assert vals[0] == vals[1]  # один batch_start
    assert "find" in started and "db" in started  # оба стартовали до конца другого
    assert all(e["duration_ms"] >= 0 for e in results)


def test_prefetch_ttl(monkeypatch):
    """C3: префетч одноразовый и живёт ровно cache_ttl."""
    import asyncio
    from config import CONFIG

    async def fake_retrieve(query, timings, top_n=10, min_score=None):
        return [], [], [{"id": "p1", "doc": "d", "text": "t", "score": 0.9}]
    monkeypatch.setattr(core, "retrieve", fake_retrieve)
    core._PREFETCH.clear()
    asyncio.run(core.prefetch("q"))
    assert core.prefetch_get("q") is not None
    assert core.prefetch_get("q") is None  # второе чтение — промах
    core._PREFETCH["q"] = (time.time() - CONFIG.harness.cache_ttl - 1, [], [], [])
    assert core.prefetch_get("q") is None  # TTL истёк


def test_speculative_search_hit(monkeypatch):
    """C3: попадание в префетч помечено speculative и не вызывает retrieve."""
    import asyncio
    import harness

    calls = []

    async def fake_retrieve(query, timings, top_n=10, min_score=None):
        calls.append(query)
        return [], [], [{"id": "p1", "doc": "d", "text": "t", "score": 0.9}]

    monkeypatch.setattr(core, "retrieve", fake_retrieve)
    core._PREFETCH.clear()
    asyncio.run(core.prefetch("q"))
    calls.clear()
    st = harness.new_state("q")
    asyncio.run(harness.search_passages(st, "q"))
    assert calls == []  # search_passages не ходил в поиск — взяли из префетча
    assert st["last_speculative"] is True and st["last_search_top"] == 0.9


def test_fast_path_streams_answer(monkeypatch):
    """C4+C5: уверенный реранкер — один стриминговый вызов, токены идут в ленту до answer_finished."""
    import asyncio
    import harness
    from types import SimpleNamespace as NS

    def _call(cid, name, args):
        return NS(id=cid, content="", function=NS(name=name, arguments=args),
                  model_dump=lambda: {"id": cid, "function": {"name": name, "arguments": args}})

    async def fake_search(st, query, k=3):
        st["last_search_top"] = 0.93
        st["pid"]["P1"], st["text"]["p1"], st["doc"]["p1"] = "p1", "текст один", "doc"
        st["pid"]["P2"], st["text"]["p2"], st["doc"]["p2"] = "p2", "текст два", "doc"
        return "пассаж про X"

    monkeypatch.setitem(harness.IMPL, "search_passages", fake_search)

    async def _noop_resolve(ses, query):
        return query

    async def _noop_budget():
        return 32000

    async def _noop_seed(st):
        return ""

    monkeypatch.setattr(harness, "resolve", _noop_resolve)
    monkeypatch.setattr(harness, "budget", _noop_budget)
    monkeypatch.setattr(harness, "seed_hint", _noop_seed)
    monkeypatch.setattr(harness.core, "embed", lambda texts, *a, **k: [[0.0] * 1024 for _ in texts])

    search = NS(content="", model_extra={}, tool_calls=[_call("s1", "search_passages", '{"query": "X"}')])
    chat_calls = [search]

    async def fake_chat(msgs, usage=None, **kw):
        return chat_calls.pop(0)

    chunks = ["Факт один [P1]\n", "Факт два [P2]"]

    async def fake_stream(msgs, on_token, usage=None, **kw):
        for ch in chunks:
            await on_token(ch)
        return "".join(chunks)

    async def _noop_verify(text, sources):
        return []

    async def _no_off(st, claims):
        return []

    monkeypatch.setattr(harness.core, "chat", fake_chat)
    monkeypatch.setattr(harness.core, "chat_stream", fake_stream)
    monkeypatch.setattr(harness.core, "verify", _noop_verify)
    monkeypatch.setattr(harness, "off_topic", _no_off)

    res = asyncio.run(harness.run("вопрос"))
    ev = res["events"]
    assert any(e["type"] == "fast_path_taken" for e in ev)
    tokens = [e for e in ev if e["type"] == "token"]
    assert len(tokens) >= 2
    finish_i = next(i for i, e in enumerate(ev) if e["type"] == "answer_finished")
    assert all(i < finish_i for i, e in enumerate(ev) if e["type"] == "token")
    assert set(ev[finish_i]["citations"]) == {"P1", "P2"}
    assert chat_calls == []  # один вызов: поиск; ответ пришёл стримом
    assert "Факт один [P1]" in res["answer"] and "Факт два [P2]" in res["answer"]


def test_chat_stream_emits_chunks(monkeypatch):
    """C5: провайдер стримит чанки в on_token и возвращает полный текст + usage."""
    import asyncio
    from types import SimpleNamespace as NS

    class Chunk:
        def __init__(self, text=None, usage=None):
            self.choices = [NS(delta=NS(content=text))] if text is not None else []
            self.usage = usage

    class Completions:
        async def create(self, **kw):
            async def gen():
                yield Chunk("раз")
                yield Chunk("два")
                yield Chunk(usage=NS(prompt_tokens=3, completion_tokens=2))
            return gen()

    monkeypatch.setattr(core, "llm", NS(chat=NS(completions=Completions())))
    got, usage = [], {}

    async def on_token(t):
        got.append(t)

    text = asyncio.run(core.chat_stream([{"role": "user", "content": "q"}], on_token, usage=usage))
    assert text == "раздва" and got == ["раз", "два"]
    assert usage == {"prompt": 3, "completion": 2}


def test_plain_text_answer_is_parsed_and_verified(monkeypatch):
    """Текст без answer-инструмента — тоже ответ: факты с [P#], markdown вычищается, статус answered."""
    import asyncio
    import harness
    from types import SimpleNamespace as NS

    def _call(cid, name, args):
        return NS(id=cid, content="", function=NS(name=name, arguments=args),
                  model_dump=lambda: {"id": cid, "function": {"name": name, "arguments": args}})

    async def fake_find(st, query, k=5):
        st["pid"]["P1"], st["text"]["p1"], st["doc"]["p1"] = "p1", "Иоганн Бернулли — учитель Эйлера.", "doc"
        st["pid"]["P2"], st["text"]["p2"], st["doc"]["p2"] = "p2", "Эйлер работал в Петербурге.", "doc"
        return "иоганн бернулли"

    monkeypatch.setitem(harness.IMPL, "find_entities", fake_find)

    async def _noop_resolve(ses, query):
        return query

    async def _noop_budget():
        return 32000

    async def _noop_seed(st):
        return ""

    monkeypatch.setattr(harness, "resolve", _noop_resolve)
    monkeypatch.setattr(harness, "budget", _noop_budget)
    monkeypatch.setattr(harness, "seed_hint", _noop_seed)

    text = ("1. **Прямая связь**: Иоганн Бернулли был учителем Леонарда Эйлера [P1].\n"
            "2. **Ещё факт**: Эйлер работал в Петербурге [P2].")
    calls = [NS(content="", model_extra={}, tool_calls=[_call("f1", "find_entities", '{"query": "x"}')]),
             NS(content=text, model_extra={}, tool_calls=[])]

    async def fake_chat(msgs, usage=None, **kw):
        return calls.pop(0)

    async def _noop_verify(text, sources):
        return []

    async def _no_off(st, claims):
        return []

    monkeypatch.setattr(harness.core, "chat", fake_chat)
    monkeypatch.setattr(harness.core, "verify", _noop_verify)
    monkeypatch.setattr(harness, "off_topic", _no_off)

    res = asyncio.run(harness.run("вопрос"))
    assert res["outcome"] == "answered"
    ev = res["events"]
    assert sum(e["type"] == "token" for e in ev) >= 2  # C5: текст и в стриме
    assert "**" not in res["answer"] and "[" not in res["answer"].replace("[P", "")
    assert "Иоганн Бернулли был учителем Леонарда Эйлера [P1]" in res["answer"]
    assert len(res["claims"]) == 2


def test_canonical_idempotent_basic():
    from graph import canonical
    names = {"якоб бернулли", "иоганн бернулли", "бернулли", "эйлер"}
    res = canonical(names)
    # идемпотентность: повторный прогон на результатах не меняет отображение
    assert canonical(set(res.values())) == {}
