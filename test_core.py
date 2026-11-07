"""Самопроверка чистой логики без моделей: uv run pytest -q test_core.py"""
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
    for i in range(1, 8):
        st["pid"][f"P{i}"] = f"p:{i}"
        st["text"][f"p:{i}"] = f"текст {i}"
    out = asyncio.run(read_passage(st, ["P1", "p1", "[P2]", "P3", "P4", "P5", "P6", "P7"]))
    assert "[P5] текст 5" in out and "текст 6" not in out and "не прочитаны (не больше 5 за вызов): P6 P7" in out
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
    text, claims, errors = render(st, {"claims": [{"text": "**Эйлер** учился в Базеле", "sources": ["[P1]", "p2", "P1"]}]})
    assert (text, errors) == ("Эйлер учился в Базеле [P1] [P2]", []) and claims[0]["sources"] == ["P1", "P2"]
    assert render(st, {"claims": [{"text": "факт", "sources": []}]})[2] == ["нет sources: «факт»"]
    assert "неизвестные хэндлы P9" in render(st, {"claims": [{"text": "факт", "sources": ["P9"]}]})[2][0]
    assert render(st, {"claims": [], "no_answer": True})[0].startswith("В предоставленном контексте нет")
    assert render(st, {"text": "старый формат [P1]"}) == ("старый формат [P1]", [], [])
    assert render(st, {"claims": []})[2]


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

    async def put(ev):
        got.append(ev)
    st["emit"] = put
    asyncio.run(emit(st, "call", tool="neighbors"))
    assert got == [{"type": "call", "tool": "neighbors"}]


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
    st = new_state("Я студент")
    assert render(st, {"reply": "Хорошо, спрашивайте."}) == ("Хорошо, спрашивайте.", [], [])
    assert render(st, {"reply": "Эйлер родился в Базеле [P1]"})[2]


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
