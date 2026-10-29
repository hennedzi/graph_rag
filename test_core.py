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
