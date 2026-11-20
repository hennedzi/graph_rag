from cli import iter_sse, truncate


def test_iter_sse_collects_multiline_data():
    lines = [b'data: {"type":"a"}\n', b'\n',
             b': keep-alive\n',
             b'data: {"type":"done",\n', b'data: "result": {"answer":"ok"}}\n', b'\n']
    assert list(iter_sse(lines)) == [{"type": "a"}, {"type": "done", "result": {"answer": "ok"}}]


def test_truncate():
    assert truncate("abc", 5) == "abc"
    assert truncate("abcdef", 5) == "abcde…"
    assert truncate("a\nb", 5) == "a b"