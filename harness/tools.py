"""
Инструменты harness-агента: find_entities, neighbors, search_passages, read_passage, rank_by_ppr, find_paths, query_db.
"""
import json
from config import CONFIG
import core
import graph
from core import normalize, timeit
from .state import handle, preview, check_filter, links, remember, rel_line, scope_where, passage_lines, graph_part, qvec, _entities, search_rows, new_state
from .prompts import TOOLS, PREVIEW

FIX_ROUNDS = CONFIG.harness.fix_rounds


async def find_entities(st, query: str, k: int = 5) -> str:
    hits = await _entities(query, min(k, 10))
    if not hits:
        return "ничего не найдено"
    return "\n".join(f"{h['text']} (пассажей: {len(h['passage_ids'])})" for h in hits) + \
        "\n→ neighbors(entity) — связи сущности с хэндлами [P#]; rank_by_ppr([..]) — шире по графу; find_paths(a, b) — как связаны две"


async def neighbors(st, entity: str, hops: int = 1) -> str:
    async def rels(ents: list[str], limit: int):
        lit = json.dumps(ents, ensure_ascii=False)
        return search_rows(await qvec(st), f'type == "relation" and (src in {lit} or dst in {lit})', limit,
                           ["text", "src", "dst", "passage_ids"])
    e, g = normalize(entity), graph.get()
    e = g["alias"].get(e, e)
    rows = await rels(g["group"].get(e, [e]), 20)
    if not rows:
        top = await _entities(entity, 1)
        e = top[0]["text"] if top else e
        rows = await rels([e], 20) if top else []
    if not rows:
        return f"у «{e}» нет связей"
    out = f"сущность: {e}\n" + "\n".join(rel_line(st, r) for r in rows)
    if hops >= 2:
        seen = {r["id"] for r in rows}
        ends = sorted({x for r in rows for x in (r["src"], r["dst"])} - {e})
        far = [r for r in await rels(ends, 15 + len(seen)) if r["id"] not in seen][:15]
        out += "\nчерез соседей:\n" + ("\n".join(rel_line(st, r) for r in far) or "нет")
    return out


async def query_db(st, kind: str, filter: str = "", query: str = "", count: bool = False, k: int = 10) -> str:
    if kind not in ("passage", "entity", "relation"):
        raise ValueError("kind: passage | entity | relation")
    where = check_filter(st, filter)
    if kind == "passage" and (sw := scope_where(st)):
        where = sw + (f" and ({where})" if where else "")
    expr = f'type == "{kind}"' + (f" and ({where})" if where else "")
    if count:
        return f"строк: {core.milvus.query(core.COLLECTION, filter=expr, output_fields=['count(*)'])[0]['count(*)']}"
    k = max(1, min(int(k), 20))
    rows = core.hybrid(query, (await core.embed([query]))[0], kind, k, where) if query else \
        list(core.milvus.query(core.COLLECTION, filter=expr, output_fields=core.FIELDS, limit=k))
    if not rows:
        return "ничего не найдено"
    if kind == "passage":
        return passage_lines(st, rows)
    if kind == "entity":
        return "\n".join(f"{r['text']} (пассажей: {len(r['passage_ids'])})" for r in rows)
    return "\n".join(rel_line(st, r) for r in rows)


async def search_passages(st, query: str, k: int = 3) -> str:
    hit = core.prefetch_get(query)
    if hit:
        ents, rels, ps = hit
        st["last_speculative"] = True
    else:
        ents, rels, ps = await core.retrieve(query, st["timings"], top_n=min(k, 10), min_score=float("-inf"))
        st["last_speculative"] = False
    if st.get("scope"):
        ps = [p for p in ps if p["doc"] == st["scope"]]
    st["last_search_top"] = max((p.get("score", 0.0) for p in ps), default=0.0)
    head = graph_part(st, ents, rels)
    if st.get("scope"):
        head = f"(область: пассажи только из {st['scope']}; граф и сущности — по всей базе)\n" + head
    strong = [p for p in ps if p["score"] >= core.MIN_RERANK_SCORE]
    if strong:
        body = passage_lines(st, strong)
    elif not ps:
        body = "пассажей не найдено"
    else:
        body = f"выше порога {core.MIN_RERANK_SCORE} нет; слабые совпадения — проверь превью:\n" + passage_lines(st, ps[:5])
    return head + body + f"\n(превью до {PREVIEW} симв.; полный текст — read_passage)"


async def seed_hint(st) -> str:
    hits = core.hybrid(st["query"], await qvec(st), "entity", 3)
    return " · ".join(f"{h['text']} ({len(h['passage_ids'])} пасс.)" for h in hits)


async def read_passage(st, ids: list[str]) -> str:
    asked = list(dict.fromkeys(h.strip("[] ").upper() for h in ids))
    hs, rest = asked[:10], asked[10:]
    unknown = [h for h in hs if h not in st["pid"]]
    missing = [st["pid"][h] for h in hs if h in st["pid"] and st["pid"][h] not in st["text"]]
    for r in core.get_rows(missing, ["text", "doc"]):
        st["text"][r["id"]], st["doc"][r["id"]] = r["text"], r["doc"]
    out = []
    for h in hs:
        if h in unknown:
            out.append(f"[{h}] неизвестный хэндл: это не P# из выдачи; для имени сущности вызови neighbors")
        elif h in st["read"]:
            out.append(f"[{h}] уже прочитан выше")
        else:
            st["read"].add(h)
            out.append(f"[{h}] {st['text'].get(st['pid'][h], '(не найден)')}")
    if rest:
        out.append(f"не прочитаны (не больше 10 за вызов): {' '.join(rest)} — запроси, только если они нужны для ответа")
    return "\n\n".join(out)


async def _seeds(names: list[str]) -> tuple[dict, list[str]]:
    g = graph.get()
    found = [await graph.resolve_any(g, n) for n in names[:5]]
    return g, list(dict.fromkeys(e for e in found if e))


async def rank_by_ppr(st, entities: list[str], k: int = 8) -> str:
    g, seeds = await _seeds([entities] if isinstance(entities, str) else entities)
    if not seeds:
        return "ни одной из сущностей нет в графе — сначала find_entities"
    k = max(1, min(int(k), 10))
    r = graph.ppr(g["adj"], dict.fromkeys(seeds, 1.0))
    top = sorted((e for e in r if e not in seeds), key=r.get, reverse=True)[:k]
    if not top:
        return f"seed: {', '.join(seeds)}\nу seed нет связей в графе"
    ps = sorted(graph.passage_scores(r, g["ent_passages"]).items(), key=lambda x: -x[1])[:5]
    texts = {p["id"]: p for p in core.get_rows([p for p, _ in ps], ["text", "doc"])}
    rows = [texts[p] | {"score": s * 100} for p, s in ps if p in texts]
    near = [graph.relation_between(g, s, e) for e in top for s in seeds if frozenset((s, e)) in g["rels"]]
    out = (f"seed: {', '.join(seeds)}\nсущности по PPR (% массы): "
           + " · ".join(f"{e} {r[e] * 100:.1f}" for e in top)
           + "\nпассажи по массе сущностей (%):\n" + passage_lines(st, rows))
    for e in top:
        for x in top + seeds:
            if x != e and frozenset((x, e)) in g["rels"]:
                remember(st, graph.relation_between(g, x, e))
    return out + ("\nсвязи seed с топом:\n" + "\n".join(rel_line(st, x) for x in near[:8]) if near else "")


def path_line(st, g: dict, path: list[str]) -> str:
    out, hs = path[0], []
    for a, b in zip(path, path[1:]):
        r = graph.relation_between(g, a, b)
        p = graph.predicate(r)
        fwd = g["alias"].get(r["src"], r["src"]) == a
        out += f" —{p}→ {b}" if fwd else f" ←{p}— {b}"
        remember(st, r)
        hs += [handle(st, x) for x in r["passage_ids"][:2]]
    return out + "  " + " ".join(f"[{h}]" for h in dict.fromkeys(hs))


async def find_paths(st, source: str, target: str, max_hops: int = 3) -> str:
    g, ends = await _seeds([source, target])
    if len(ends) < 2:
        return "одна из сущностей не найдена в графе (или это одна и та же сущность) — уточни через find_entities"
    hops = max(1, min(int(max_hops), 4))
    found = graph.paths(g["adj"], ends[0], ends[1], hops)
    if not found:
        return f"путей «{ends[0]}» → «{ends[1]}» до {hops} хопов нет; попробуй rank_by_ppr или search_passages"
    return f"пути {ends[0]} → {ends[1]}:\n" + "\n".join(path_line(st, g, p) for p in found)


async def list_documents(st) -> str:
    rows = core.milvus.query(core.COLLECTION, filter='type == "passage"', output_fields=["doc"], limit=1000)
    docs = sorted({r.get("doc") for r in rows if r.get("doc")})
    if not docs:
        return "документов нет"
    return "документы:\n" + "\n".join(f"- {d}" for d in docs)

IMPL = {"find_entities": find_entities, "neighbors": neighbors, "rank_by_ppr": rank_by_ppr, "find_paths": find_paths, "list_documents": list_documents,
        "query_db": query_db,
        "search_passages": search_passages, "read_passage": read_passage}