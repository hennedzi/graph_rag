"""
Граф сущностей в памяти и алгоритмы поверх него (без networkx — только dict/list).

  G = (V, E): V — сущности (после склейки псевдонимов), E — отношения из Milvus (неориентированно,
  вес ребра = число отношений между парой).

  • canonical — склейка псевдонимов: «эйлер» → «леонард эйлер», если полное имя с такой фамилией одно.
  • ppr — Personalized PageRank степенным методом:  r⁽ᵗ⁺¹⁾ = (1−α)·Ãᵀ·r⁽ᵗ⁾ + α·v_q,
    v_q — распределение рестарта по seed-сущностям вопроса. Масса «висячих» вершин возвращается в v_q.
  • passage_scores — entity-weighted passage ranking: пассаж набирает массу своих сущностей,
    сущность делит свою массу поровну между пассажами (иначе хабы с сотнями пассажей забивают топ).
  • paths — простые пути между двумя сущностями (BFS, ≤ max_hops), короткие и через «узкие» вершины выше:
    путь через хаб («математика», «семья бернулли») почти ничего не объясняет.

Граф строится лениво при первом запросе и сбрасывается ETL после записи (invalidate).
"""

import math
from collections import defaultdict, deque

import core

ALPHA, ITERS, TOL = 0.15, 30, 1e-6  # α — вероятность рестарта; 30 итераций с запасом: ошибка ≤ (1−α)^t
MAX_EXPAND = 20000  # потолок раскрытых состояний BFS путей: на плотном графе число простых путей растёт экспоненциально


# ─── Чистые алгоритмы (тестируются без Milvus) ─────────────────
def canonical(names) -> dict[str, str]:
    """
    alias → полное имя: однословное имя, совпадающее с последним словом РОВНО одного многословного имени.
    Консервативно: «бернулли» (якоб, иоганн, даниил…) не склеивается ни с кем; «якоб бернулли» и
    «якоб ii бернулли» — тоже нет (это разные люди), поэтому подмножества слов в общем случае не берём.
    shortcut: только фамилия → полное имя; падежи и опечатки — кластеризация эмбеддингов (FUTURE.md, вариант C).
    """
    by_last = defaultdict(set)
    for n in names:
        if len(w := n.split()) > 1:
            by_last[w[-1]].add(n)
    return {n: next(iter(by_last[n])) for n in names if " " not in n and len(n) >= 4 and len(by_last.get(n, ())) == 1}


def ppr(adj: dict[str, dict[str, float]], seeds: dict[str, float], alpha: float = ALPHA,
        iters: int = ITERS, tol: float = TOL) -> dict[str, float]:
    """Personalized PageRank. Обходит только достижимые из seed вершины: на разреженном графе это O(iters·E_reach)."""
    total = sum(w for s, w in seeds.items() if s in adj) or 0
    if not total:
        return {}
    v = {s: w / total for s, w in seeds.items() if s in adj}
    deg = {}
    r = dict(v)
    for _ in range(iters):
        nxt = {s: alpha * w for s, w in v.items()}
        dangling = 0.0
        for u, ru in r.items():
            nb = adj[u]
            if not nb:
                dangling += ru
                continue
            d = deg.get(u) or deg.setdefault(u, sum(nb.values()))
            for w_, wt in nb.items():
                nxt[w_] = nxt.get(w_, 0.0) + (1 - alpha) * ru * wt / d
        for s, w in v.items():
            nxt[s] += (1 - alpha) * dangling * w
        diff = sum(abs(nxt.get(k, 0.0) - r.get(k, 0.0)) for k in nxt.keys() | r.keys())
        r = nxt
        if diff < tol:
            break
    return r


def passage_scores(rank: dict[str, float], ent_passages: dict[str, set]) -> dict[str, float]:
    out = defaultdict(float)
    for e, s in rank.items():
        ps = ent_passages.get(e) or ()
        for p in ps:
            out[p] += s / len(ps)
    return dict(out)


def paths(adj: dict[str, dict[str, float]], a: str, b: str, max_hops: int = 3, limit: int = 5) -> list[list[str]]:
    """Простые пути a → b длиной ≤ max_hops; ранжирование: меньше хопов, затем меньше Σ log(степени промежуточных)."""
    if a not in adj or b not in adj or a == b:
        return []
    found, queue, expanded = [], deque([[a]]), 0
    while queue and expanded < MAX_EXPAND:
        path = queue.popleft()
        expanded += 1
        for nb in adj[path[-1]]:
            if nb == b:
                found.append(path + [b])
            elif nb not in path and len(path) < max_hops:
                queue.append(path + [nb])
    return sorted(found, key=lambda p: (len(p), sum(math.log(1 + len(adj[x])) for x in p[1:-1])))[:limit]


def stats(adj: dict[str, dict[str, float]]) -> dict:
    """Метрики структуры графа: вершины, рёбра, средняя степень, компоненты связности, доля крупнейшей."""
    seen, sizes = set(), []
    for s in adj:
        if s in seen:
            continue
        seen.add(s)
        stack, n = [s], 0
        while stack:
            u = stack.pop()
            n += 1
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        sizes.append(n)
    V, E = len(adj), sum(len(nb) for nb in adj.values()) // 2
    return {"nodes": V, "edges": E, "avg_degree": round(2 * E / V, 2) if V else 0, "components": len(sizes),
            "largest_component": max(sizes, default=0)}


def build(entities: list[dict], relations: list[dict]) -> dict:
    """Строки Milvus → граф. entities: text, passage_ids; relations: id, text, src, dst, passage_ids."""
    alias = canonical({e["text"] for e in entities} | {x for r in relations for x in (r["src"], r["dst"])})
    c = lambda n: alias.get(n, n)  # noqa: E731
    adj, ent_p, rels, group = defaultdict(lambda: defaultdict(float)), defaultdict(set), defaultdict(list), defaultdict(set)
    for e in entities:
        adj[c(e["text"])]  # изолированная сущность — тоже вершина
        ent_p[c(e["text"])].update(e["passage_ids"])
        group[c(e["text"])].add(e["text"])
    for r in relations:
        s, d = c(r["src"]), c(r["dst"])
        group[s].add(r["src"])
        group[d].add(r["dst"])
        ent_p[s].update(r["passage_ids"])
        ent_p[d].update(r["passage_ids"])
        if s == d:  # склейка превратила связь в петлю — для обхода бесполезна
            continue
        adj[s][d] += 1
        adj[d][s] += 1
        rels[frozenset((s, d))].append(r)
    raw = defaultdict(set)
    for r in relations:
        if r["src"] != r["dst"]:
            raw[r["src"]].add(r["dst"])
            raw[r["dst"]].add(r["src"])
    for e in entities:
        raw[e["text"]]
    return {"adj": {k: dict(v) for k, v in adj.items()}, "ent_passages": dict(ent_p), "rels": dict(rels),
            "alias": alias, "group": {k: sorted(v) for k, v in group.items()},
            "stats": stats(adj) | {"aliases": len(alias)}, "stats_raw": stats(raw)}


# ─── Кэш поверх Milvus ─────────────────────────────────────────
_CACHE: dict = {}


def invalidate():
    _CACHE.clear()


def scan(kind: str, fields: list[str], where: str = "") -> list[dict]:
    """Все строки типа kind (итератором: query без limit упирается в 16384 строки)."""
    expr = f'type == "{kind}"' + (f" and ({where})" if where else "")
    it = core.milvus.query_iterator(core.COLLECTION, batch_size=1000, filter=expr, output_fields=fields)
    out = []
    while batch := it.next():
        out += batch
    it.close()
    return out


def get() -> dict:
    """shortcut: весь граф в памяти процесса (~200 байт на связь); для миллионов связей — CSR-массивы или граф-БД."""
    if "g" not in _CACHE:
        with core.timeit("graph_build", {}):
            _CACHE["g"] = build(scan("entity", ["text", "passage_ids"]),
                                scan("relation", ["text", "src", "dst", "passage_ids"]))
        core.log.info("Граф сущностей: %s", _CACHE["g"]["stats"])
    return _CACHE["g"]


def resolve(g: dict, name: str) -> str | None:
    n = core.normalize(name)
    n = g["alias"].get(n, n)
    return n if n in g["adj"] else None


async def resolve_any(g: dict, name: str) -> str | None:
    """Точное имя, иначе ближайшая сущность гибридным поиском («Эйлер» → «леонард эйлер»)."""
    if e := resolve(g, name):
        return e
    hits = core.hybrid(name, (await core.embed([name]))[0], "entity", 1)
    return resolve(g, hits[0]["text"]) if hits else None


def predicate(r: dict) -> str:
    t = r["text"]
    return t[len(r["src"]):len(t) - len(r["dst"])].strip() if t.startswith(r["src"]) and t.endswith(r["dst"]) else t


def relation_between(g: dict, a: str, b: str) -> dict:
    return g["rels"][frozenset((a, b))][0]
