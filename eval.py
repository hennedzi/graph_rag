"""
Eval-раннер harness: вопросы из jsonl → harness.run → метрики в runs/eval/*.json и .csv (при остановленном сервере).

  uv run eval.py                                   # все инструменты
  uv run eval.py --without rank_by_ppr,find_paths  # базовая линия: только neighbors (для сравнения с PPR)
  uv run eval.py --only multi eval/questions.jsonl

Строка jsonl: {"q": вопрос, "type": "single"|"multi", "expect": [подстроки, все должны быть в ответе]}.
expect=[] — ожидается no_answer. hit — грубая автопроверка; фактологию стоит просмотреть глазами в .json.
"""

import argparse
import asyncio
import csv
import json
import subprocess
from datetime import datetime
from statistics import mean

import core
import harness


def hit(answer: str, expect: list[str]) -> bool:
    a = core.normalize(answer)
    return a.startswith(core.normalize(core.NO_CONTEXT)[:30]) if not expect else all(core.normalize(e) in a for e in expect)


def summary(rows: list[dict]) -> dict:
    keys = ("hit", "attributed", "steps", "context_tokens", "tokens_spent", "unsupported", "total_ms")
    return {k: round(mean(float(r[k]) for r in rows), 3) for k in keys} | {"n": len(rows)} if rows else {"n": 0}


async def main(path: str, without: set[str], only: str | None):
    harness.TOOLS = [t for t in harness.TOOLS if t["function"]["name"] not in without]
    for name in without:
        harness.IMPL.pop(name, None)
    qs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    qs = [q for q in qs if not only or q.get("type") == only]
    rows = []
    for i, q in enumerate(qs, 1):
        r = await harness.run(q["q"])  # каждый вопрос — новая сессия: замеры не зависят от порядка
        rows.append({"q": q["q"], "type": q.get("type", ""), "hit": hit(r["answer"], q.get("expect", [])),
                     "attributed": bool(r["claims"]) and all(c["sources"] for c in r["claims"]) or not q.get("expect"),
                     "steps": r["steps"], "context_tokens": r["context_tokens"], "tokens_spent": r["tokens_spent"],
                     "unsupported": len(r["unsupported"]), "total_ms": r["timings_ms"].get("total", 0),
                     "tools": " ".join(t["tool"] for t in r["trace"]), "answer": r["answer"]})
        core.log.info("[%d/%d] %s → hit=%s шагов=%s", i, len(qs), q["q"], rows[-1]["hit"], r["steps"])
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"commit": commit, "without": sorted(without), "questions": path, "model": core.LLM_MODEL,
           "summary": {t: summary([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})}
           | {"all": summary(rows)}, "rows": rows}
    out = core.RUNS_DIR / "eval" / datetime.now().strftime("%Y%m%d_%H%M%S")  # подпапка: /runs её не листает
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["q"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2), f"\n→ {out}.json / .csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="eval/questions.jsonl")
    p.add_argument("--without", default="", help="инструменты через запятую, которые убрать у агента")
    p.add_argument("--only", choices=["single", "multi"])
    a = p.parse_args()
    asyncio.run(main(a.path, {x.strip() for x in a.without.split(",") if x.strip()}, a.only))
