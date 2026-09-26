"""Runs every measurement and writes results/bench.json and the tables in the README."""
import json
import time
from pathlib import Path

import numpy as np

from . import data, support
from .embed import embedder
from .replay import (Setting, Shape, around_change, choose, curve, embed_time, half, neighbours, precision, replay,
                     summary, traffic, wilson)

EMBEDDERS = ("bge-small", "minilm", "tfidf")
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def run(embedders=EMBEDDERS, shape=Shape()):
    t0 = time.time()
    train, test = data.questions("train"), data.questions("test")
    tune_cache = [x for x in train if half(x[0]) == 0]
    tune_queries = [x for x in train if half(x[0]) == 1]
    stream = traffic("test", seed=2, shape=shape)
    out = {"embedders": {}, "assumptions": {"price_per_million": support.PRICE, "system_tokens": support.SYSTEM_TOKENS,
                                            "prefix_read": support.PREFIX_READ, "llm_latency_s": support.LATENCY,
                                            "questions_per_month": shape.per_month}}
    for name in embedders:
        e = embedder(name)
        tuned = curve(neighbours(e, tune_cache, tune_queries))
        setting = choose(tuned)
        held_out = neighbours(e, train, test)
        cov, hits, wrong = precision(held_out, setting)
        bare = precision(held_out, Setting(setting.threshold, 1, 0.0))
        vectors = e.encode([q.text for q in stream])
        embed_s = embed_time(e, [q.text for q in stream])
        rows = replay(stream, vectors, setting)
        out["embedders"][name] = {
            "setting": setting.__dict__, "embed_ms": round(1000 * embed_s, 2),
            "held_out": {"coverage": round(cov, 4), "hits": hits, "wrong": wrong, "wrong_rate": round(wrong / max(1, hits), 4),
                         "wrong_ci": wilson(wrong, hits),
                         "without_agreement": {"coverage": round(bare[0], 4), "hits": bare[1], "wrong": bare[2],
                                               "wrong_rate": round(bare[2] / max(1, bare[1]), 4)}},
            "curve": [r for r in curve(held_out) if r["neighbours"] in (1, setting.neighbours) and r["band"] in (0.0, setting.band)],
            "replay": summary(rows, embed_s),
        }
        if name == embedders[0]:
            n = len(rows)
            out["best"] = name
            out["replay_rows"] = [{k: r[k] for k in ("i", "t", "layer", "similarity", "personal", "intent", "text", "served_for",
                                                     "served_intent", "wrong", "disagreed")} for r in rows]
            out["prompt_change"] = around_change(rows, int(n * shape.prompt_change))
            out["fee_change"] = around_change(rows, int(n * shape.fee_change))
            out["variants"] = {
                "exact layer only": summary(replay(stream, vectors, setting, semantic=False), embed_s),
                "similarity floor, no agreement check": summary(replay(stream, vectors, Setting(setting.threshold, 1, 0.0)), embed_s),
                "keyed on the question alone": summary(replay(stream, vectors, setting, mode="question"), embed_s),
            }
            out["wrong_examples"] = [{"question": r["text"], "intent": r["intent"], "served_for": r["served_for"],
                                      "served_intent": r["served_intent"], "similarity": r["similarity"]}
                                     for r in rows if r["wrong"]][:10]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def pct(x):
    return f"{100 * x:.1f}%"


def report(r):
    best = r["embedders"][r["best"]]
    s = best["replay"]
    lines = [f"Replay: {s['queries']:,} questions over {s['hours']} hours at {r['assumptions']['questions_per_month']:,} a month, "
             f"embedder {r['best']}, setting {best['setting']}.", "",
             "| | Hit rate | Wrong semantic hits | Leaks | Stale | $ per 1,000 questions | p50 | p95 |", "|---|---|---|---|---|---|---|---|"]
    none = s["latency_s"]["no cache"]
    lines.append(f"| no cache | 0% | - | - | - | {s['cost_per_1000']['no cache']:.2f} | {none['p50']:.2f} s | {none['p95']:.2f} s |")
    lines.append(f"| provider prefix cache only | 0% | - | - | - | {s['cost_per_1000']['provider prefix cache']:.2f} | {none['p50']:.2f} s | {none['p95']:.2f} s |")
    for label, v in [("exact layer only", r["variants"]["exact layer only"]), ("this cache", s),
                     ("similarity floor, no agreement check", r["variants"]["similarity floor, no agreement check"]),
                     ("keyed on the question alone", r["variants"]["keyed on the question alone"])]:
        lat = v["latency_s"]["cache"]
        lines.append(f"| {label} | {pct(v['hit_rate'])} | {v['wrong_semantic']}/{v['semantic_hits']} | {v['leaks']} | {v['stale']} | "
                     f"{v['cost_per_1000']['this cache + prefix cache']:.2f} | {lat['p50']:.2f} s | {lat['p95']:.2f} s |")
    lines += ["", "| Embedder | Setting | Held-out coverage | Wrong (held-out) | Without the agreement check | Replay hit rate | Embed ms |",
              "|---|---|---|---|---|---|---|"]
    for name, e in r["embedders"].items():
        h, st = e["held_out"], e["setting"]
        lines.append(f"| {name} | ≥ {st['threshold']}, {st['neighbours']} neighbours within {st['band']} | {pct(h['coverage'])} | "
                     f"{h['wrong']}/{h['hits']} ({pct(h['wrong_rate'])}) | {h['without_agreement']['wrong']}/{h['without_agreement']['hits']} "
                     f"({pct(h['without_agreement']['wrong_rate'])}) | {pct(e['replay']['hit_rate'])} | {e['embed_ms']} |")
    pc, fc = r["prompt_change"], r["fee_change"]
    lines += ["", f"Prompt change: hit rate {pct(pc['before'])} in the 300 questions before, {pct(pc['after'])} in the 300 after; "
              f"answers served from the old prompt afterwards: {pc['served_from_before']}.",
              f"Fee change: answers quoting the old fees served afterwards: {fc['served_from_before']}."]
    return "\n".join(lines)


def save(r, path=RESULTS / "bench.json"):
    """The results without the per-question rows, which go to replay_rows.json for the demo page."""
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in r.items() if k != "replay_rows"}, indent=1, default=float))
    if "replay_rows" in r:
        (path.parent / "replay_rows.json").write_text(json.dumps(r["replay_rows"], default=float))
