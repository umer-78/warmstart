"""Builds the dashboard's data (docs/data.json) from results/bench.json and results/replay_rows.json."""
import json

from .bench import RESULTS, ROOT

WINDOW = 250


def build(out=ROOT / "docs"):
    r = json.loads((RESULTS / "bench.json").read_text())
    rows = json.loads((RESULTS / "replay_rows.json").read_text())
    series = []
    for start in range(0, len(rows), WINDOW):
        chunk = rows[start:start + WINDOW]
        series.append({"hour": round(chunk[-1]["t"] / 3600, 2), "at": start + len(chunk),
                       "hit": round(sum(x["layer"] != "miss" for x in chunk) / len(chunk), 4),
                       "exact": round(sum(x["layer"] == "exact" for x in chunk) / len(chunk), 4),
                       "semantic": round(sum(x["layer"] == "semantic" for x in chunk) / len(chunk), 4)})
    seen, pairs = set(), []
    for x in rows:
        if x["layer"] == "semantic" and not x["wrong"] and (x["text"], x["served_for"]) not in seen:
            seen.add((x["text"], x["served_for"]))
            pairs.append({"question": x["text"], "served_for": x["served_for"], "similarity": x["similarity"]})
    pairs.sort(key=lambda p: p["similarity"])
    step = max(1, len(pairs) // 12)
    curves = {name: e["curve"] for name, e in r["embedders"].items()}
    payload = {"summary": {k: r[k] for k in ("best", "prompt_change", "fee_change", "variants", "wrong_examples", "assumptions")},
               "embedders": {name: {k: e[k] for k in ("setting", "embed_ms", "held_out", "replay")} for name, e in r["embedders"].items()},
               "curves": curves, "series": series, "examples": pairs[::step][:12], "queries": len(rows)}
    out.mkdir(exist_ok=True)
    (out / "data.json").write_text(json.dumps(payload, indent=1))
    print(f"wrote docs/data.json: {len(series)} windows, {len(payload['examples'])} examples")
    return payload
