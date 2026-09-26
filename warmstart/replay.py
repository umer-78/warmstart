"""The measurements: how precise the semantic layer is at each setting, and what the whole
cache does to production-shaped traffic.

Precision first, on every Banking77 question at once. A cache holding every training
question answers each held-out test question from its nearest neighbour; a hit is wrong
when that neighbour was labelled with a different intent. The setting is chosen on the
training questions alone (one half as the cache, the other as the traffic) and then
measured on the test questions.

Then the replay: 10,000 test questions arriving at 400,000 a month (about 19 hours), with
intents on a Zipf curve so the ten most common take about six in ten
questions, as at a real support desk. 500 customers on three tiers send them. Halfway
through, the system prompt changes; three quarters in, the fee schedule changes. A served
answer is wrong when it was written for another intent, a leak when it carries another
customer's account data, and stale when it came from an older prompt or fee schedule.
"""
import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np

from . import data, support
from .cache import DAY, Key, SemanticCache, ttl_for

MODEL = "frontier-model"
TOOLS = [{"name": "lookup_account"}, {"name": "open_ticket"}]
PROMPTS = {"v1": "You are the support assistant for Northwind Bank. Be brief and exact.",
           "v2": "You are the support assistant for Northwind Bank. Be brief, exact and warm, and offer a next step."}
TIERS = (("standard", 0.70), ("plus", 0.25), ("premium", 0.05))
TARGET = 0.005        # tune for at most 0.5% wrong hits: half the ship gate's 1%, for the gap between tuning and test
MIN_HITS = 300        # and only where there are enough hits for that rate to mean something
THRESHOLDS = np.round(np.arange(0.80, 0.9951, 0.01), 2)
NEIGHBOURS = (1, 3, 10)      # 1 means no agreement check
BANDS = (0.02, 0.04, 0.08)
PER_MONTH = 400_000


@dataclass(frozen=True)
class Setting:
    threshold: float
    neighbours: int = 10
    band: float = 0.08


# ------------------------------------------------------------------ precision
def half(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 2


def neighbours(embedder, cache, queries, k=10):
    """Each query's k most similar cached questions (identical wordings excluded, so the
    numbers are about paraphrases), with their similarities and intents."""
    c = embedder.encode([t for t, _ in cache])
    q = embedder.encode([t for t, _ in queries])
    sims = q @ c.T
    index = {}
    for j, (t, _) in enumerate(cache):
        index.setdefault(t, []).append(j)
    for i, (t, _) in enumerate(queries):
        sims[i, index.get(t, [])] = -1.0
    order = np.argsort(-sims, axis=1)[:, :k]
    return {"sims": np.take_along_axis(sims, order, 1), "intents": np.array([i for _, i in cache])[order],
            "truth": np.array([i for _, i in queries])}


def precision(nb, setting):
    """(coverage, hits, wrong hits) for one setting over precomputed neighbours."""
    sims, intents, truth = nb["sims"], nb["intents"], nb["truth"]
    best = sims[:, 0]
    agree = np.ones(len(best), bool)
    for j in range(1, setting.neighbours):
        close = sims[:, j] >= best - setting.band
        agree &= ~close | (intents[:, j] == intents[:, 0])
    hit = (best >= setting.threshold) & agree
    wrong = hit & (intents[:, 0] != truth)
    return float(hit.mean()), int(hit.sum()), int(wrong.sum())


def curve(nb, neighbours_=NEIGHBOURS, bands=BANDS, thresholds=THRESHOLDS):
    rows = []
    for k in neighbours_:
        for band in (bands if k > 1 else (0.0,)):
            for t in thresholds:
                cov, n, w = precision(nb, Setting(float(t), k, band))
                rows.append({"threshold": float(t), "neighbours": k, "band": band, "coverage": round(cov, 4),
                             "hits": n, "wrong": w, "wrong_rate": round(w / n, 4) if n else 0.0})
    return rows


def choose(rows, target=TARGET, min_hits=MIN_HITS):
    """The setting that answers the most questions while keeping wrong hits under target."""
    ok = [r for r in rows if r["hits"] >= min_hits and r["wrong_rate"] <= target]
    best = max(ok, key=lambda r: (r["coverage"], r["threshold"])) if ok else max(rows, key=lambda r: r["threshold"])
    return Setting(best["threshold"], best["neighbours"], best["band"])


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4))


# ------------------------------------------------------------------ the replay
@dataclass(frozen=True)
class Query:
    t: float
    text: str
    intent: str
    customer: str
    tier: str


@dataclass(frozen=True)
class Shape:
    queries: int = 10_000
    customers: int = 500
    per_month: int = PER_MONTH
    zipf: float = 1.1
    prompt_change: float = 0.5
    fee_change: float = 0.75


def traffic(split, seed, shape=Shape()):
    rng = np.random.default_rng(seed)
    by_intent = {}
    for text, intent in data.questions(split):
        by_intent.setdefault(intent, []).append(text)
    order = [sorted(by_intent)[i] for i in rng.permutation(len(by_intent))]
    weights = 1 / np.arange(1, len(order) + 1) ** shape.zipf
    weights /= weights.sum()
    names, shares = zip(*TIERS)
    tier = rng.choice(names, size=shape.customers, p=shares)
    span = shape.queries / (shape.per_month / 30) * DAY
    out = []
    for t in np.sort(rng.uniform(0, span, shape.queries)):
        intent = order[rng.choice(len(order), p=weights)]
        texts = by_intent[intent]
        c = int(rng.integers(shape.customers))
        out.append(Query(float(t), texts[int(rng.integers(len(texts)))], intent, f"c{c:04d}", str(tier[c])))
    return out


def replay(stream, vectors, setting, *, mode="full", semantic=True, seed=0, shape=Shape()):
    """Run the stream through a fresh cache. mode "full" keys on everything that changes the
    answer; "question" keys on the question alone, the trap. One row per query."""
    clock = [0.0]
    cache = SemanticCache(setting.threshold if semantic else math.inf, clock=lambda: clock[0],
                          neighbours=setting.neighbours if mode == "full" else 1, band=setting.band)
    llm = support.latencies(len(stream), seed)
    prompt, policy, rows = "v1", "p1", []
    n = len(stream)
    for i, (q, v) in enumerate(zip(stream, vectors)):
        clock[0] = q.t
        if i == int(n * shape.prompt_change):
            prompt = "v2"
        if i == int(n * shape.fee_change):
            policy = "p2"
            if mode == "full":
                cache.invalidate(tag="fees")
        personal = q.intent in data.PERSONAL
        if mode == "full":
            key = Key.of(MODEL, 0.2, TOOLS, q.tier, PROMPTS[prompt], "shared")
        else:
            key = Key.of(MODEL)
        started = time.perf_counter()
        found = cache.lookup(key, q.text, v)
        lookup_s = time.perf_counter() - started
        hit = found.layer != "miss"
        if hit:
            response = found.entry.response
        else:
            response = support.answer(q.intent, q.customer, prompt, policy, personal)
            cache.store(key, q.text, v, response, ttl_for(q.text, personal), tags={"fees"} if q.intent in data.FEES else ())
        near = not hit and found.nearest is not None and found.similarity >= setting.threshold - cache.near_miss
        rows.append({
            "i": i, "t": q.t, "layer": found.layer, "similarity": round(found.similarity, 4), "personal": personal,
            "intent": q.intent, "text": q.text, "served_for": found.entry.text if hit else None,
            "served_intent": response["intent"] if hit else None,
            "wrong": hit and response["intent"] != q.intent,
            "leak": hit and response["customer"] not in (None, q.customer),
            "stale": hit and (response["prompt"] != prompt or (q.intent in data.FEES and response["policy"] != policy)),
            "disagreed": found.disagreed, "near": near,
            "near_right": near and found.nearest.response["intent"] == q.intent,
            "llm_s": float(llm[i]), "lookup_s": lookup_s,
            "cost": 0.0 if hit else support.call_cost(q.text, personal),
            "cost_prefix": 0.0 if hit else support.call_cost(q.text, personal, prefix_cache=True),
        })
    return rows


def summary(rows, embed_s=0.0):
    n = len(rows)
    semantic = [r for r in rows if r["layer"] == "semantic"]
    hits = [r for r in rows if r["layer"] != "miss"]
    wrong = sum(r["wrong"] for r in semantic)
    no_cache = np.array([r["llm_s"] for r in rows])
    cached = np.array([embed_s + r["lookup_s"] + (r["llm_s"] if r["layer"] == "miss" else 0.0) for r in rows])
    full = sum(support.call_cost(r["text"], r["personal"]) for r in rows)
    prefix = sum(support.call_cost(r["text"], r["personal"], prefix_cache=True) for r in rows)
    near = [r for r in rows if r["near"]]
    pct = lambda a, k: round(float(np.percentile(a, k)), 3)
    return {
        "queries": n, "hours": round((rows[-1]["t"] - rows[0]["t"]) / 3600, 1),
        "hit_rate": round(len(hits) / n, 4), "exact_rate": round(sum(r["layer"] == "exact" for r in rows) / n, 4),
        "semantic_rate": round(len(semantic) / n, 4), "semantic_hits": len(semantic), "wrong_semantic": wrong,
        "wrong_rate": round(wrong / len(semantic), 4) if semantic else 0.0, "wrong_ci": wilson(wrong, len(semantic)),
        "wrong_exact": sum(r["wrong"] for r in rows if r["layer"] == "exact"),
        "leaks": sum(r["leak"] for r in rows), "stale": sum(r["stale"] for r in rows),
        "disagreed": sum(r["disagreed"] for r in rows), "near_misses": len(near), "near_misses_right": sum(r["near_right"] for r in near),
        "personal_share": round(sum(r["personal"] for r in rows) / n, 4),
        "general_hit_rate": round(sum(r["layer"] != "miss" for r in rows if not r["personal"]) / max(1, sum(not r["personal"] for r in rows)), 4),
        "cost_per_1000": {"no cache": round(1000 * full / n, 2), "provider prefix cache": round(1000 * prefix / n, 2),
                          "this cache": round(1000 * sum(r["cost"] for r in rows) / n, 2),
                          "this cache + prefix cache": round(1000 * sum(r["cost_prefix"] for r in rows) / n, 2)},
        "latency_s": {"no cache": {"p50": pct(no_cache, 50), "p95": pct(no_cache, 95)},
                      "cache": {"p50": pct(cached, 50), "p95": pct(cached, 95)},
                      "cache hit": {"p50": pct(cached[[r["layer"] != "miss" for r in rows]], 50) if hits else 0.0}},
    }


def around_change(rows, at, width=300):
    """Hit rate just before and just after a change, to show it emptied the cache."""
    before = rows[max(0, at - width):at]
    after = rows[at:at + width]
    rate = lambda rs: round(sum(r["layer"] != "miss" for r in rs) / max(1, len(rs)), 4)
    return {"before": rate(before), "after": rate(after), "served_from_before": sum(r["stale"] for r in rows[at:])}


def embed_time(embedder, texts, repeats=200):
    """Median seconds to embed one question on its own, the way the cache sees them."""
    times = []
    for k in range(repeats):
        started = time.perf_counter()
        embedder.encode([texts[k % len(texts)]])
        times.append(time.perf_counter() - started)
    return float(np.median(times))
