"""Unit tests. None downloads anything: the benchmark itself is `python -m warmstart gate`."""
import json
import re

import httpx
import numpy as np
from fastapi.testclient import TestClient

from warmstart import support
from warmstart.cache import DAY, HOUR, Key, SemanticCache, normalize, ttl_for
from warmstart.proxy import create_app
from warmstart.replay import Query, Setting, Shape, replay, summary, wilson

WORDS = "card arrive arrived when where will my is the how long take does delivery fee exchange rate pin change reset top up".split()


class Bag:
    """A stand-in embedder: counts of a fixed vocabulary, so shared words mean similar vectors."""

    def encode(self, texts, batch=None):
        rows = []
        for t in texts:
            tokens = re.findall(r"[a-z]+", t.lower())
            v = np.array([tokens.count(w) for w in WORDS], np.float32) + 1e-3
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)


def vec(text):
    return Bag().encode([text])[0]


KEY = Key.of("m", 0.2, [], "standard", "prompt v1", "shared")


# ------------------------------------------------------------------ the cache
def test_exact_layer_ignores_case_punctuation_and_spacing():
    c = SemanticCache(0.99, clock=lambda: 0)
    c.store(KEY, "Where is my card?", vec("where is my card"), "A", DAY)
    assert c.lookup(KEY, "  where IS my card ", vec("where is my card")).layer == "exact"
    assert normalize("Where's my card?!") == "where s my card"


def test_semantic_layer_serves_above_the_floor_and_logs_near_misses():
    c = SemanticCache(0.75, clock=lambda: 0, near_miss=0.2)
    c.store(KEY, "when will my card arrive", vec("when will my card arrive"), "A", DAY)
    hit = c.lookup(KEY, "when does my card arrive", vec("when does my card arrive"))
    assert hit.layer == "semantic" and hit.similarity >= 0.75
    miss = c.lookup(KEY, "how do I change my pin", vec("how do I change my pin"))
    assert miss.layer == "miss"


def test_the_key_separates_tier_prompt_version_and_customer():
    c = SemanticCache(0.5, clock=lambda: 0)
    c.store(KEY, "how long does delivery take", vec("how long does delivery take"), "A", DAY)
    for other in (Key.of("m", 0.2, [], "premium", "prompt v1", "shared"),
                  Key.of("m", 0.2, [], "standard", "prompt v2", "shared"),
                  Key.of("m", 0.2, [], "standard", "prompt v1", "customer:c1"),
                  Key.of("m", 0.9, [], "standard", "prompt v1", "shared"),
                  Key.of("m", 0.2, [{"name": "refund"}], "standard", "prompt v1", "shared")):
        assert c.lookup(other, "how long does delivery take", vec("how long does delivery take")).layer == "miss", other


def test_entries_expire_and_can_be_invalidated_by_tag_or_prompt():
    now = [0.0]
    c = SemanticCache(0.9, clock=lambda: now[0])
    c.store(KEY, "what is the exchange fee", vec("what is the exchange fee"), "A", HOUR, tags={"fees"})
    c.store(KEY, "how long does delivery take", vec("how long does delivery take"), "B", DAY)
    now[0] = 2 * HOUR
    assert c.lookup(KEY, "what is the exchange fee", vec("what is the exchange fee")).layer == "miss"
    assert c.invalidate(tag="fees") == 1
    assert c.invalidate(prompt=KEY.prompt) == 1
    assert c.lookup(KEY, "how long does delivery take", vec("how long does delivery take")).layer == "miss"


def test_no_semantic_hit_where_cached_neighbours_disagree():
    c = SemanticCache(0.75, clock=lambda: 0, neighbours=10, band=0.3)
    c.store(KEY, "when will my card arrive", vec("when will my card arrive"), {"answer_id": "arrival"}, DAY)
    c.store(KEY, "when does my card arrive", vec("when does my card arrive"), {"answer_id": "estimate"}, DAY)
    found = c.lookup(KEY, "when will the card arrive", vec("when will the card arrive"))
    assert found.layer == "miss" and found.disagreed
    lone = SemanticCache(0.75, clock=lambda: 0, neighbours=1)
    lone.store(KEY, "when will my card arrive", vec("when will my card arrive"), {"answer_id": "arrival"}, DAY)
    assert lone.lookup(KEY, "when will the card arrive", vec("when will the card arrive")).layer == "semantic"


def test_ttl_is_short_for_account_answers_and_questions_about_now():
    assert ttl_for("what is the exchange fee", personal=False) == DAY
    assert ttl_for("is my transfer still pending", personal=False) == HOUR
    assert ttl_for("what is the exchange fee", personal=True) == HOUR


# ------------------------------------------------------------------ the replay
def stream_of(pairs, customers=("c1", "c2")):
    return [Query(float(i), text, intent, customers[i % len(customers)], "standard") for i, (text, intent) in enumerate(pairs)]


def test_keying_on_the_question_alone_leaks_account_answers_and_the_full_key_does_not():
    pairs = [("where is my card", "card_arrival")] * 6          # card_arrival is answered from the customer's account
    stream = stream_of(pairs)
    vectors = Bag().encode([t for t, _ in pairs])
    full = summary(replay(stream, vectors, Setting(0.9, 1, 0.0), shape=Shape(queries=len(pairs))))
    naive = summary(replay(stream, vectors, Setting(0.9, 1, 0.0), mode="question", shape=Shape(queries=len(pairs))))
    assert full["leaks"] == 0 and naive["leaks"] > 0


def test_a_prompt_change_empties_the_cache():
    pairs = [("how long does delivery take", "card_delivery_estimate")] * 8
    rows = replay(stream_of(pairs), Bag().encode([t for t, _ in pairs]), Setting(0.9, 1, 0.0),
                  shape=Shape(queries=len(pairs), prompt_change=0.5, fee_change=0.99))
    assert [r["layer"] for r in rows] == ["miss", "exact", "exact", "exact", "miss", "exact", "exact", "exact"]
    assert not any(r["stale"] for r in rows)


def test_wilson_interval_brackets_the_rate():
    lo, hi = wilson(4, 616)
    assert lo < 4 / 616 < hi and hi < 0.02
    assert wilson(0, 0) == (0.0, 0.0)


def test_cost_model_counts_the_prefix_discount():
    full, cached = support.call_cost("how long does delivery take", False), support.call_cost("how long does delivery take", False, prefix_cache=True)
    assert cached < full and round((full - cached) * 1e6 / support.PRICE["input"]) == round(support.SYSTEM_TOKENS * (1 - support.PREFIX_READ))
    lat = support.latencies(20000, seed=1)
    assert abs(np.median(lat) - support.LATENCY["median"]) < 0.05 and abs(np.percentile(lat, 95) - support.LATENCY["p95"]) < 0.15


# ------------------------------------------------------------------ the proxy
def upstream(calls):
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        question = body["messages"][-1]["content"]
        if body.get("stream"):
            chunks = [{"choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}], "model": body["model"]}
                      for part in ("Cards arrive ", "in 5 days.")]
            chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "model": body["model"]})
            sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "model": body["model"],
                                         "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Answer to: {question}"},
                                                      "finish_reason": "stop"}], "usage": {"total_tokens": 42}})
    return httpx.MockTransport(handler)


def client_with(calls):
    return TestClient(create_app("http://upstream/v1", embed=Bag(), transport=upstream(calls)))


def ask(client, question, system="You are the support assistant.", headers=None, **extra):
    body = {"model": "m", "temperature": 0.2, "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=headers or {})


def test_proxy_serves_repeats_from_the_cache():
    calls = []
    c = client_with(calls)
    first = ask(c, "How long does delivery take?")
    again = ask(c, "how long does delivery take")
    assert first.headers["x-cache"] == "miss" and again.headers["x-cache"] == "exact"
    assert again.json()["choices"][0]["message"]["content"] == "Answer to: How long does delivery take?"
    assert len(calls) == 1
    assert c.get("/v1/cache/stats").json()["tokens_saved"] == 42
    assert 'warmstart_requests_total{result="exact"} 1' in c.get("/metrics").text


def test_proxy_keeps_customer_answers_to_that_customer():
    calls = []
    c = client_with(calls)
    ask(c, "where is my card", headers={"X-Customer-Id": "c1"})
    assert ask(c, "where is my card", headers={"X-Customer-Id": "c2"}).headers["x-cache"] == "miss"
    assert ask(c, "where is my card", headers={"X-Customer-Id": "c1"}).headers["x-cache"] == "exact"
    assert ask(c, "where is my card", headers={"X-Cache-Scope": "off"}).headers["x-cache"] == "bypass"


def test_proxy_misses_after_a_system_prompt_change_and_on_follow_ups():
    calls = []
    c = client_with(calls)
    ask(c, "what is the exchange fee")
    assert ask(c, "what is the exchange fee", system="A new prompt.").headers["x-cache"] == "miss"
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                                       {"role": "user", "content": "what is the exchange fee"}]}
    assert c.post("/v1/chat/completions", json=body).headers["x-cache"] == "bypass"
    assert c.post("/v1/cache/invalidate", json={"system_prompt": "You are the support assistant."}).json()["removed"] == 1


def test_proxy_caches_a_stream_once_it_finishes():
    calls = []
    c = client_with(calls)
    with c.stream("POST", "/v1/chat/completions", json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "when will my card arrive"}]}) as r:
        assert r.headers["x-cache"] == "miss"
        lines = [line for line in r.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"
    with c.stream("POST", "/v1/chat/completions", json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "when will my card arrive"}]}) as r:
        assert r.headers["x-cache"] == "exact"
        text = "".join(json.loads(line[5:])["choices"][0]["delta"].get("content", "") for line in r.iter_lines() if line.startswith("data: {"))
    assert text == "Cards arrive in 5 days." and len(calls) == 1
