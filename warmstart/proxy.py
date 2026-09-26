"""A drop-in proxy for the OpenAI chat completions API with the cache in front of it.
Point an application's base URL here and it keeps working; misses go to the upstream.

The application tells the cache what it cannot see for itself, in three headers:
  X-Customer-Tier   the customer's tier, part of the key
  X-Customer-Id     who is asking
  X-Cache-Scope     "shared" when the answer is the same for everybody, "customer" when it
                    is built from this customer's data, "off" to skip the cache. With a
                    customer id and no scope, the safe default is "customer".
and optionally X-Cache-Tags (comma separated) to invalidate entries by later.

Only single questions are cached (a system prompt and one user message): a follow-up
depends on the conversation before it. Streamed responses are cached once they finish
normally. Unanimity among neighbours compares the answers' embeddings, since real
responses to paraphrases are worded differently. Not measured in this repository: the
benchmark runs the cache directly, and these routes are covered by unit tests.
"""
import json
import time
import uuid

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .cache import Key, SemanticCache, digest, ttl_for

SETTING = {"threshold": 0.96, "neighbours": 10, "band": 0.08}   # chosen by the benchmark for bge-small


def question_of(body):
    """(system prompt, question) for a single-question request, or None."""
    messages = body.get("messages") or []
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if len(rest) != 1 or rest[0].get("role") != "user":
        return None
    content = rest[0].get("content")
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "\n".join(str(m.get("content", "")) for m in system), str(content or "")


def create_app(upstream, *, embed=None, cache=None, api_key=None, transport=None, answer_floor=0.9):
    if embed is None:
        from .embed import embedder
        embed = embedder("bge-small")

    def same(a, b):
        return float(np.dot(a["answer_vector"], b["answer_vector"])) >= answer_floor

    cache = cache or SemanticCache(SETTING["threshold"], neighbours=SETTING["neighbours"], band=SETTING["band"], same=same)
    client = httpx.AsyncClient(base_url=upstream.rstrip("/"), transport=transport, timeout=120)
    counts = {"exact": 0, "semantic": 0, "miss": 0, "bypass": 0}
    saved_tokens = [0]
    app = FastAPI(title="warmstart")

    def key_for(body, headers, system):
        scope = headers.get("x-cache-scope") or ("customer" if headers.get("x-customer-id") else "shared")
        if scope == "customer" and not headers.get("x-customer-id"):
            return None, scope
        return Key.of(body.get("model", ""), body.get("temperature", 1.0), body.get("tools"), headers.get("x-customer-tier", "standard"),
                      system, f"customer:{headers['x-customer-id']}" if scope == "customer" else "shared"), scope

    def upstream_headers():
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def store(key, scope, question, vector, response, headers):
        content = response["choices"][0]["message"].get("content") or ""
        tags = {t.strip() for t in headers.get("x-cache-tags", "").split(",") if t.strip()}
        cache.store(key, question, vector, {"body": response, "answer_vector": embed.encode([content])[0]},
                    ttl_for(question, scope == "customer"), tags=tags)

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        parsed = question_of(body)
        key, scope = key_for(body, request.headers, parsed[0]) if parsed else (None, "off")
        if parsed is None or key is None or scope == "off":
            counts["bypass"] += 1
            return await forward(body, "bypass")
        question = parsed[1]
        vector = embed.encode([question])[0]
        found = cache.lookup(key, question, vector)
        if found.layer != "miss":
            counts[found.layer] += 1
            cached = found.entry.response["body"]
            saved_tokens[0] += (cached.get("usage") or {}).get("total_tokens", 0)
            headers = {"X-Cache": found.layer, "X-Cache-Similarity": f"{found.similarity:.4f}"}
            if body.get("stream"):
                return StreamingResponse(replay_stream(cached), media_type="text/event-stream", headers=headers)
            return JSONResponse({**cached, "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "created": int(time.time())}, headers=headers)
        counts["miss"] += 1
        if body.get("stream"):
            return StreamingResponse(stream_and_store(body, key, scope, question, vector, request.headers),
                                     media_type="text/event-stream", headers={"X-Cache": "miss"})
        reply = await client.post("/chat/completions", json=body, headers=upstream_headers())
        data = reply.json()
        if reply.status_code == 200 and data.get("choices") and data["choices"][0].get("finish_reason") == "stop":
            store(key, scope, question, vector, data, request.headers)
        return JSONResponse(data, status_code=reply.status_code, headers={"X-Cache": "miss"})

    async def forward(body, label):
        if body.get("stream"):
            async def passthrough():
                async with client.stream("POST", "/chat/completions", json=body, headers=upstream_headers()) as r:
                    async for line in r.aiter_lines():
                        if line:
                            yield line + "\n\n"
            return StreamingResponse(passthrough(), media_type="text/event-stream", headers={"X-Cache": label})
        reply = await client.post("/chat/completions", json=body, headers=upstream_headers())
        return JSONResponse(reply.json(), status_code=reply.status_code, headers={"X-Cache": label})

    async def stream_and_store(body, key, scope, question, vector, headers):
        """Pass the upstream stream through while keeping a copy; cache it only if it finished normally."""
        parts, finish, model = [], None, body.get("model", "")
        async with client.stream("POST", "/chat/completions", json=body, headers=upstream_headers()) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                yield line + "\n\n"
                payload = line.removeprefix("data:").strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                parts.append((choice.get("delta") or {}).get("content") or "")
                finish = choice.get("finish_reason") or finish
                model = chunk.get("model", model)
        if finish == "stop":
            store(key, scope, question, vector, {"object": "chat.completion", "model": model, "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "".join(parts)}, "finish_reason": "stop"}]}, headers)

    @app.post("/v1/cache/invalidate")
    async def invalidate(request: Request):
        body = await request.json()
        removed = cache.invalidate(tag=body.get("tag"), prompt=digest(body["system_prompt"]) if "system_prompt" in body else None,
                                   model=body.get("model"))
        return {"removed": removed}

    @app.get("/v1/cache/stats")
    async def stats():
        total = sum(counts.values())
        hits = counts["exact"] + counts["semantic"]
        return {"requests": counts, "hit_rate": round(hits / total, 4) if total else 0.0, "entries": cache.size(),
                "near_misses": list(cache.near_misses)[-20:], "tokens_saved": saved_tokens[0]}

    @app.get("/metrics")
    async def metrics():
        lines = ["# TYPE warmstart_requests_total counter"]
        lines += [f'warmstart_requests_total{{result="{k}"}} {v}' for k, v in counts.items()]
        lines += ["# TYPE warmstart_cache_entries gauge", f"warmstart_cache_entries {cache.size()}",
                  "# TYPE warmstart_tokens_saved_total counter", f"warmstart_tokens_saved_total {saved_tokens[0]}",
                  "# TYPE warmstart_near_misses_total counter", f"warmstart_near_misses_total {cache.stats['near_miss']}"]
        return PlainTextResponse("\n".join(lines) + "\n")

    return app


def replay_stream(body):
    """A cached answer, sent as one chunk in the streaming format."""
    content = body["choices"][0]["message"].get("content") or ""
    base = {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model", "")}
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': content}, 'finish_reason': None}]})}\n\n"
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"
