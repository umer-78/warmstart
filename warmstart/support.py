"""The support model the cache sits in front of, for the replay: what a call costs and how
long it takes. These are assumptions, stated here and printed with every result, because
the replay has no API key: a frontier model at $3 and $15 per million input and output
tokens, a 1,200-token static system block, and a median response time of 1.6 s with a
95th percentile of 4 s. The hit rates and the false-hit rates are measured; the dollars
and seconds are those rates multiplied by these numbers.
"""
import math

import numpy as np

PRICE = {"input": 3.00, "output": 15.00}   # $ per million tokens
SYSTEM_TOKENS = 1200                       # policies, tone and tool definitions, the same on every call
PREFIX_READ = 0.10                         # a provider prefix-cache read bills the system block at 10%
LOOKUP_TOKENS = 150                        # account data fetched into the prompt for a personal question
ANSWER_TOKENS = {"general": 170, "personal": 220}
LATENCY = {"median": 1.6, "p95": 4.0}      # seconds


def tokens(text):
    return max(1, round(len(text.split()) * 1.3))


def call_cost(question, personal, prefix_cache=False):
    system = SYSTEM_TOKENS * (PREFIX_READ if prefix_cache else 1.0)
    prompt = system + tokens(question) + (LOOKUP_TOKENS if personal else 0)
    answer = ANSWER_TOKENS["personal" if personal else "general"]
    return (prompt * PRICE["input"] + answer * PRICE["output"]) / 1e6


def latencies(n, seed):
    """Response times for n calls, lognormal with the stated median and 95th percentile."""
    sigma = math.log(LATENCY["p95"] / LATENCY["median"]) / 1.6449
    return np.random.default_rng(seed).lognormal(math.log(LATENCY["median"]), sigma, n)


def answer(intent, customer, prompt_version, policy_version, personal):
    """The response a call returns. A personal answer carries the customer's data; every
    answer records which prompt and policy produced it, so a stale one can be caught."""
    body = f"[{intent}] answer under prompt {prompt_version}, policy {policy_version}"
    if personal:
        body += f", from the account of {customer}"
    return {"answer_id": f"{intent}/{customer if personal else 'all'}/{prompt_version}/{policy_version}",
            "intent": intent, "customer": customer if personal else None, "prompt": prompt_version,
            "policy": policy_version, "text": body}
