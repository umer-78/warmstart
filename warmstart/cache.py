"""The cache: an exact match first, then a semantic match above a similarity floor, both
inside a partition that the key decides.

A similarity floor alone was not enough: on Banking77, a question's nearest neighbour at
0.98 similarity still wanted a different answer about 2% of the time, because paraphrases
sit on the boundary between two answers ("When will I get my card?" and "How soon will I
get my card?" are labelled as different intents). So a semantic hit also has to be
unanimous: every cached question within `band` of the best match must have been given the
same answer. Where they disagree, the question sits on a boundary and goes to the model.
`support` can also ask for more than one cached question above the floor before serving;
in the replay, two witnesses removed 72% of semantic hits for a small gain in precision,
so it is off by default.

The key is everything that changes the right answer besides the question: the model, the
temperature, the tools the model may call, the customer's tier, the version of the system
prompt, and the scope. A question about the customer's own account is scoped to that
customer, so it is never answered from someone else's entry. The application says which
requests carry customer data; the cache does not guess.
"""
import hashlib
import json
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

HOUR, DAY = 3600, 86400
TIME_SENSITIVE = re.compile(r"\b(today|tonight|now|right now|currently|yesterday|this morning|pending|still)\b", re.I)


def normalize(text):
    """What the exact layer compares: case, punctuation and spacing do not matter."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Key:
    model: str
    temperature: float
    tools: str            # digest of the tool definitions
    tier: str
    prompt: str           # digest of the system prompt: change the prompt and every old entry is out of reach
    scope: str            # "shared", or "customer:<id>" for answers built from one customer's data

    @classmethod
    def of(cls, model, temperature=1.0, tools=None, tier="standard", system_prompt="", scope="shared"):
        return cls(model, round(float(temperature), 2), digest(tools or []), tier, digest(system_prompt), scope)


def ttl_for(text, personal):
    """Account answers and questions about right now go stale within the hour; the rest last a day."""
    return HOUR if personal or TIME_SENSITIVE.search(text) else DAY


@dataclass
class Entry:
    text: str
    response: object
    vector: np.ndarray
    created: float
    expires: float
    tags: frozenset = frozenset()
    meta: dict = field(default_factory=dict)
    hits: int = 0
    alive: bool = True


@dataclass
class Lookup:
    layer: str                 # "exact", "semantic" or "miss"
    entry: Entry = None        # the entry served, on a hit
    similarity: float = 0.0    # the best similarity found, hit or not
    nearest: Entry = None      # the best candidate, hit or not
    disagreed: bool = False    # close enough, but the cached neighbours gave different answers


class Partition:
    def __init__(self):
        self.entries, self.exact, self._arrays = [], {}, None

    def add(self, entry):
        self.entries.append(entry)
        self.exact[normalize(entry.text)] = entry
        self._arrays = None

    def arrays(self):
        """The vectors, expiry times and live flags as arrays, rebuilt only after a change."""
        if self._arrays is None:
            self._arrays = (np.vstack([e.vector for e in self.entries]),
                            np.array([e.expires for e in self.entries]),
                            np.array([e.alive for e in self.entries]))
        return self._arrays


def same_answer(a, b):
    """Whether two cached responses give the same answer. The replay's responses carry the
    answer they were written as; the proxy compares answer embeddings instead (proxy.py)."""
    return a == b if not isinstance(a, dict) else a.get("answer_id", a) == b.get("answer_id", b)


class SemanticCache:
    def __init__(self, threshold=0.9, clock=time.time, near_miss=0.03, neighbours=10, band=0.08, support=1, same=same_answer):
        self.threshold, self.clock, self.near_miss = threshold, clock, near_miss
        self.neighbours, self.band, self.support, self.same = neighbours, band, support, same
        self.partitions = {}
        self.stats = Counter()
        self.near_misses = deque(maxlen=1000)

    def lookup(self, key, text, vector, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        now = self.clock()
        part = self.partitions.get(key)
        if part is None or not part.entries:
            self.stats["miss"] += 1
            return Lookup("miss")
        exact = part.exact.get(normalize(text))
        if exact and exact.alive and exact.expires > now:
            exact.hits += 1
            self.stats["exact"] += 1
            return Lookup("exact", exact, 1.0, exact)
        matrix, expires, alive = part.arrays()
        sims = np.where(alive & (expires > now), matrix @ vector, -np.inf)
        best = int(np.argmax(sims))
        similarity, nearest = float(sims[best]), part.entries[best]
        if not np.isfinite(similarity):
            self.stats["miss"] += 1
            return Lookup("miss")
        if similarity >= threshold:
            close = np.argsort(-sims)[: self.neighbours]
            rivals = [j for j in close if j != best and sims[j] >= similarity - self.band]
            unanimous = all(self.same(part.entries[j].response, nearest.response) for j in rivals)
            witnesses = 1 + sum(sims[j] >= threshold for j in rivals)   # rivals are unanimous here, so all agree
            if unanimous and witnesses >= self.support:
                nearest.hits += 1
                self.stats["semantic"] += 1
                return Lookup("semantic", nearest, similarity, nearest)
            self.stats["miss"] += 1
            self.stats["disagreed"] += 1
            return Lookup("miss", None, similarity, nearest, disagreed=True)
        self.stats["miss"] += 1
        if similarity >= threshold - self.near_miss:
            self.stats["near_miss"] += 1
            self.near_misses.append({"text": text, "nearest": nearest.text, "similarity": round(similarity, 4)})
        return Lookup("miss", None, similarity, nearest)

    def store(self, key, text, vector, response, ttl, tags=(), meta=None):
        now = self.clock()
        entry = Entry(text, response, np.asarray(vector, np.float32), now, now + ttl, frozenset(tags), dict(meta or {}))
        self.partitions.setdefault(key, Partition()).add(entry)
        self.stats["stored"] += 1
        return entry

    def invalidate(self, tag=None, prompt=None, model=None):
        """Retire entries by tag (a policy or catalogue change), by prompt digest, or by model."""
        removed = 0
        for key, part in self.partitions.items():
            for e in part.entries:
                if e.alive and ((tag and tag in e.tags) or (prompt and key.prompt == prompt) or (model and key.model == model)):
                    e.alive = False
                    part._arrays = None
                    removed += 1
        self.stats["invalidated"] += removed
        return removed

    def size(self):
        now = self.clock()
        return sum(e.alive and e.expires > now for p in self.partitions.values() for e in p.entries)
