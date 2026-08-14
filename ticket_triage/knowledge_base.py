"""Lightweight knowledge base of FAQs / past resolutions used to ground auto-responses.

Uses a pure-Python TF-IDF-ish cosine similarity so the project has zero heavy
dependencies (no numpy/sklearn required). Good enough for a KB of hundreds to
low-thousands of entries; swap in a vector DB for larger scale.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .models import Category, KBMatch

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class KnowledgeBase:
    def __init__(self, entries: list[dict]):
        self._entries = entries
        self._doc_tokens: list[Counter] = [
            Counter(_tokenize(f"{e['question']} {e['resolution']}")) for e in entries
        ]
        self._df: Counter = Counter()
        for toks in self._doc_tokens:
            for term in toks:
                self._df[term] += 1
        self._n_docs = max(len(entries), 1)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "KnowledgeBase":
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        return cls(entries)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0

    def _score(self, query_tokens: Counter, doc_tokens: Counter) -> float:
        dot = 0.0
        for term, qcount in query_tokens.items():
            if term in doc_tokens:
                idf = self._idf(term)
                dot += qcount * doc_tokens[term] * idf * idf
        if dot == 0:
            return 0.0
        q_norm = math.sqrt(sum((c * self._idf(t)) ** 2 for t, c in query_tokens.items()))
        d_norm = math.sqrt(sum((c * self._idf(t)) ** 2 for t, c in doc_tokens.items()))
        if q_norm == 0 or d_norm == 0:
            return 0.0
        return dot / (q_norm * d_norm)

    def search(
        self,
        query: str,
        category: Category | None = None,
        top_k: int = 3,
        min_score: float = 0.08,
    ) -> list[KBMatch]:
        query_tokens = Counter(_tokenize(query))
        results: list[KBMatch] = []
        for entry, doc_tokens in zip(self._entries, self._doc_tokens):
            if category is not None and entry.get("category") != category.value:
                continue
            score = self._score(query_tokens, doc_tokens)
            if score >= min_score:
                results.append(
                    KBMatch(
                        kb_id=entry["id"],
                        question=entry["question"],
                        resolution=entry["resolution"],
                        score=round(score, 3),
                    )
                )
        results.sort(key=lambda m: m.score, reverse=True)
        return results[:top_k]
