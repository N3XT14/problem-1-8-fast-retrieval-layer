"""
Rule-based NL query parser: text: StructuredQuery.

Extracts structured filters with regex, uses the remainder as the semantic search string.
Deterministic and cheap — no LLM call, fits inside the parse latency budget.
"""

from __future__ import annotations

import re

from models import QueryFilters, StructuredQuery

# Patterns matched in order; matched spans are stripped from the semantic string
_PATTERNS: list[tuple[str, str]] = [
    ("video_id",       r"\bin\s+video[_\s]+([\w]+)\b"),
    ("video_id",       r"\bfrom\s+video[_\s]+([\w]+)\b"),
    ("timestamp_gte",  r"\bafter\s+(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b"),
    ("timestamp_lte",  r"\bbefore\s+(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b"),
    ("scene_id",       r"\bin\s+scene[_\s]+([\w]+)\b"),
    ("tags",           r"\btagged?\s+(?:with\s+)?([\w]+)\b"),
    ("tags",           r"\bwith\s+tag\s+([\w]+)\b"),
]

_TOP_K_PATTERN = re.compile(r"\btop[_\s](\d+)\b|\bfirst[_\s](\d+)\b", re.I)

def parse_query(text: str, top_k: int = 10) -> StructuredQuery:
    """Parses a natural language query into a StructuredQuery."""
    raw = text.strip()
    remainder = raw
    filters = QueryFilters()
    tags: list[str] = []

    for field, pattern in _PATTERNS:
        for m in re.finditer(pattern, remainder, re.I):
            value = m.group(1)
            if field == "video_id":
                filters.video_id = value
            elif field == "timestamp_gte":
                filters.timestamp_gte = float(value)
            elif field == "timestamp_lte":
                filters.timestamp_lte = float(value)
            elif field == "scene_id":
                filters.scene_id = value
            elif field == "tags":
                tags.append(value.lower())
            remainder = remainder[:m.start()] + " " + remainder[m.end():]

    if tags:
        filters.tags = list(dict.fromkeys(tags))  # deduplicate, preserve order

    # top_k extraction
    m = _TOP_K_PATTERN.search(remainder)
    if m:
        top_k = int(m.group(1) or m.group(2))
        remainder = remainder[:m.start()] + " " + remainder[m.end():]

    semantic = re.sub(r"\s{2,}", " ", remainder).strip()

    return StructuredQuery(
        raw=raw,
        semantic=semantic,
        filters=filters,
        top_k=top_k,
    )
