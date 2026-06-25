from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    rank: int
    source: str  # "dense" | "sparse" | "hybrid" | "reranked"
