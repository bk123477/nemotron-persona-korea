from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING, Any

from rank_bm25 import BM25Okapi

from ..schema import Document, SearchResult

if TYPE_CHECKING:
    from ..config import RAGConfig

logger = logging.getLogger(__name__)

# BM25 인덱스에 보관할 최소 메타데이터 (메모리 절약)
_META_FIELDS = {
    "sex", "age", "province", "district", "occupation",
    "education_level", "marital_status",
}


def _tokenize(text: str, mode: str = "whitespace") -> list[str]:
    if mode == "char":
        return list(text.replace(" ", ""))
    return text.split()


class SparseRetriever:
    """BM25 기반 Sparse 검색기."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    def is_loaded(self) -> bool:
        return self._bm25 is not None

    def doc_count(self) -> int:
        return len(self._ids)

    def build(self, documents: list[Document]) -> None:
        logger.info("BM25 인덱스 생성 중... (%d 문서)", len(documents))
        tokenized = [_tokenize(d.text, self.config.bm25_tokenizer) for d in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._ids = [d.id for d in documents]
        self._texts = [d.text for d in documents]
        self._metadatas = [
            {k: v for k, v in d.metadata.items() if k in _META_FIELDS}
            for d in documents
        ]
        self._save()
        logger.info("BM25 인덱스 완료 → %s", self.config.bm25_path)

    def _save(self) -> None:
        self.config.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self._bm25,
            "ids": self._ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
        }
        with open(self.config.bm25_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 저장 완료 (%.1f MB)", self.config.bm25_path.stat().st_size / 1e6)

    def load(self) -> None:
        with open(self.config.bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._ids = data["ids"]
        self._texts = data["texts"]
        self._metadatas = data["metadatas"]
        logger.info("BM25 로드 완료 (%d 문서)", len(self._ids))

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        if not self.is_loaded():
            raise RuntimeError("BM25 인덱스가 없습니다. build() 또는 load()를 먼저 실행하세요.")
        k = top_k or self.config.sparse_top_k
        tokens = _tokenize(query, self.config.bm25_tokenizer)
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:k]

        results: list[SearchResult] = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break
            results.append(
                SearchResult(
                    id=self._ids[idx],
                    text=self._texts[idx],
                    metadata=self._metadatas[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    source="sparse",
                )
            )
        return results
