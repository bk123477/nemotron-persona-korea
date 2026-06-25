from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ..schema import Document, SearchResult

if TYPE_CHECKING:
    from ..config import RAGConfig

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500

# ChromaDB에 저장할 메타데이터 필드 (인구통계 필드만 유지)
_META_FIELDS = {
    "uuid", "sex", "age", "marital_status", "military_status",
    "family_type", "housing_type", "education_level",
    "bachelors_field", "occupation", "district", "province", "country",
}


def _sanitize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB 저장용 메타데이터 정제 (age는 int, 나머지는 str)."""
    result: dict[str, Any] = {}
    for key in _META_FIELDS:
        val = meta.get(key)
        if val is None:
            continue
        if key == "age":
            try:
                result[key] = int(val)
            except (ValueError, TypeError):
                result[key] = str(val)
        else:
            result[key] = str(val)
    return result


class DenseRetriever:
    """ChromaDB + sentence-transformers 기반 Dense 검색기."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        config.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model,
            device="cpu",
        )
        self._client = chromadb.PersistentClient(path=str(config.chroma_dir))
        self._col = self._client.get_or_create_collection(
            name=config.chroma_collection,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._col.count()

    def add(self, documents: list[Document], resume: bool = False) -> int:
        """
        문서를 배치로 upsert.

        Args:
            resume: True면 이미 인덱싱된 ID를 스킵 (중단 후 재시작 시 사용)

        Returns:
            실제로 추가/업데이트된 문서 수
        """
        already = self._col.count()
        if resume and already > 0:
            # ChromaDB에 있는 ID 목록을 페이지 단위로 가져와 스킵
            existing_ids: set[str] = set()
            offset = 0
            page = 1000
            while True:
                result = self._col.get(limit=page, offset=offset, include=[])
                ids = result["ids"]
                if not ids:
                    break
                existing_ids.update(ids)
                offset += len(ids)
            documents = [d for d in documents if d.id not in existing_ids]
            logger.info("Resume 모드: 기존 %d건 스킵, 신규 %d건 추가 예정", len(existing_ids), len(documents))

        added = 0
        for i in range(0, len(documents), _BATCH_SIZE):
            batch = documents[i : i + _BATCH_SIZE]
            self._col.upsert(
                ids=[d.id for d in batch],
                documents=[d.text for d in batch],
                metadatas=[_sanitize_meta(d.metadata) for d in batch],
            )
            added += len(batch)
            logger.info("Dense: %d / %d (전체 DB: %d건)", added, len(documents), self._col.count())
        return added

    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Args:
            where: ChromaDB 메타데이터 필터
                   예) {"sex": "여자"} 또는 {"age": {"$gte": 60}}
        """
        k = min(top_k or self.config.dense_top_k, self._col.count() or 1)
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": k}
        if where:
            kwargs["where"] = where

        res = self._col.query(**kwargs)
        results: list[SearchResult] = []
        for rank, (doc_id, text, meta, dist) in enumerate(
            zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            )
        ):
            results.append(
                SearchResult(
                    id=doc_id,
                    text=text,
                    metadata=meta,
                    score=1.0 - dist,  # cosine distance → similarity
                    rank=rank,
                    source="dense",
                )
            )
        return results
