from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..schema import SearchResult

if TYPE_CHECKING:
    from ..config import RAGConfig

logger = logging.getLogger(__name__)

try:
    import cohere
    _COHERE_AVAILABLE = True
except ImportError:
    _COHERE_AVAILABLE = False


class CohereReranker:
    """Cohere Rerank v3 (rerank-multilingual-v3.0) 기반 재순위 모듈."""

    def __init__(self, config: RAGConfig) -> None:
        if not _COHERE_AVAILABLE:
            raise ImportError("cohere 패키지가 없습니다: pip install cohere>=5.0.0")
        if not config.cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY가 설정되지 않았습니다.\n"
                "  export COHERE_API_KEY=<your-key>  를 실행하세요."
            )
        self._client = cohere.ClientV2(api_key=config.cohere_api_key)
        self._model = config.cohere_rerank_model
        logger.info("Cohere Reranker 초기화 완료 (model=%s)", self._model)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Args:
            query: 원본 검색 질의
            candidates: RRF 등으로 1차 병합된 후보 결과
            top_k: 최종 반환 수 (None이면 candidates 전체)

        Returns:
            Cohere relevance_score 기준으로 재정렬된 결과 (source="reranked")
        """
        if not candidates:
            return []

        k = top_k or len(candidates)
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=[c.text for c in candidates],
            top_n=k,
        )

        reranked: list[SearchResult] = []
        for rank, item in enumerate(response.results):
            base = candidates[item.index]
            reranked.append(
                SearchResult(
                    id=base.id,
                    text=base.text,
                    metadata=base.metadata,
                    score=item.relevance_score,
                    rank=rank,
                    source="reranked",
                )
            )
        logger.debug("Cohere rerank: %d → %d 결과", len(candidates), len(reranked))
        return reranked
