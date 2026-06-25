from __future__ import annotations

import logging
from typing import Any

from .config import RAGConfig
from .reranking.cohere import CohereReranker
from .retrieval.dense import DenseRetriever
from .retrieval.hybrid import reciprocal_rank_fusion
from .retrieval.sparse import SparseRetriever
from .schema import SearchResult

logger = logging.getLogger(__name__)


def _apply_metadata_filter(results: list[SearchResult], where: dict) -> list[SearchResult]:
    """ChromaDB where 구문을 Python 측에서 재현해 후처리 필터를 적용.

    ChromaDB 표준 연산자($eq/$ne/$gte/$lte/$gt/$lt) 외에
    $contains (부분 문자열 일치)를 추가 지원한다.
    """

    def _match(meta: dict, clause: dict) -> bool:
        for key, condition in clause.items():
            if key == "$and":
                if not all(_match(meta, sub) for sub in condition):
                    return False
            elif key == "$or":
                if not any(_match(meta, sub) for sub in condition):
                    return False
            elif isinstance(condition, dict):
                val = meta.get(key)
                for op, ref in condition.items():
                    if val is None:
                        return False
                    try:
                        val_n, ref_n = float(val), float(ref)
                    except (TypeError, ValueError):
                        val_n, ref_n = None, None
                    if op == "$eq" and str(val) != str(ref):
                        return False
                    elif op == "$ne" and str(val) == str(ref):
                        return False
                    elif op == "$gte" and (val_n is None or val_n < ref_n):
                        return False
                    elif op == "$lte" and (val_n is None or val_n > ref_n):
                        return False
                    elif op == "$gt" and (val_n is None or val_n <= ref_n):
                        return False
                    elif op == "$lt" and (val_n is None or val_n >= ref_n):
                        return False
                    elif op == "$contains" and str(ref) not in str(val):
                        return False
            else:
                if str(meta.get(key, "")) != str(condition):
                    return False
        return True

    return [r for r in results if _match(r.metadata, where)]


# ChromaDB가 지원하는 연산자 집합 ($contains 등 Python 전용 연산자는 제외)
_CHROMA_OPS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}


def _to_chroma_where(where: dict) -> dict | None:
    """where 절에서 ChromaDB가 지원하지 않는 연산자($contains 등)를 제거해 반환.

    ChromaDB에 전달할 수 없는 조건만 있으면 None을 반환한다.
    ChromaDB는 $and/$or에 최소 2개 항목이 필요하므로, 1개로 줄어들면 unwrap한다.
    """
    def _filter_clause(clause: dict) -> dict | None:
        result = {}
        for key, condition in clause.items():
            if key in ("$and", "$or"):
                subs = [_filter_clause(sub) for sub in condition]
                subs = [s for s in subs if s]
                if len(subs) >= 2:
                    result[key] = subs
                elif len(subs) == 1:
                    # 단일 항목이면 $and/$or 벗겨내고 그냥 조건만 사용
                    result.update(subs[0])
                # 0개면 무시
            elif isinstance(condition, dict):
                supported = {op: ref for op, ref in condition.items() if op in _CHROMA_OPS}
                if supported:
                    result[key] = supported
            else:
                # 단순 동등 비교 {"field": "value"} → ChromaDB 지원
                result[key] = condition
        return result if result else None

    return _filter_clause(where)


class SearchMode:
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class HybridRAGPipeline:
    """
    Hybrid RAG 파이프라인 (대한항공 아키텍처 참조)
    ────────────────────────────────────────────
    [Dense (ChromaDB)]  ┐
                        ├→ RRF Fusion → [Cohere Rerank v3] → 결과
    [Sparse (BM25)]     ┘
    ────────────────────────────────────────────
    LLM(OpenRouter) 연동은 별도로 추가 예정.
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.dense = DenseRetriever(self.config)
        self.sparse = SparseRetriever(self.config)
        self._reranker: CohereReranker | None = None

        # BM25 인덱스가 있으면 자동 로드
        if self.config.bm25_path.exists():
            self.sparse.load()

        # Cohere API 키가 있으면 Reranker 초기화
        if self.config.cohere_api_key:
            try:
                self._reranker = CohereReranker(self.config)
            except Exception as exc:
                logger.warning("Cohere Reranker 비활성화: %s", exc)

    # ── 검색 ────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        mode: str = SearchMode.HYBRID,
        top_k: int | None = None,
        use_rerank: bool = True,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Args:
            query     : 자연어 검색 질의 (한국어)
            mode      : "dense" | "sparse" | "hybrid"
            top_k     : 최종 반환 수 (기본값 config.final_top_k)
            use_rerank: Cohere 재순위 적용 여부 (API 키 필요)
            where     : 메타데이터 필터. ChromaDB 표준 연산자 외에 $contains(부분 일치) 지원.
                        예) {"sex": "여자"} | {"age": {"$gte": 60}} | {"occupation": {"$contains": "전문"}}

        Returns:
            SearchResult 리스트 (rank 오름차순)
        """
        k = top_k or self.config.final_top_k
        pool_k = k * 3  # rerank pool: 최종보다 3배 많은 후보 수집

        # ChromaDB에 넘길 수 있는 연산자만 추린 필터 ($contains 등 제거)
        chroma_where = _to_chroma_where(where) if where else None

        if mode == SearchMode.DENSE:
            candidates = self.dense.search(query, top_k=pool_k * 2, where=chroma_where)
            if where:
                candidates = _apply_metadata_filter(candidates, where)

        elif mode == SearchMode.SPARSE:
            if not self.sparse.is_loaded():
                raise RuntimeError("BM25 인덱스가 없습니다. ingest.py를 먼저 실행하세요.")
            candidates = self.sparse.search(query, top_k=pool_k)
            if where:
                candidates = _apply_metadata_filter(candidates, where)

        else:  # HYBRID
            dense_res = self.dense.search(query, top_k=pool_k, where=chroma_where)
            sparse_res = (
                self.sparse.search(query, top_k=pool_k)
                if self.sparse.is_loaded()
                else []
            )
            if sparse_res:
                candidates = reciprocal_rank_fusion(
                    dense_res, sparse_res,
                    k=self.config.rrf_k,
                    top_k=pool_k,
                )
            else:
                # BM25 없으면 Dense만 사용
                logger.warning("BM25 인덱스 없음 → Dense 단독 사용")
                candidates = dense_res

            # BM25는 where를 우회하므로 항상 Python 후처리 필터 적용
            if where:
                candidates = _apply_metadata_filter(candidates, where)

        if use_rerank and self._reranker:
            return self._reranker.rerank(query, candidates, top_k=k)

        return candidates[:k]

    # ── 상태 확인 ────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "dense_docs": self.dense.count(),
            "sparse_docs": self.sparse.doc_count(),
            "reranker": self._reranker is not None,
            "reranker_model": self.config.cohere_rerank_model if self._reranker else None,
            "embedding_model": self.config.embedding_model,
        }
