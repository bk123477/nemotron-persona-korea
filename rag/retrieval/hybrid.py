from __future__ import annotations

from collections import defaultdict

from ..schema import SearchResult


def reciprocal_rank_fusion(
    *result_lists: list[SearchResult],
    k: int = 60,
    top_k: int | None = None,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion (RRF) — 여러 랭킹 결과를 병합한다.

    RRF(d) = Σ  1 / (k + rank_i(d) + 1)

    Args:
        result_lists: 병합할 검색 결과 목록들 (dense, sparse, ...)
        k: RRF 상수 (기본값 60, 논문 권장값)
        top_k: 반환할 상위 결과 수 (None이면 전체)

    Returns:
        source="hybrid"로 표시된 병합 결과 리스트
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, SearchResult] = {}

    for result_list in result_lists:
        for result in result_list:
            rrf_scores[result.id] += 1.0 / (k + result.rank + 1)
            if result.id not in doc_map:
                doc_map[result.id] = result

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    merged: list[SearchResult] = []
    for rank, doc_id in enumerate(sorted_ids):
        base = doc_map[doc_id]
        merged.append(
            SearchResult(
                id=base.id,
                text=base.text,
                metadata=base.metadata,
                score=rrf_scores[doc_id],
                rank=rank,
                source="hybrid",
            )
        )
    return merged
