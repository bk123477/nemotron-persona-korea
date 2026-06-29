"""
단위 테스트: retrieval 레이어 — RRF 알고리즘 및 메타데이터 필터
"""
from __future__ import annotations

import pytest
from rag.schema import SearchResult


def _sr(id: str, rank: int, score: float = 1.0, source: str = "dense") -> SearchResult:
    return SearchResult(id=id, text="test", metadata={}, rank=rank, score=score, source=source)


# ── RRF 알고리즘 ──────────────────────────────────────────────────────

class TestRRF:
    def test_basic_merge(self):
        from rag.retrieval.hybrid import reciprocal_rank_fusion
        dense = [_sr(f"d{i}", i) for i in range(5)]
        sparse = [_sr(f"s{i}", i) for i in range(5)]
        results = reciprocal_rank_fusion(dense, sparse, k=60, top_k=10)
        assert len(results) == 10

    def test_overlap_boosts_rank(self):
        """양쪽 리스트에 모두 있는 문서가 상위에 위치해야 한다."""
        from rag.retrieval.hybrid import reciprocal_rank_fusion
        dense = [_sr("shared", 0), _sr("d1", 1), _sr("d2", 2)]
        sparse = [_sr("shared", 0), _sr("s1", 1), _sr("s2", 2)]
        results = reciprocal_rank_fusion(dense, sparse, k=60, top_k=5)
        assert results[0].id == "shared"

    def test_top_k_limit(self):
        from rag.retrieval.hybrid import reciprocal_rank_fusion
        dense = [_sr(f"d{i}", i) for i in range(10)]
        sparse = [_sr(f"s{i}", i) for i in range(10)]
        results = reciprocal_rank_fusion(dense, sparse, k=60, top_k=3)
        assert len(results) <= 3

    def test_empty_sparse(self):
        """sparse 결과가 빈 경우 dense 결과만 반환해야 한다."""
        from rag.retrieval.hybrid import reciprocal_rank_fusion
        dense = [_sr(f"d{i}", i) for i in range(5)]
        results = reciprocal_rank_fusion(dense, [], k=60, top_k=5)
        assert len(results) == 5

    def test_rank_assigned_sequentially(self):
        from rag.retrieval.hybrid import reciprocal_rank_fusion
        dense = [_sr(f"d{i}", i) for i in range(3)]
        sparse = [_sr(f"s{i}", i) for i in range(3)]
        results = reciprocal_rank_fusion(dense, sparse, k=60, top_k=6)
        for i, r in enumerate(results):
            assert r.rank == i


# ── 메타데이터 필터 ───────────────────────────────────────────────────

def _sr_meta(id: str, rank: int, **meta) -> SearchResult:
    return SearchResult(id=id, text="test", metadata=meta, rank=rank, score=1.0, source="dense")


class TestMetadataFilter:
    def test_eq_filter(self):
        from rag.pipeline import _apply_metadata_filter
        results = [
            _sr_meta("a", 0, sex="여자", age=30),
            _sr_meta("b", 1, sex="남자", age=25),
        ]
        filtered = _apply_metadata_filter(results, {"sex": "여자"})
        assert len(filtered) == 1
        assert filtered[0].id == "a"

    def test_gte_filter(self):
        from rag.pipeline import _apply_metadata_filter
        results = [
            _sr_meta("a", 0, age=70),
            _sr_meta("b", 1, age=30),
            _sr_meta("c", 2, age=65),
        ]
        filtered = _apply_metadata_filter(results, {"age": {"$gte": 65}})
        assert {r.id for r in filtered} == {"a", "c"}

    def test_contains_filter(self):
        from rag.pipeline import _apply_metadata_filter
        results = [
            _sr_meta("a", 0, occupation="전문 의사"),
            _sr_meta("b", 1, occupation="농업"),
            _sr_meta("c", 2, occupation="전문 엔지니어"),
        ]
        filtered = _apply_metadata_filter(results, {"occupation": {"$contains": "전문"}})
        assert {r.id for r in filtered} == {"a", "c"}

    def test_and_filter(self):
        from rag.pipeline import _apply_metadata_filter
        results = [
            _sr_meta("a", 0, sex="여자", age=25),
            _sr_meta("b", 1, sex="여자", age=35),
            _sr_meta("c", 2, sex="남자", age=25),
        ]
        filtered = _apply_metadata_filter(results, {
            "$and": [{"sex": "여자"}, {"age": {"$gte": 30}}]
        })
        assert len(filtered) == 1
        assert filtered[0].id == "b"

    def test_or_filter(self):
        from rag.pipeline import _apply_metadata_filter
        results = [
            _sr_meta("a", 0, province="서울"),
            _sr_meta("b", 1, province="부산"),
            _sr_meta("c", 2, province="경기"),
        ]
        filtered = _apply_metadata_filter(results, {
            "$or": [{"province": "서울"}, {"province": "부산"}]
        })
        assert {r.id for r in filtered} == {"a", "b"}

    def test_to_chroma_where_strips_contains(self):
        """$contains 연산자는 ChromaDB where에서 제거돼야 한다."""
        from rag.pipeline import _to_chroma_where
        where = {"occupation": {"$contains": "전문"}}
        result = _to_chroma_where(where)
        assert result is None  # $contains만 있으면 None 반환

    def test_to_chroma_where_keeps_eq(self):
        from rag.pipeline import _to_chroma_where
        where = {"sex": "여자"}
        result = _to_chroma_where(where)
        assert result == {"sex": "여자"}
