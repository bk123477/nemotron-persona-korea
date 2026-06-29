"""
단위 테스트: HybridRAGPipeline 내부 로직
"""
from __future__ import annotations

import pytest
from rag.schema import SearchResult


def _sr(id: str, rank: int, **meta) -> SearchResult:
    return SearchResult(id=id, text="t", metadata=meta, rank=rank, score=1.0, source="dense")


class TestToChromaWhere:
    def test_simple_eq(self):
        from rag.pipeline import _to_chroma_where
        assert _to_chroma_where({"sex": "여자"}) == {"sex": "여자"}

    def test_removes_contains(self):
        from rag.pipeline import _to_chroma_where
        result = _to_chroma_where({"occupation": {"$contains": "전문"}})
        assert result is None

    def test_mixed_keeps_supported(self):
        from rag.pipeline import _to_chroma_where
        result = _to_chroma_where({
            "$and": [
                {"sex": "여자"},
                {"occupation": {"$contains": "전문"}},
            ]
        })
        # $contains가 제거되면 $and가 단일 항목으로 unwrap
        assert result == {"sex": "여자"}

    def test_gte_lte_preserved(self):
        from rag.pipeline import _to_chroma_where
        result = _to_chroma_where({"age": {"$gte": 60, "$lte": 80}})
        assert result == {"age": {"$gte": 60, "$lte": 80}}

    def test_and_with_two_conditions(self):
        from rag.pipeline import _to_chroma_where
        result = _to_chroma_where({
            "$and": [{"sex": "여자"}, {"province": "서울"}]
        })
        assert result == {"$and": [{"sex": "여자"}, {"province": "서울"}]}

    def test_empty_condition_returns_none(self):
        from rag.pipeline import _to_chroma_where
        result = _to_chroma_where({"occupation": {"$contains": "x"}})
        assert result is None


class TestApplyMetadataFilter:
    def test_basic_filtering(self):
        from rag.pipeline import _apply_metadata_filter
        results = [
            _sr("a", 0, province="서울"),
            _sr("b", 1, province="부산"),
        ]
        out = _apply_metadata_filter(results, {"province": "서울"})
        assert [r.id for r in out] == ["a"]

    def test_nested_and_or(self):
        from rag.pipeline import _apply_metadata_filter
        results = [
            _sr("a", 0, sex="여자", age=30, province="서울"),
            _sr("b", 1, sex="남자", age=70, province="서울"),
            _sr("c", 2, sex="여자", age=70, province="부산"),
        ]
        out = _apply_metadata_filter(results, {
            "$and": [
                {"province": "서울"},
                {"$or": [{"sex": "여자"}, {"age": {"$gte": 60}}]},
            ]
        })
        assert {r.id for r in out} == {"a", "b"}

    def test_lte_filter(self):
        from rag.pipeline import _apply_metadata_filter
        results = [_sr(str(i), i, age=20 + i * 10) for i in range(5)]
        out = _apply_metadata_filter(results, {"age": {"$lte": 40}})
        assert all(r.metadata["age"] <= 40 for r in out)

    def test_ne_filter(self):
        from rag.pipeline import _apply_metadata_filter
        results = [
            _sr("a", 0, sex="여자"),
            _sr("b", 1, sex="남자"),
        ]
        out = _apply_metadata_filter(results, {"sex": {"$ne": "남자"}})
        assert [r.id for r in out] == ["a"]
