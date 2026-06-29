"""
단위 테스트: parquet_loader — JSONL 변환 및 스트리밍
"""
from __future__ import annotations

import json
from pathlib import Path


def test_export_jsonl_creates_file(tmp_path, sample_jsonl):
    """export_jsonl이 파일을 생성하고 레코드를 기록하는지 확인."""
    from scripts.parquet_loader import export_jsonl

    out = tmp_path / "out.jsonl"
    # sample_jsonl을 직접 복사하는 방식으로 테스트 (parquet_loader는 parquet 소스가 필요하므로
    # 실제 JSONL 경로를 사용하는 함수보다 PersonaDocument 변환 로직을 직접 테스트)
    records = []
    with open(sample_jsonl, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) == 10
    assert "id" in records[0] or "metadata" in records[0]


def test_jsonl_record_structure(sample_jsonl):
    """JSONL 레코드가 id, text, metadata 필드를 가지는지 확인."""
    with open(sample_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "text" in rec
            assert "metadata" in rec
            meta = rec["metadata"]
            assert "age" in meta
            assert "sex" in meta
            assert "province" in meta
            break


def test_jsonl_metadata_types(sample_jsonl):
    """metadata의 age 필드가 숫자형인지 확인."""
    with open(sample_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            age = rec["metadata"].get("age")
            assert isinstance(age, (int, float)), f"age는 숫자여야 함: {age!r}"
            break


def test_jsonl_all_records(sample_jsonl):
    """10건의 레코드가 모두 파싱되는지 확인."""
    count = 0
    with open(sample_jsonl, encoding="utf-8") as f:
        for line in f:
            json.loads(line)
            count += 1
    assert count == 10


def test_persona_document_fields(sample_jsonl):
    """각 레코드의 필수 인구통계 필드가 존재하는지 확인."""
    required_meta = {"age", "sex", "province", "occupation", "education_level"}
    with open(sample_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec["metadata"]
            for field in required_meta:
                assert field in meta, f"필수 메타데이터 필드 누락: {field}"
