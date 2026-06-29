"""
pytest 공통 픽스처 및 설정
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# 프로젝트 루트를 sys.path에 추가
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    """테스트용 미니 JSONL 파일 (10건)."""
    records = []
    sexes = ["남자", "여자"]
    provinces = ["서울", "부산", "경기", "대구", "인천"]
    occs = ["교사", "의사", "엔지니어", "농업종사자", "사무직"]
    edus = ["4년제", "고졸", "대학원", "전문대"]
    maritals = ["배우자있음", "미혼", "이혼"]
    families = ["부부+자녀", "혼자", "부부만"]

    for i in range(10):
        records.append({
            "id": f"uuid-{i:04d}",
            "text": f"Persona: 테스트 페르소나 {i}번. Professional persona: 직업 설명 {i}.",
            "metadata": {
                "uuid": f"uuid-{i:04d}",
                "age": 20 + i * 5,
                "sex": sexes[i % 2],
                "province": provinces[i % len(provinces)],
                "district": f"테스트구{i}",
                "occupation": occs[i % len(occs)],
                "education_level": edus[i % len(edus)],
                "marital_status": maritals[i % len(maritals)],
                "family_type": families[i % len(families)],
                "housing_type": "아파트",
                "military_status": "해당없음" if i % 2 == 1 else "만기전역",
                "bachelors_field": "공학계열" if i % 3 == 0 else "",
                "country": "대한민국",
            },
        })

    out = tmp_path / "personas.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


@pytest.fixture
def sample_search_results():
    """테스트용 SearchResult 리스트."""
    from rag.schema import SearchResult

    return [
        SearchResult(
            id=f"id-{i}",
            text=f"Persona: 페르소나 텍스트 {i}.",
            metadata={
                "sex": "남자" if i % 2 == 0 else "여자",
                "age": 30 + i * 3,
                "province": "서울" if i < 3 else "부산",
                "occupation": "교사" if i % 2 == 0 else "의사",
                "education_level": "4년제",
                "marital_status": "미혼" if i < 2 else "배우자있음",
            },
            rank=i,
            score=1.0 / (i + 1),
            source="dense",
        )
        for i in range(5)
    ]
