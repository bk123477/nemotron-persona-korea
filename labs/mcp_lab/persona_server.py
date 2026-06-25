"""
Nemotron-Personas-Korea MCP 서버

Tools:
  search_personas(query, sex, province, age_min, age_max, top_k)
  get_persona_by_id(uuid)
  get_demographic_stats(province, age_group)

Resources:
  persona://{uuid}  — 개별 페르소나 전체 문서

Prompts:
  persona_roleplay  — 검색된 페르소나로 롤플레이 시작 프롬프트

실행:
    cd /home/minkih/nemotron-persona-korea
    .venv/bin/python3 -m labs.mcp_lab.persona_server
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import labs.common.config  # noqa: F401 — SSL/proxy 패치
from labs.common.config import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL_CACHE, JSONL_PATH

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="persona-korea",
    instructions=(
        "Nemotron-Personas-Korea 데이터셋 MCP 서버입니다. "
        "한국인 인구통계 페르소나를 검색·조회하고 통계를 집계합니다. "
        "search_personas로 자연어 검색, get_persona_by_id로 직접 조회, "
        "get_demographic_stats로 인구통계 통계를 확인하세요."
    ),
)


# ── 내부 헬퍼 ────────────────────────────────────────────────────────

def _get_retriever(top_k: int = 5):
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_CACHE),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def _chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(CHROMA_COLLECTION)


def _meta_to_text(meta: dict) -> str:
    return (
        f"성별: {meta.get('sex','?')} | 나이: {meta.get('age','?')}세 | "
        f"지역: {meta.get('province','?')} {meta.get('district','')} | "
        f"직업: {meta.get('occupation','?')} | 학력: {meta.get('education_level','?')} | "
        f"혼인: {meta.get('marital_status','?')} | 가구: {meta.get('family_type','?')}"
    )


# ════════════════════════════════════════════════════════════════════
# Tools
# ════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_personas(
    query: str,
    sex: Optional[str] = None,
    province: Optional[str] = None,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    top_k: int = 5,
) -> str:
    """
    자연어 쿼리로 페르소나를 의미 기반(semantic) 검색합니다.

    Args:
        query: 검색 질의 (예: "여행을 좋아하는 전문직 종사자")
        sex: 성별 필터 ("남자" 또는 "여자")
        province: 시도 필터 (예: "서울", "부산", "광주")
        age_min: 최소 나이
        age_max: 최대 나이
        top_k: 반환할 결과 수 (최대 20)

    Returns:
        검색된 페르소나 목록 (JSON 문자열)
    """
    top_k = min(top_k, 20)

    # where 필터 구성 (ChromaDB 지원 연산자만)
    clauses = []
    if sex:
        clauses.append({"sex": sex})
    if province:
        clauses.append({"province": province})
    if age_min is not None:
        clauses.append({"age": {"$gte": age_min}})
    if age_max is not None:
        clauses.append({"age": {"$lte": age_max}})

    search_kwargs: dict = {"k": top_k}
    if len(clauses) == 1:
        search_kwargs["filter"] = clauses[0]
    elif len(clauses) > 1:
        search_kwargs["filter"] = {"$and": clauses}

    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_CACHE),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(query)

    results = []
    for doc in docs:
        meta = doc.metadata
        results.append({
            "id": meta.get("id") or meta.get("uuid", ""),
            "profile": _meta_to_text(meta),
            "text_preview": doc.page_content[:300],
            "metadata": {k: v for k, v in meta.items()
                         if k in ("sex", "age", "province", "district",
                                  "occupation", "education_level",
                                  "marital_status", "family_type")},
        })

    return json.dumps(
        {"query": query, "count": len(results), "results": results},
        ensure_ascii=False, indent=2,
    )


@mcp.tool()
def get_persona_by_id(uuid: str) -> str:
    """
    UUID로 특정 페르소나를 직접 조회합니다.

    Args:
        uuid: 페르소나 고유 ID (32자리 16진수)

    Returns:
        페르소나 전체 정보 (JSON 문자열)
    """
    # ChromaDB에서 먼저 조회
    try:
        col = _chroma_collection()
        result = col.get(ids=[uuid], include=["documents", "metadatas"])
        if result["ids"]:
            meta = result["metadatas"][0]
            text = result["documents"][0]
            return json.dumps({
                "id": uuid,
                "source": "chromadb",
                "profile": _meta_to_text(meta),
                "metadata": meta,
                "text": text,
            }, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # ChromaDB에 없으면 JSONL에서 순차 탐색
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("id") == uuid or obj.get("metadata", {}).get("uuid") == uuid:
                return json.dumps({
                    "id": uuid,
                    "source": "jsonl",
                    "profile": _meta_to_text(obj.get("metadata", {})),
                    "metadata": obj.get("metadata", {}),
                    "text": obj.get("text", ""),
                }, ensure_ascii=False, indent=2)

    return json.dumps({"error": f"uuid '{uuid}'를 찾을 수 없습니다."}, ensure_ascii=False)


@mcp.tool()
def get_demographic_stats(
    province: Optional[str] = None,
    age_group: Optional[str] = None,
) -> str:
    """
    인덱싱된 페르소나의 인구통계 통계를 집계합니다.

    Args:
        province: 시도 필터 (예: "서울"). None이면 전체
        age_group: 연령대 필터 ("청년"=18~29 / "중년"=30~49 / "장년"=50~64 / "고령"=65+)

    Returns:
        통계 집계 결과 (JSON 문자열)
    """
    col = _chroma_collection()
    total = col.count()

    # 전체 메타데이터 로드 (인덱스 규모가 작으므로 허용)
    all_meta = col.get(limit=total, include=["metadatas"])["metadatas"]

    # 연령대 범위
    age_ranges = {
        "청년": (18, 29),
        "중년": (30, 49),
        "장년": (50, 64),
        "고령": (65, 120),
    }

    # 필터 적용
    filtered = all_meta
    if province:
        filtered = [m for m in filtered if m.get("province") == province]
    if age_group and age_group in age_ranges:
        lo, hi = age_ranges[age_group]
        filtered = [
            m for m in filtered
            if lo <= int(m.get("age", 0)) <= hi
        ]

    if not filtered:
        return json.dumps({"error": "조건에 맞는 페르소나가 없습니다."}, ensure_ascii=False)

    # 집계
    from collections import Counter

    sex_dist = Counter(m.get("sex", "?") for m in filtered)
    prov_dist = Counter(m.get("province", "?") for m in filtered)
    occ_dist = Counter(m.get("occupation", "?") for m in filtered)
    edu_dist = Counter(m.get("education_level", "?") for m in filtered)
    marital_dist = Counter(m.get("marital_status", "?") for m in filtered)
    ages = [int(m.get("age", 0)) for m in filtered if m.get("age")]

    stats = {
        "filter": {"province": province, "age_group": age_group},
        "total_indexed": total,
        "matched": len(filtered),
        "age": {
            "min": min(ages) if ages else None,
            "max": max(ages) if ages else None,
            "avg": round(sum(ages) / len(ages), 1) if ages else None,
        },
        "sex": dict(sex_dist.most_common()),
        "province_top5": dict(prov_dist.most_common(5)),
        "occupation_top5": dict(occ_dist.most_common(5)),
        "education": dict(edu_dist.most_common()),
        "marital_status": dict(marital_dist.most_common()),
    }

    return json.dumps(stats, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════════
# Resource
# ════════════════════════════════════════════════════════════════════

@mcp.resource("persona://{uuid}")
def get_persona_resource(uuid: str) -> str:
    """
    개별 페르소나 문서를 Resource로 노출합니다.
    URI: persona://{uuid}
    """
    return get_persona_by_id(uuid)


# ════════════════════════════════════════════════════════════════════
# Prompt
# ════════════════════════════════════════════════════════════════════

@mcp.prompt()
def persona_roleplay(query: str = "한국인 페르소나") -> str:
    """
    검색된 페르소나로 롤플레이를 시작하는 프롬프트를 생성합니다.

    Args:
        query: 롤플레이할 페르소나 검색 질의

    Returns:
        LLM에 전달할 시스템 프롬프트 문자열
    """
    # 가장 유사한 페르소나 1건 검색
    result_json = search_personas(query=query, top_k=1)
    result = json.loads(result_json)

    if not result.get("results"):
        return (
            "페르소나를 찾을 수 없습니다. "
            "search_personas 도구로 먼저 검색해 보세요."
        )

    persona = result["results"][0]
    meta = persona.get("metadata", {})
    preview = persona.get("text_preview", "")

    return (
        f"당신은 아래 프로필의 한국인입니다. "
        f"이 사람의 성격, 경험, 가치관을 그대로 살려 1인칭으로 자연스럽게 대화하세요.\n\n"
        f"[프로필]\n{persona['profile']}\n\n"
        f"[페르소나 설명]\n{preview}\n\n"
        f"대화 시 유의사항:\n"
        f"- '{meta.get('province','')}' 지역 사투리나 어투를 자연스럽게 반영하세요.\n"
        f"- {meta.get('age','?')}세 {meta.get('sex','?')}의 관심사와 생활 방식을 반영하세요.\n"
        f"- 직업({meta.get('occupation','?')})과 관련된 경험을 자연스럽게 언급하세요.\n"
        f"- 모르는 사실은 꾸며내지 마세요."
    )


# ── 진입점 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
