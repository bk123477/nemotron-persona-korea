"""
구조화 페르소나 추출기 — PydanticOutputParser + LCEL

비정형 페르소나 텍스트에서 핵심 속성을 타입 안전하게 추출한다.

사용법:
    cd /home/minkih/nemotron-persona-korea
    # JSONL에서 무작위 1건 추출
    .venv/bin/python3 -m labs.02_langchain.extractor

    # 직접 텍스트 입력
    .venv/bin/python3 -m labs.02_langchain.extractor --text "서울 강남에 사는 40대 의사..."

    # 쿼리로 검색 후 추출
    .venv/bin/python3 -m labs.02_langchain.extractor --query "부산 60대 어부"
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import labs.common.config  # noqa: F401
from labs.common.config import (
    JSONL_PATH, LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, _PROXY,
)
from labs.common.loader import get_chroma_retriever

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ── Pydantic 스키마 ──────────────────────────────────────────────────

class PersonaTraits(BaseModel):
    """페르소나 핵심 속성 구조화 결과."""
    name_or_alias: str = Field(description="이름 또는 별칭 (없으면 '이름 미상')")
    age_group: str = Field(description="연령대 (예: 20대, 30대, 60대 이상)")
    occupation_summary: str = Field(description="직업 요약 (1~2 단어)")
    personality_keywords: list[str] = Field(
        description="성격·기질 키워드 3개 (예: ['성실함', '사교적', '보수적'])"
    )
    top_hobby: str = Field(description="가장 두드러진 취미 또는 관심사")
    travel_style: str = Field(description="여행 스타일 요약 (없으면 '정보 없음')")
    life_goal: str = Field(description="삶의 목표 또는 가치관 한 줄 요약")
    notable_fact: str = Field(description="이 페르소나에서 가장 인상적인 사실 하나")


# ── LLM + 체인 ──────────────────────────────────────────────────────

def build_extractor_chain():
    """persona text → PersonaTraits LCEL 체인."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY가 없습니다.")
    import httpx

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        http_client=httpx.Client(proxy=_PROXY, verify=False),
        temperature=0.0,
        max_tokens=512,
    )

    parser = PydanticOutputParser(pydantic_object=PersonaTraits)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 한국어 페르소나 텍스트 분석 전문가입니다.\n"
         "주어진 페르소나 텍스트를 읽고 아래 JSON 형식으로 핵심 속성을 추출하세요.\n\n"
         "{format_instructions}"),
        ("human", "다음 페르소나 텍스트를 분석하세요:\n\n{persona_text}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


# ── 유틸 ────────────────────────────────────────────────────────────

def _random_persona_text(max_chars: int = 1200) -> str:
    """JSONL에서 무작위 레코드의 text를 반환."""
    total = sum(1 for _ in open(JSONL_PATH, encoding="utf-8"))
    idx = random.randint(0, total - 1)
    with open(JSONL_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)["text"][:max_chars]
    return ""


def _query_persona_text(query: str, max_chars: int = 1200) -> tuple[str, dict]:
    """검색 질의로 페르소나 text와 metadata를 반환."""
    retriever = get_chroma_retriever(top_k=1)
    docs = retriever.invoke(query)
    if not docs:
        raise RuntimeError("검색 결과가 없습니다. 인덱스에 문서를 먼저 적재하세요.")
    doc = docs[0]
    return doc.page_content[:max_chars], doc.metadata


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="페르소나 텍스트 → 구조화 속성 추출")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, help="직접 입력할 페르소나 텍스트")
    group.add_argument("--query", type=str, help="ChromaDB 검색 질의")
    args = parser.parse_args()

    meta: dict = {}
    if args.text:
        persona_text = args.text
    elif args.query:
        print(f"[검색 중: '{args.query}']")
        persona_text, meta = _query_persona_text(args.query)
        if meta:
            print(
                f"[매칭] {meta.get('sex','?')} / {meta.get('age','?')}세 / "
                f"{meta.get('province','?')} / {meta.get('occupation','?')}\n"
            )
    else:
        print("[JSONL에서 무작위 페르소나 선택]")
        persona_text = _random_persona_text()

    print("─" * 50)
    print("[원본 텍스트 미리보기]")
    print(persona_text[:300] + ("..." if len(persona_text) > 300 else ""))
    print("─" * 50)

    chain = build_extractor_chain()
    print("[추출 중...]")
    result: PersonaTraits = chain.invoke({"persona_text": persona_text})

    print("\n[추출 결과]")
    print(f"  이름/별칭    : {result.name_or_alias}")
    print(f"  연령대       : {result.age_group}")
    print(f"  직업 요약    : {result.occupation_summary}")
    print(f"  성격 키워드  : {', '.join(result.personality_keywords)}")
    print(f"  주요 취미    : {result.top_hobby}")
    print(f"  여행 스타일  : {result.travel_style}")
    print(f"  삶의 목표    : {result.life_goal}")
    print(f"  인상적 사실  : {result.notable_fact}")

    print("\n[JSON]")
    print(result.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
