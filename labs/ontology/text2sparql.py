"""
Stage 5: Text2SPARQL — LLM 기반 자연어 → SPARQL 변환

NVIDIA NIM LLM을 사용해 자연어 질문을 SPARQL 쿼리로 변환하고
온톨로지 그래프에서 실행한다.

사용법:
    cd /home/minkih/nemotron-persona-korea
    .venv/bin/python3 -m labs.ontology.text2sparql "서울 고령층의 직업 분포는?"
    .venv/bin/python3 -m labs.ontology.text2sparql  # 대화형 모드
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import labs.common.config  # noqa: F401 — SSL/proxy 패치
from labs.common.config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, _PROXY
from labs.ontology.query import _TTL_PATH, load_graph

_SYSTEM_PROMPT = """당신은 RDF/SPARQL 전문가입니다.
아래 온톨로지 스키마에 기반해 자연어 질문을 SPARQL SELECT 쿼리로 변환하세요.

## 네임스페이스
  PREFIX nemo: <http://nemotron.persona.kr/ontology#>
  PREFIX data: <http://nemotron.persona.kr/data/>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
  PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

## 클래스 및 접근 프로퍼티
  nemo:Persona            — 개별 페르소나
  nemo:Province           — 시도 (nemo:hasProvinceName → xsd:string)
  nemo:District           — 시군구 (nemo:hasDistrictName → xsd:string)
  nemo:OccupationCategory — 직업 (nemo:hasOccupationName → xsd:string)
  nemo:EducationLevel     — 학력 (nemo:hasEducationName → xsd:string)
  nemo:MaritalStatus      — 혼인상태 (rdfs:label @ko)
  nemo:FamilyType         — 가구형태 (rdfs:label @ko)
  nemo:HousingType        — 주거형태 (rdfs:label @ko)
  nemo:MilitaryStatus     — 병역상태 (rdfs:label @ko)
  nemo:SexCategory        — 성별 (rdfs:label @ko: "남자" / "여자")
  nemo:AgeGroup           — 연령대 (rdfs:label @ko)
    data:age_youth   = 청년 (18~29세)
    data:age_middle  = 중년 (30~49세)
    data:age_senior  = 장년 (50~64세)
    data:age_elderly = 고령 (65세 이상)

## Persona 프로퍼티
  nemo:hasAge (xsd:integer)     — 나이
  nemo:hasSex                   → nemo:SexCategory
  nemo:livesInProvince          → nemo:Province
  nemo:livesInDistrict          → nemo:District
  nemo:hasOccupation            → nemo:OccupationCategory
  nemo:hasEducation             → nemo:EducationLevel
  nemo:hasMaritalStatus         → nemo:MaritalStatus
  nemo:hasFamilyType            → nemo:FamilyType
  nemo:hasHousingType           → nemo:HousingType
  nemo:hasMilitaryStatus        → nemo:MilitaryStatus
  nemo:hasAgeGroup              → nemo:AgeGroup
  nemo:hasBachelorsField (xsd:string) — 전공 계열

## 작성 규칙
1. SPARQL SELECT 쿼리만 출력하세요 (설명 없이, 코드 블록 없이)
2. INSERT / DELETE / UPDATE 금지
3. 집계 시 GROUP BY + ORDER BY DESC(?count) 사용
4. 한국어 리터럴 필터: FILTER(LANG(?var) = "ko")
5. LIMIT 20 이하로 제한
6. 잘 모르면 가장 단순한 질의를 생성하세요"""


def natural_to_sparql(question: str) -> str:
    """자연어 질문 → SPARQL 쿼리 문자열."""
    if not OPENROUTER_API_KEY:
        raise ValueError("NVIDIA_API_KEY가 없습니다. ~/.env를 확인하세요.")

    import httpx
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        http_client=httpx.Client(proxy=_PROXY, verify=False),
        temperature=0.0,
        max_tokens=800,
    )

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"자연어 질문: {question}\n\nSPARQL 쿼리:"),
    ])

    sparql = response.content.strip()
    # 마크다운 코드 블록 제거
    if "```" in sparql:
        lines = sparql.split("\n")
        clean, in_block = [], False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block or not sparql.startswith("```"):
                clean.append(line)
        sparql = "\n".join(clean).strip()
    return sparql


def run_text2sparql(question: str, ttl_path: Path = _TTL_PATH) -> None:
    """자연어 질문을 SPARQL로 변환하고 결과를 출력한다."""
    print(f"[질문] {question}")
    print("[SPARQL 생성 중...]")

    try:
        sparql = natural_to_sparql(question)
    except Exception as e:
        print(f"[LLM 오류] {e}")
        return

    print(f"\n[생성된 SPARQL]\n{'─'*50}\n{sparql}\n{'─'*50}")

    print("\n[그래프 로딩 중...]")
    try:
        g = load_graph(ttl_path)
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        return

    print("[쿼리 실행 중...]")
    try:
        rows = list(g.query(sparql))
        print(f"\n[결과] {len(rows)}건")
        print('─' * 50)
        for i, row in enumerate(rows[:20]):
            print(f"  {i+1:2d}. " + " | ".join(str(v) for v in row))
        if len(rows) > 20:
            print(f"  ... (총 {len(rows)}건 중 20건 표시)")
    except Exception as e:
        print(f"[쿼리 실행 오류] {e}")
        print("생성된 SPARQL에 문제가 있을 수 있습니다. 질문을 바꿔 다시 시도하세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Text2SPARQL — 자연어 → SPARQL 변환")
    parser.add_argument("question", nargs="?", default=None,
                        help="자연어 질문 (없으면 대화형 모드)")
    parser.add_argument("--ttl", type=str, default=None,
                        help="TTL 파일 경로 (기본: labs/ontology/data/personas.ttl)")
    args = parser.parse_args()

    ttl_path = Path(args.ttl) if args.ttl else _TTL_PATH

    if args.question:
        run_text2sparql(args.question, ttl_path)
        return

    print("Text2SPARQL 대화형 모드 (종료: q)\n" + "─" * 55)
    print("예시 질문:")
    print("  - 서울 고령층의 직업 분포를 알려줘")
    print("  - 학력이 높은 미혼 여성의 나이 분포는?")
    print("  - 부산에서 가장 많은 직업 종류는?")
    print("  - 연령대별 혼인 상태 비교")
    print("─" * 55)

    try:
        while True:
            question = input("\n질문> ").strip()
            if not question or question.lower() in ("q", "quit", "exit"):
                break
            print()
            run_text2sparql(question, ttl_path)
    except (KeyboardInterrupt, EOFError):
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
