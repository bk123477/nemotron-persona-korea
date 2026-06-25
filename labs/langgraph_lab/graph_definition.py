"""
LangGraph 페르소나 분석 그래프 정의

그래프 구조:
    START
      ↓
    classify  ─────────────────────────────────────────┐
      ↓                                                 │
    [route_by_age]                                      │
      ├─ elderly → elderly_analysis ──────────────┐    │
      └─ general ─────────────────────────────────┤    │
                                                   ↓    │
                                            search_similar
                                                   ↓
                        ┌──────────────────────────┤
                        ↓          ↓               ↓
               occ_analysis  region_analysis  family_analysis   ← 병렬 실행
                        └──────────────────────────┘
                                                   ↓
                                           outlier_check
                                                   ↓
                                         human_review  ← interrupt
                                                   ↓
                                    [route_approval]
                                      ├─ approved → summarize → END
                                      └─ rejected → END
"""
from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, Literal

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from labs.langgraph_lab.nodes import (
    classify_node,
    elderly_analysis_node,
    search_similar_node,
    occupation_analysis_node,
    region_analysis_node,
    family_analysis_node,
    outlier_check_node,
    human_review_node,
    summarize_node,
)


# ── State 정의 ───────────────────────────────────────────────────────

class PersonaAnalysisState(TypedDict):
    # 입력
    persona_query: str

    # classify 이후
    persona_text: str
    metadata: dict
    age: int
    age_group: str
    is_elderly: bool

    # elderly_analysis 이후 (선택)
    elderly_note: str

    # search_similar 이후
    similar_personas: list[dict]

    # 병렬 분석 (occupation / region / family)
    occupation_analysis: str
    region_analysis: str
    family_analysis: str

    # outlier_check 이후
    outlier_flags: list[str]

    # human_review 이후
    human_approved: bool
    human_feedback: str

    # 최종
    summary: str

    # 실행 로그 (모든 노드가 append)
    log: Annotated[list[str], operator.add]


# ── 라우터 함수 ─────────────────────────────────────────────────────

def route_by_age(state: PersonaAnalysisState) -> Literal["elderly_analysis", "search_similar"]:
    """65세 이상이면 고령층 특화 분석 노드로, 아니면 바로 유사 검색으로."""
    return "elderly_analysis" if state["is_elderly"] else "search_similar"


def route_approval(state: PersonaAnalysisState) -> Literal["summarize", END]:
    """사용자 승인 여부에 따라 요약 생성 또는 종료."""
    return "summarize" if state.get("human_approved", False) else END


# ── 그래프 구성 ─────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    PersonaAnalysisState 기반 StateGraph를 빌드해 반환한다.

    Args:
        checkpointer: human-in-the-loop에 필요한 체크포인터
                      (기본값 None이면 interrupt 없이 실행)
    """
    builder = StateGraph(PersonaAnalysisState)

    # 노드 등록
    builder.add_node("classify", classify_node)
    builder.add_node("elderly_analysis", elderly_analysis_node)
    builder.add_node("search_similar", search_similar_node)
    builder.add_node("occ_analysis", occupation_analysis_node)
    builder.add_node("region_analysis", region_analysis_node)
    builder.add_node("family_analysis", family_analysis_node)
    builder.add_node("outlier_check", outlier_check_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("summarize", summarize_node)

    # 엣지: START → classify
    builder.add_edge(START, "classify")

    # 조건부 엣지: classify → (age 기준) elderly_analysis or search_similar
    builder.add_conditional_edges("classify", route_by_age)

    # 고령층 분석 → 유사 검색
    builder.add_edge("elderly_analysis", "search_similar")

    # 유사 검색 → 병렬 3개 노드 (fan-out)
    builder.add_edge("search_similar", "occ_analysis")
    builder.add_edge("search_similar", "region_analysis")
    builder.add_edge("search_similar", "family_analysis")

    # 병렬 3개 → outlier_check (fan-in, 모두 완료 후 실행)
    builder.add_edge("occ_analysis", "outlier_check")
    builder.add_edge("region_analysis", "outlier_check")
    builder.add_edge("family_analysis", "outlier_check")

    # outlier_check → human_review (interrupt 발생)
    builder.add_edge("outlier_check", "human_review")

    # 조건부 엣지: human_review → summarize or END
    builder.add_conditional_edges("human_review", route_approval)

    # summarize → END
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer)


def get_mermaid(graph) -> str:
    """Mermaid 다이어그램 문자열 반환."""
    return graph.get_graph().draw_mermaid()
