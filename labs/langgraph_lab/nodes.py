"""
LangGraph 페르소나 분석 그래프 — 노드 함수 모음

각 노드는 PersonaAnalysisState를 받아 변경된 키만 dict로 반환한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import labs.common.config  # noqa: F401 — SSL/proxy 패치
from labs.common.config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, _PROXY
from labs.common.loader import get_chroma_retriever

from langgraph.types import interrupt


# ── LLM 헬퍼 ────────────────────────────────────────────────────────

def _llm_call(prompt: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
    """OpenRouter LLM 단순 호출. API 키 없으면 '[LLM 없음]' 반환."""
    if not OPENROUTER_API_KEY:
        return "[LLM 없음 — OPENROUTER_API_KEY를 설정하세요]"
    import httpx
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        http_client=httpx.Client(proxy=_PROXY, verify=False),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


# ════════════════════════════════════════════════════════════════════
# 노드 1 — 페르소나 검색 + 인구통계 분류
# ════════════════════════════════════════════════════════════════════

def classify_node(state: dict) -> dict:
    """ChromaDB에서 페르소나를 검색하고 인구통계를 분류한다."""
    query = state["persona_query"]
    retriever = get_chroma_retriever(top_k=1)
    docs = retriever.invoke(query)

    if not docs:
        raise RuntimeError(
            "ChromaDB 검색 결과가 없습니다. "
            ".venv/bin/python3 -m rag.ingest --max-docs 1000 으로 인덱스를 먼저 구축하세요."
        )

    doc = docs[0]
    meta = doc.metadata
    age = int(meta.get("age", 0))

    age_group = (
        "청년 (18~29세)" if age < 30
        else "중년 (30~49세)" if age < 50
        else "장년 (50~64세)" if age < 65
        else "고령 (65세 이상)"
    )

    log_msg = (
        f"[분류] {meta.get('sex','?')} / {age}세({age_group}) / "
        f"{meta.get('province','?')} {meta.get('district','')} / "
        f"{meta.get('occupation','?')} / 학력:{meta.get('education_level','?')}"
    )

    return {
        "persona_text": doc.page_content,
        "metadata": meta,
        "age": age,
        "age_group": age_group,
        "is_elderly": age >= 65,
        "log": [log_msg],
    }


# ════════════════════════════════════════════════════════════════════
# 노드 2 — 고령층 특화 분석 (age >= 65 라우팅 시)
# ════════════════════════════════════════════════════════════════════

def elderly_analysis_node(state: dict) -> dict:
    """65세 이상 페르소나에 대한 고령층 특화 분석."""
    meta = state["metadata"]
    text = state["persona_text"][:600]

    prompt = (
        f"아래는 {meta.get('age')}세 한국인 페르소나입니다.\n\n{text}\n\n"
        "고령층 관점에서 다음을 분석하세요 (3문장 이내):\n"
        "1. 건강·신체 활동 수준\n"
        "2. 사회적 연결망 (가족, 이웃)\n"
        "3. 디지털 적응 수준 예측"
    )
    result = _llm_call(prompt, temperature=0.2)

    return {
        "elderly_note": result,
        "log": [f"[고령층 분석] 완료 ({meta.get('age')}세)"],
    }


# ════════════════════════════════════════════════════════════════════
# 노드 3 — 유사 페르소나 검색
# ════════════════════════════════════════════════════════════════════

def search_similar_node(state: dict) -> dict:
    """현재 페르소나와 유사한 페르소나 상위 5개를 검색한다."""
    text = state["persona_text"]
    retriever = get_chroma_retriever(top_k=6)  # 자기 자신 포함 가능하므로 +1
    docs = retriever.invoke(text[:300])

    similar = []
    my_id = state["metadata"].get("id") or state["metadata"].get("uuid", "")
    for doc in docs:
        doc_id = doc.metadata.get("id") or doc.metadata.get("uuid", "")
        if doc_id == my_id:
            continue
        similar.append({
            "sex": doc.metadata.get("sex", "?"),
            "age": doc.metadata.get("age", "?"),
            "province": doc.metadata.get("province", "?"),
            "occupation": doc.metadata.get("occupation", "?"),
            "education_level": doc.metadata.get("education_level", "?"),
            "text_preview": doc.page_content[:200],
        })
        if len(similar) >= 5:
            break

    log_msg = f"[유사 페르소나] {len(similar)}건 검색됨"
    return {
        "similar_personas": similar,
        "log": [log_msg],
    }


# ════════════════════════════════════════════════════════════════════
# 노드 4a — 직업 분석 (병렬)
# ════════════════════════════════════════════════════════════════════

def occupation_analysis_node(state: dict) -> dict:
    """직업 특성 분석."""
    meta = state["metadata"]
    occ = meta.get("occupation", "알 수 없음")
    edu = meta.get("education_level", "알 수 없음")
    age = state["age"]
    text = state["persona_text"][:400]

    prompt = (
        f"직업: {occ} / 학력: {edu} / 나이: {age}세\n\n"
        f"페르소나 텍스트:\n{text}\n\n"
        "이 직업 특성을 2~3문장으로 분석하세요:\n"
        "- 직업 안정성 및 전문성 수준\n"
        "- 이 직업에서 나타나는 주요 역량과 생활 패턴"
    )
    result = _llm_call(prompt, temperature=0.2)

    return {
        "occupation_analysis": result,
        "log": [f"[직업 분석] {occ}"],
    }


# ════════════════════════════════════════════════════════════════════
# 노드 4b — 지역 분석 (병렬)
# ════════════════════════════════════════════════════════════════════

def region_analysis_node(state: dict) -> dict:
    """지역 특성 분석."""
    meta = state["metadata"]
    province = meta.get("province", "알 수 없음")
    district = meta.get("district", "")
    housing = meta.get("housing_type", "알 수 없음")

    prompt = (
        f"거주지: {province} {district} / 주거 형태: {housing}\n\n"
        "이 지역 및 주거 특성을 2~3문장으로 분석하세요:\n"
        "- 해당 지역의 사회경제적 특성\n"
        "- 주거 환경이 생활 방식에 미치는 영향"
    )
    result = _llm_call(prompt, temperature=0.2)

    return {
        "region_analysis": result,
        "log": [f"[지역 분석] {province} {district}"],
    }


# ════════════════════════════════════════════════════════════════════
# 노드 4c — 가족형태 분석 (병렬)
# ════════════════════════════════════════════════════════════════════

def family_analysis_node(state: dict) -> dict:
    """가족 구성 및 생애주기 분석."""
    meta = state["metadata"]
    family_type = meta.get("family_type", "알 수 없음")
    marital = meta.get("marital_status", "알 수 없음")
    military = meta.get("military_status", "알 수 없음")
    age = state["age"]

    prompt = (
        f"가구 형태: {family_type} / 혼인 상태: {marital} / "
        f"나이: {age}세 / 병역: {military}\n\n"
        "이 가족 구성 특성을 2~3문장으로 분석하세요:\n"
        "- 현재 생애주기 단계\n"
        "- 가족 구조가 소비·여가 패턴에 미치는 영향"
    )
    result = _llm_call(prompt, temperature=0.2)

    return {
        "family_analysis": result,
        "log": [f"[가족 분석] {family_type} / {marital}"],
    }


# ════════════════════════════════════════════════════════════════════
# 노드 5 — 이상값 감지
# ════════════════════════════════════════════════════════════════════

def outlier_check_node(state: dict) -> dict:
    """인구통계 조합의 이상값·희귀 패턴을 탐지한다 (규칙 기반)."""
    meta = state["metadata"]
    age = state["age"]
    flags: list[str] = []

    edu = meta.get("education_level", "")
    occ = meta.get("occupation", "")
    marital = meta.get("marital_status", "")
    family_type = meta.get("family_type", "")
    military = meta.get("military_status", "")

    # 나이–학력 불일치
    if age < 25 and "대학원" in edu:
        flags.append("⚠ 25세 미만 대학원 졸업 (조기 졸업 또는 데이터 이상)")
    if age > 70 and "4년제" in edu and "교수" not in occ and "연구" not in occ:
        flags.append("⚠ 70대 이상 4년제 대졸 — 고학력 고령층 (희귀 조합)")

    # 병역–성별 불일치
    sex = meta.get("sex", "")
    if sex == "여자" and military not in ("", "해당없음", "해당 없음") and "현역" in military:
        flags.append("⚠ 여성 현역 병역 기록 (데이터 확인 필요)")

    # 나이–혼인 희귀 조합
    if age < 22 and "배우자있음" in marital:
        flags.append("⚠ 22세 미만 기혼 (조기 결혼)")
    if age > 80 and "미혼" in marital:
        flags.append("⚠ 80세 이상 미혼 (희귀 패턴)")

    # 1인 가구 고령
    if age >= 75 and "혼자" in family_type:
        flags.append("⚠ 75세 이상 1인 가구 (사회적 고립 위험)")

    summary = (
        f"{len(flags)}개 이상값 감지됨: {', '.join(flags)}"
        if flags
        else "이상값 없음 — 일반적인 인구통계 조합"
    )

    return {
        "outlier_flags": flags,
        "log": [f"[이상값 감지] {summary}"],
    }


# ════════════════════════════════════════════════════════════════════
# 노드 6 — Human-in-the-loop 검토
# ════════════════════════════════════════════════════════════════════

def human_review_node(state: dict) -> dict:
    """중간 분석 결과를 출력하고 사용자 승인을 대기한다 (interrupt)."""
    meta = state["metadata"]

    # 현재까지 분석 요약 출력
    review_text = (
        f"\n{'═'*60}\n"
        f"[중간 검토] 분석 대상 페르소나\n"
        f"{'─'*60}\n"
        f"  {meta.get('sex','?')} / {state['age']}세 ({state['age_group']}) / "
        f"{meta.get('province','?')} / {meta.get('occupation','?')}\n\n"
    )

    if state.get("elderly_note"):
        review_text += f"[고령층 분석]\n{state['elderly_note']}\n\n"

    if state.get("occupation_analysis"):
        review_text += f"[직업 분석]\n{state['occupation_analysis']}\n\n"

    if state.get("region_analysis"):
        review_text += f"[지역 분석]\n{state['region_analysis']}\n\n"

    if state.get("family_analysis"):
        review_text += f"[가족 분석]\n{state['family_analysis']}\n\n"

    flags = state.get("outlier_flags", [])
    if flags:
        review_text += f"[⚠ 이상값]\n" + "\n".join(f"  {f}" for f in flags) + "\n\n"
    else:
        review_text += "[이상값] 없음\n\n"

    review_text += f"{'─'*60}\n"

    # interrupt: 실행을 멈추고 사용자 입력을 기다림
    decision: str = interrupt({
        "message": review_text + "계속 진행하려면 'yes', 취소는 'no', 피드백은 텍스트 입력:",
        "state_snapshot": {
            "age": state["age"],
            "occupation": meta.get("occupation"),
            "province": meta.get("province"),
            "outlier_count": len(flags),
        },
    })

    approved = str(decision).strip().lower() in ("yes", "y", "승인", "ok", "계속", "진행")
    return {
        "human_approved": approved,
        "human_feedback": str(decision),
        "log": [f"[인간 검토] 결정: '{decision}' → {'승인' if approved else '거부'}"],
    }


# ════════════════════════════════════════════════════════════════════
# 노드 7 — 최종 종합 요약
# ════════════════════════════════════════════════════════════════════

def summarize_node(state: dict) -> dict:
    """모든 분석 결과를 종합해 최종 페르소나 인사이트를 생성한다."""
    meta = state["metadata"]

    sections: list[str] = []

    if state.get("occupation_analysis"):
        sections.append(f"[직업] {state['occupation_analysis']}")
    if state.get("region_analysis"):
        sections.append(f"[지역] {state['region_analysis']}")
    if state.get("family_analysis"):
        sections.append(f"[가족] {state['family_analysis']}")
    if state.get("elderly_note"):
        sections.append(f"[고령층 특성] {state['elderly_note']}")

    similar = state.get("similar_personas", [])
    if similar:
        sim_desc = "; ".join(
            f"{p['sex']}/{p['age']}세/{p['province']}/{p['occupation']}"
            for p in similar[:3]
        )
        sections.append(f"[유사 페르소나] {sim_desc}")

    flags = state.get("outlier_flags", [])
    if flags:
        sections.append(f"[이상값] {'; '.join(flags)}")

    feedback = state.get("human_feedback", "")
    feedback_note = f"\n[검토자 피드백] {feedback}" if feedback and feedback.lower() not in ("yes","y","ok","승인","계속","진행") else ""

    context = "\n\n".join(sections)
    prompt = (
        f"아래는 한국인 페르소나 분석 결과입니다.\n\n"
        f"대상: {meta.get('sex','?')} / {state['age']}세 / "
        f"{meta.get('province','?')} / {meta.get('occupation','?')}\n\n"
        f"{context}{feedback_note}\n\n"
        "위 분석을 바탕으로 이 페르소나의 핵심 특성과 마케팅/서비스 관점의 인사이트를 "
        "3~5문장으로 종합하세요."
    )
    summary = _llm_call(prompt, temperature=0.3, max_tokens=500)

    return {
        "summary": summary,
        "log": ["[요약] 최종 종합 완료"],
    }
