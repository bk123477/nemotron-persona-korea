"""
페르소나 주입 챗봇 — LCEL 체인

검색된 페르소나를 SystemPrompt에 주입하고 LLM이 그 사람처럼 대화한다.

사용법:
    cd /home/minkih/nemotron-persona-korea
    .venv/bin/python3 -m labs.02_langchain.persona_chat "광주에 사는 70대 하역 노동자"
    .venv/bin/python3 -m labs.02_langchain.persona_chat  # 대화형 모드
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 프로젝트 루트 경로 설정
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import labs.common.config  # noqa: F401 — SSL/proxy 패치 적용
from labs.common.config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, _PROXY
from labs.common.loader import get_chroma_retriever

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI


# ── LLM 초기화 ──────────────────────────────────────────────────────

def _make_llm() -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY가 없습니다. ~/.env를 확인하세요.")
    import httpx
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        http_client=httpx.Client(proxy=_PROXY, verify=False),
        temperature=0.7,
        max_tokens=512,
    )


# ── 페르소나 추출 ────────────────────────────────────────────────────

def _docs_to_persona(docs: list[Document]) -> dict[str, Any]:
    """검색된 문서에서 가장 유사도 높은 1개 페르소나를 추출."""
    if not docs:
        raise ValueError("검색 결과가 없습니다. 인덱스에 문서를 먼저 적재하세요.")
    doc = docs[0]
    meta = doc.metadata

    profile = (
        f"이름은 알 수 없지만, 다음 특성을 가진 한국인입니다:\n"
        f"- 성별: {meta.get('sex', '알 수 없음')}\n"
        f"- 나이: {meta.get('age', '알 수 없음')}세\n"
        f"- 거주지: {meta.get('province', '')} {meta.get('district', '')}\n"
        f"- 직업: {meta.get('occupation', '알 수 없음')}\n"
        f"- 학력: {meta.get('education_level', '알 수 없음')}\n"
        f"- 혼인 상태: {meta.get('marital_status', '알 수 없음')}\n"
        f"- 가구 형태: {meta.get('family_type', '알 수 없음')}\n"
        f"\n[페르소나 상세]\n{doc.page_content[:800]}"
    )
    return {"persona_profile": profile, "doc": doc}


# ── 시스템 프롬프트 ──────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """당신은 아래 프로필의 한국인입니다. 이 사람의 성격, 경험, 가치관, 말투를 그대로 살려 자연스럽게 대화하세요.

{persona_profile}

대화 규칙:
- 1인칭("나는", "저는")으로 대답하세요.
- 이 사람의 직업, 지역, 나이대에 어울리는 어투와 관심사를 반영하세요.
- 모르는 사실은 꾸며내지 말고 "잘 모르겠네요"라고 하세요.
- 한국어로만 대답하세요."""


def build_chain():
    """retriever → persona 주입 → chat 체인 (LCEL)."""
    retriever = get_chroma_retriever(top_k=3)
    llm = _make_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    # retriever_chain: query → docs → persona_profile
    retriever_chain = (
        RunnableLambda(lambda x: x["persona_query"])
        | retriever
        | RunnableLambda(_docs_to_persona)
    )

    # 전체 LCEL 체인
    chain = (
        RunnablePassthrough.assign(persona_info=retriever_chain)
        | RunnableLambda(lambda x: {
            "persona_profile": x["persona_info"]["persona_profile"],
            "history": x.get("history", []),
            "question": x["question"],
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever_chain


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    persona_query = sys.argv[1] if len(sys.argv) > 1 else None

    if not persona_query:
        persona_query = input("어떤 페르소나를 불러올까요? (예: 서울 30대 직장인): ").strip()
        if not persona_query:
            persona_query = "한국인 페르소나"

    print(f"\n[페르소나 검색 중: '{persona_query}']")
    chain, retriever_chain = build_chain()

    # 페르소나 미리보기
    retriever = get_chroma_retriever(top_k=1)
    docs = retriever.invoke(persona_query)
    if docs:
        meta = docs[0].metadata
        print(
            f"[매칭된 페르소나] "
            f"{meta.get('sex','?')} / {meta.get('age','?')}세 / "
            f"{meta.get('province','?')} / {meta.get('occupation','?')}"
        )

    print("\n대화를 시작합니다. (종료: q)\n" + "─" * 50)
    history: list = []

    try:
        while True:
            user_input = input("나 > ").strip()
            if not user_input or user_input.lower() in ("q", "quit", "exit"):
                break

            response = chain.invoke({
                "persona_query": persona_query,
                "question": user_input,
                "history": history,
            })
            print(f"페르소나 > {response}\n")

            history.append(HumanMessage(content=user_input))
            history.append(SystemMessage(content=response))
            # 컨텍스트 윈도우 관리: 최근 6턴 유지
            if len(history) > 12:
                history = history[-12:]

    except (KeyboardInterrupt, EOFError):
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
