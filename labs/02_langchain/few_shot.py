"""
Few-shot 페르소나 생성기 — 유사 페르소나 N개를 few-shot 예시로 자동 삽입

검색된 유사 페르소나들을 few-shot 예시로 활용해
새로운 인구통계 조건에 맞는 페르소나 텍스트를 생성한다.

사용법:
    cd /home/minkih/nemotron-persona-korea

    # 기본 (유사 페르소나 3개 자동 검색 후 신규 생성)
    .venv/bin/python3 -m labs.02_langchain.few_shot "40대 여성 교사 경기도"

    # few-shot 수 조정
    .venv/bin/python3 -m labs.02_langchain.few_shot "20대 남성 개발자 서울" --n-shots 5

    # 생성할 페르소나의 인구통계 직접 지정
    .venv/bin/python3 -m labs.02_langchain.few_shot "부산 60대 어부" \\
        --target-sex 남자 --target-age 63 --target-province 부산
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import labs.common.config  # noqa: F401
from labs.common.config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, _PROXY
from labs.common.loader import get_chroma_retriever

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI


# ── Few-shot 예시 포매터 ─────────────────────────────────────────────

def _doc_to_example(doc: Document) -> dict[str, str]:
    """LangChain Document → few-shot 예시 dict."""
    meta = doc.metadata
    demographic = (
        f"{meta.get('sex','?')} / {meta.get('age','?')}세 / "
        f"{meta.get('province','')} {meta.get('district','')} / "
        f"{meta.get('occupation','?')} / 학력:{meta.get('education_level','?')}"
    )
    # text에서 'Persona:' 섹션만 추출
    text = doc.page_content
    persona_section = text
    if "Professional persona:" in text:
        persona_section = text[:text.index("Professional persona:")].replace("Persona:", "").strip()

    return {
        "demographic": demographic,
        "persona": persona_section[:300],
    }


# ── 체인 구성 ────────────────────────────────────────────────────────

def build_few_shot_chain(n_shots: int = 3):
    """
    검색된 유사 페르소나 N개 → few-shot 프롬프트 → 신규 페르소나 생성 체인.

    반환: (chain, retriever)
    chain.invoke({"query": ..., "target_demographic": ...})
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY가 없습니다.")
    import httpx

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        http_client=httpx.Client(proxy=_PROXY, verify=False),
        temperature=0.85,
        max_tokens=600,
    )
    retriever = get_chroma_retriever(top_k=n_shots)

    # few-shot 예시 프롬프트 템플릿
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "다음 인구통계 조건: {demographic}"),
        ("ai", "{persona}"),
    ])

    def _run(inputs: dict) -> str:
        query: str = inputs["query"]
        target: str = inputs.get("target_demographic", query)

        # 유사 페르소나 검색
        docs = retriever.invoke(query)
        if not docs:
            raise RuntimeError("검색 결과 없음. 인덱스를 먼저 구축하세요.")

        examples = [_doc_to_example(d) for d in docs]

        # FewShotChatMessagePromptTemplate 동적 구성
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        final_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 한국 사회의 다양한 인물 페르소나를 자연스럽고 구체적으로 서술하는 작가입니다.\n"
             "아래 예시들을 참고해 주어진 인구통계 조건에 맞는 새로운 페르소나를 200자 내외로 작성하세요.\n"
             "실제 인물을 묘사하듯 생생하고 구체적인 한국어 문장으로 써주세요."),
            few_shot_prompt,
            ("human", "다음 인구통계 조건: {target_demographic}"),
        ])

        chain = final_prompt | llm | StrOutputParser()
        return chain.invoke({"target_demographic": target})

    return _run, retriever


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot 페르소나 생성기")
    parser.add_argument("query", help="유사 페르소나 검색 질의 (예: 40대 여성 교사 경기도)")
    parser.add_argument("--n-shots", type=int, default=3, help="few-shot 예시 수 (기본: 3)")
    parser.add_argument("--target-sex", type=str, default=None)
    parser.add_argument("--target-age", type=int, default=None)
    parser.add_argument("--target-province", type=str, default=None)
    parser.add_argument("--target-occupation", type=str, default=None)
    args = parser.parse_args()

    # 타겟 인구통계 문자열 구성
    parts: list[str] = []
    if args.target_sex:
        parts.append(f"성별:{args.target_sex}")
    if args.target_age:
        parts.append(f"나이:{args.target_age}세")
    if args.target_province:
        parts.append(f"지역:{args.target_province}")
    if args.target_occupation:
        parts.append(f"직업:{args.target_occupation}")
    target_demographic = " / ".join(parts) if parts else args.query

    print(f"[유사 페르소나 검색: '{args.query}' | few-shot {args.n_shots}개]")
    run_chain, retriever = build_few_shot_chain(n_shots=args.n_shots)

    # few-shot 예시 미리보기
    docs = retriever.invoke(args.query)
    if docs:
        print(f"\n[Few-shot 예시 {len(docs)}개]")
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            print(
                f"  [{i}] {meta.get('sex','?')} / {meta.get('age','?')}세 / "
                f"{meta.get('province','?')} / {meta.get('occupation','?')}"
            )

    print(f"\n[생성 대상: {target_demographic}]")
    print("─" * 50)
    print("[생성 중...]")

    result = run_chain({"query": args.query, "target_demographic": target_demographic})

    print("\n[생성된 페르소나]")
    print(result)


if __name__ == "__main__":
    main()
