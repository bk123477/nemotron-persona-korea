"""
LangGraph 페르소나 분석 그래프 실행 CLI

사용법:
    cd /home/minkih/nemotron-persona-korea

    # 기본 실행 (human-in-the-loop 포함)
    .venv/bin/python3 -m labs.langgraph_lab.run "광주에 사는 70대 하역 노동자"

    # 자동 승인 (중단 없이 전체 실행)
    .venv/bin/python3 -m labs.langgraph_lab.run "서울 30대 직장인" --auto-approve

    # Mermaid 다이어그램 출력 후 종료
    .venv/bin/python3 -m labs.langgraph_lab.run --mermaid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import labs.common.config  # noqa: F401

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from labs.langgraph_lab.graph_definition import build_graph, get_mermaid


# ── 출력 헬퍼 ───────────────────────────────────────────────────────

_NODE_LABELS = {
    "classify":         "인구통계 분류",
    "elderly_analysis": "고령층 특화 분석",
    "search_similar":   "유사 페르소나 검색",
    "occ_analysis":     "직업 분석",
    "region_analysis":  "지역 분석",
    "family_analysis":  "가족 분석",
    "outlier_check":    "이상값 감지",
    "human_review":     "인간 검토 (interrupt)",
    "summarize":        "최종 요약",
}


def _print_event(event: dict, verbose: bool = False) -> None:
    for node_name, output in event.items():
        if node_name == "__interrupt__":
            continue
        label = _NODE_LABELS.get(node_name, node_name)
        log_lines = output.get("log", []) if isinstance(output, dict) else []
        for line in log_lines:
            print(f"  ▶ [{label}] {line}")
        if verbose and isinstance(output, dict):
            for k, v in output.items():
                if k == "log" or not v:
                    continue
                if isinstance(v, str) and len(v) > 60:
                    print(f"      {k}: {v[:200]}{'...' if len(v) > 200 else ''}")


def _print_final(state: dict) -> None:
    sep = "═" * 60
    print(f"\n{sep}")
    print("[ 최종 분석 결과 ]")
    print(sep)

    meta = state.get("metadata", {})
    print(
        f"  대상: {meta.get('sex','?')} / {state.get('age','?')}세 "
        f"({state.get('age_group','?')}) / {meta.get('province','?')} / "
        f"{meta.get('occupation','?')}"
    )

    if state.get("elderly_note"):
        print(f"\n[고령층 분석]\n{state['elderly_note']}")

    if state.get("occupation_analysis"):
        print(f"\n[직업 분석]\n{state['occupation_analysis']}")

    if state.get("region_analysis"):
        print(f"\n[지역 분석]\n{state['region_analysis']}")

    if state.get("family_analysis"):
        print(f"\n[가족 분석]\n{state['family_analysis']}")

    flags = state.get("outlier_flags", [])
    if flags:
        print("\n[⚠ 이상값]")
        for f in flags:
            print(f"  {f}")
    else:
        print("\n[이상값] 없음")

    similar = state.get("similar_personas", [])
    if similar:
        print(f"\n[유사 페르소나 Top-{len(similar)}]")
        for i, p in enumerate(similar, 1):
            print(
                f"  [{i}] {p['sex']} / {p['age']}세 / "
                f"{p['province']} / {p['occupation']}"
            )

    if state.get("summary"):
        print(f"\n[종합 인사이트]\n{state['summary']}")

    print(f"\n{sep}")

    log = state.get("log", [])
    if log:
        print("[실행 로그]")
        for line in log:
            print(f"  {line}")


# ── 메인 실행 ────────────────────────────────────────────────────────

def run(query: str, auto_approve: bool = False, verbose: bool = False) -> dict:
    """
    그래프를 실행하고 최종 state를 반환한다.

    human_review 노드에서 interrupt가 발생하면:
    - auto_approve=True: 자동으로 'yes' 전송
    - auto_approve=False: 터미널에서 사용자 입력 대기
    """
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "persona-analysis-1"}}
    initial_input = {"persona_query": query}

    print(f"\n[그래프 실행 시작] 질의: '{query}'")
    print("─" * 60)

    # ── 1단계: interrupt 전까지 실행 ────────────────────────────────
    interrupted = False
    for event in graph.stream(initial_input, config=config, stream_mode="updates"):
        _print_event(event, verbose=verbose)

        # interrupt 감지
        if "__interrupt__" in event:
            interrupted = True
            interrupt_data = event["__interrupt__"]
            payload = interrupt_data[0].value if interrupt_data else {}
            msg = payload.get("message", "분석 결과를 검토하세요.") if isinstance(payload, dict) else str(payload)
            print(msg)
            break

    # ── 2단계: 사용자 응답 후 재개 ──────────────────────────────────
    if interrupted:
        if auto_approve:
            decision = "yes"
            print(f"[자동 승인] '{decision}'")
        else:
            decision = input("입력 > ").strip() or "yes"

        print("─" * 60)
        print("[그래프 재개]")
        for event in graph.stream(
            Command(resume=decision),
            config=config,
            stream_mode="updates",
        ):
            _print_event(event, verbose=verbose)

    # ── 최종 state 반환 ─────────────────────────────────────────────
    final_state = graph.get_state(config).values
    _print_final(final_state)
    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph 페르소나 분석 그래프")
    parser.add_argument("query", nargs="?", default=None, help="검색 질의")
    parser.add_argument("--auto-approve", action="store_true", help="human 검토 자동 승인")
    parser.add_argument("--verbose", action="store_true", help="노드별 상세 출력")
    parser.add_argument("--mermaid", action="store_true", help="Mermaid 다이어그램 출력 후 종료")
    args = parser.parse_args()

    # 다이어그램 출력
    if args.mermaid:
        g = build_graph()
        print(get_mermaid(g))
        return

    query = args.query
    if not query:
        query = input("페르소나 검색 질의를 입력하세요: ").strip()
    if not query:
        query = "한국인 페르소나"

    run(query, auto_approve=args.auto_approve, verbose=args.verbose)


if __name__ == "__main__":
    main()
