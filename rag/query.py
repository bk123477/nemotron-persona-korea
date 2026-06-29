"""
Hybrid RAG 검색 데모 CLI

사용법:
    cd /home/minkih/nemotron-persona-korea

    # 단일 질의
    .venv/bin/python3 -m rag.query "광주에 사는 70대 하역 노동자"

    # 모드 선택 & JSON 출력
    .venv/bin/python3 -m rag.query "미혼 20대 여성 여행 좋아하는 사람" --mode dense --json

    # 메타데이터 필터 (Dense only)
    .venv/bin/python3 -m rag.query "전문직 종사자" --filter-sex 여자 --filter-province 서울

    # 대화형 모드
    .venv/bin/python3 -m rag.query
"""
from __future__ import annotations

import argparse
import json
import logging

from .config import RAGConfig, setup_ssl, suppress_hf_warnings
from .llm.nim import NIMLM

setup_ssl()
suppress_hf_warnings()
from .pipeline import HybridRAGPipeline, SearchMode
from .schema import SearchResult

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _format_result(r: SearchResult, verbose: bool = False) -> str:
    meta = r.metadata
    lines = [
        f"[{r.rank + 1:2d}] score={r.score:.4f}  source={r.source}",
        f"     {meta.get('sex', '?')} / {meta.get('age', '?')}세 / "
        f"{meta.get('province', '?')} {meta.get('district', '')} / "
        f"{meta.get('occupation', '?')}",
        f"     학력={meta.get('education_level', '?')}  "
        f"혼인={meta.get('marital_status', '?')}  "
        f"가구={meta.get('family_type', '?')}",
    ]
    if verbose:
        lines.append(f"     {r.text[:300]}...")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nemotron Persona Hybrid RAG 검색")
    parser.add_argument("query", nargs="?", default=None,
                        help="검색 질의 (없으면 대화형 모드)")
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"],
                        default="hybrid", help="검색 모드 (기본: hybrid)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="반환 결과 수 (기본: 5)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Cohere Reranker 비활성화")
    parser.add_argument("--verbose", action="store_true",
                        help="페르소나 텍스트 일부 출력")
    parser.add_argument("--json", action="store_true",
                        help="JSON 형식으로 출력")
    # 메타데이터 필터 (Dense 검색 시 적용)
    parser.add_argument("--filter-sex", type=str, default=None,
                        help="성별 필터 예) 여자 / 남자")
    parser.add_argument("--filter-province", type=str, default=None,
                        help="시도 필터 예) 서울 / 부산")
    parser.add_argument("--filter-age-gte", type=int, default=None,
                        help="나이 하한 예) 60")
    parser.add_argument("--filter-age-lte", type=int, default=None,
                        help="나이 상한 예) 80")
    parser.add_argument("--filter-occupation", type=str, default=None,
                        help="직업 키워드 필터 (부분 일치) 예) 전문 / 의사 / 교사 / 엔지니어")
    parser.add_argument("--answer", action="store_true",
                        help="검색 후 NVIDIA NIM LLM으로 답변 생성")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM 모델 오버라이드 예) nvidia/nemotron-3-super-120b-a12b")
    parser.add_argument("--no-stream", action="store_true",
                        help="LLM 스트리밍 비활성화")
    args = parser.parse_args()

    config = RAGConfig()
    config.final_top_k = args.top_k
    if args.llm_model:
        config.llm_model = args.llm_model
    pipeline = HybridRAGPipeline(config)

    # LLM 초기화 (--answer 플래그 시)
    llm: NIMLM | None = None
    if args.answer:
        try:
            llm = NIMLM(config)
        except ValueError as e:
            print(f"[LLM 오류] {e}")
            return

    # 상태 출력
    st = pipeline.status()
    print(
        f"[인덱스 상태]  Dense: {st['dense_docs']:,}건 | "
        f"Sparse(BM25): {st['sparse_docs']:,}건 | "
        f"Reranker: {'✓ ' + (st['reranker_model'] or '') if st['reranker'] else '✗ (COHERE_API_KEY 없음)'}"
    )

    if st["dense_docs"] == 0 and st["sparse_docs"] == 0:
        print("\n인덱스가 비어 있습니다. 먼저 실행하세요:")
        print("  .venv/bin/python3 -m rag.ingest --max-docs 1000")
        return

    # 메타데이터 필터 구성
    where: dict | None = None
    filter_clauses = []
    if args.filter_sex:
        filter_clauses.append({"sex": args.filter_sex})
    if args.filter_province:
        filter_clauses.append({"province": args.filter_province})
    if args.filter_age_gte is not None:
        filter_clauses.append({"age": {"$gte": args.filter_age_gte}})
    if args.filter_age_lte is not None:
        filter_clauses.append({"age": {"$lte": args.filter_age_lte}})
    if args.filter_occupation:
        filter_clauses.append({"occupation": {"$contains": args.filter_occupation}})
    if len(filter_clauses) == 1:
        where = filter_clauses[0]
    elif len(filter_clauses) > 1:
        where = {"$and": filter_clauses}

    def run_query(q: str) -> None:
        results = pipeline.search(
            q,
            mode=args.mode,
            top_k=args.top_k,
            use_rerank=not args.no_rerank,
            where=where,
        )
        if args.json:
            print(json.dumps([
                {
                    "rank": r.rank + 1,
                    "score": round(r.score, 6),
                    "source": r.source,
                    "id": r.id,
                    "metadata": r.metadata,
                    "text_preview": r.text[:300],
                }
                for r in results
            ], ensure_ascii=False, indent=2))
        else:
            header = (
                f"\n{'─' * 60}\n"
                f"질의: {q}  |  모드: {args.mode}  |  결과: {len(results)}건"
                + (f"  |  필터: {where}" if where else "")
                + f"\n{'─' * 60}"
            )
            print(header)
            for r in results:
                print(_format_result(r, verbose=args.verbose))

            # LLM 답변 생성
            if llm and results:
                print(f"\n{'═' * 60}")
                print(f"[LLM 답변] {config.llm_model}")
                print('═' * 60)
                if args.no_stream:
                    print(llm.answer(q, results))
                else:
                    for token in llm.answer(q, results, stream=True):
                        print(token, end="", flush=True)
                    print()

    if args.query:
        run_query(args.query)
    else:
        print("\n대화형 검색 모드 (종료: q 또는 Ctrl+C)\n")
        try:
            while True:
                q = input("질의> ").strip()
                if not q:
                    continue
                if q.lower() in ("q", "quit", "exit"):
                    break
                run_query(q)
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")


if __name__ == "__main__":
    main()
