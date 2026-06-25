"""
JSONL → Dense (ChromaDB) + Sparse (BM25) 인덱스 적재 스크립트

사용법:
    cd /home/minkih/nemotron-persona-korea
    # 테스트 (1,000건)
    .venv/bin/python3 -m rag.ingest --max-docs 1000

    # BM25만 (10만 건)
    .venv/bin/python3 -m rag.ingest --max-bm25-docs 100000 --skip-dense

    # 전체 적재
    .venv/bin/python3 -m rag.ingest
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import RAGConfig, setup_ssl, suppress_hf_warnings

setup_ssl()
suppress_hf_warnings()
from .retrieval.dense import DenseRetriever
from .retrieval.sparse import SparseRetriever
from .schema import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path, max_docs: int | None = None) -> list[Document]:
    docs: list[Document] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
            obj = json.loads(line)
            docs.append(
                Document(
                    id=obj["id"],
                    text=obj["text"],
                    metadata=obj.get("metadata", {}),
                )
            )
    logger.info("JSONL 로드 완료: %d 문서 (%s)", len(docs), path)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Nemotron Personas → RAG 인덱스 적재")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="전체 로드 문서 수 제한 (기본: 전체)")
    parser.add_argument("--max-bm25-docs", type=int, default=100_000,
                        help="BM25에 적재할 최대 문서 수 (기본: 100,000)")
    parser.add_argument("--skip-dense", action="store_true",
                        help="ChromaDB Dense 인덱스 적재 건너뜀")
    parser.add_argument("--skip-sparse", action="store_true",
                        help="BM25 Sparse 인덱스 생성 건너뜀")
    parser.add_argument("--resume", action="store_true",
                        help="이미 인덱싱된 문서를 스킵하고 이어서 적재 (중단 후 재시작)")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="임베딩 모델 오버라이드 (기본: jhgan/ko-sroberta-multitask)")
    args = parser.parse_args()

    config = RAGConfig()
    config.max_bm25_docs = args.max_bm25_docs
    if args.embedding_model:
        config.embedding_model = args.embedding_model

    logger.info("설정 — embedding: %s | bm25_limit: %d", config.embedding_model, config.max_bm25_docs)

    docs = load_jsonl(config.jsonl_path, max_docs=args.max_docs)

    # ── Dense 인덱스 ──────────────────────────────────────
    if not args.skip_dense:
        logger.info("Dense 인덱스 적재 시작 (ChromaDB)...")
        dense = DenseRetriever(config)
        before = dense.count()
        added = dense.add(docs, resume=args.resume)
        logger.info("Dense 완료: %d → %d 문서 (+%d)", before, dense.count(), added)

    # ── Sparse 인덱스 ─────────────────────────────────────
    if not args.skip_sparse:
        bm25_docs = docs[: config.max_bm25_docs]
        logger.info("Sparse 인덱스 생성 시작 (BM25, %d 문서)...", len(bm25_docs))
        sparse = SparseRetriever(config)
        sparse.build(bm25_docs)
        logger.info("Sparse 완료.")

    logger.info("적재 완료!")


if __name__ == "__main__":
    main()
