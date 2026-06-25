"""
위키 기반 RAG Q&A 시스템

생성된 위키 마크다운 문서를 ChromaDB에 임베딩하고,
자연어 질문에 위키 내용을 참조해 답한다.

사용법:
    cd /home/minkih/nemotron-persona-korea

    # 최초 1회: 위키 인덱스 구축
    .venv/bin/python3 -m labs.wiki_lab.wiki_rag --build-index

    # 대화형 Q&A
    .venv/bin/python3 -m labs.wiki_lab.wiki_rag

    # 단일 질문
    .venv/bin/python3 -m labs.wiki_lab.wiki_rag --query "서울 청년층 직업 분포는?"

    # 인덱스 재구축 (위키 업데이트 후)
    .venv/bin/python3 -m labs.wiki_lab.wiki_rag --build-index --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import labs.common.config  # noqa: F401
from labs.common.config import (
    CHROMA_DIR, EMBEDDING_MODEL_CACHE,
    LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, _PROXY,
)

_WIKI_DIR = Path(__file__).parent / "wiki"
_WIKI_COLLECTION = "persona-wiki"


# ── 위키 로딩 ────────────────────────────────────────────────────────

def _load_wiki_docs() -> list:
    """wiki/ 하위 모든 .md 파일을 LangChain Document로 로드한다."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    md_files = list(_WIKI_DIR.rglob("*.md"))
    if not md_files:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " "],
    )

    docs = []
    for path in md_files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # 상대 경로에서 카테고리/파일명 추출
        rel = path.relative_to(_WIKI_DIR)
        parts = rel.parts
        category = parts[0] if len(parts) > 1 else "general"
        title = path.stem

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": str(rel),
                    "category": category,
                    "title": title,
                    "chunk": i,
                },
            ))

    return docs


# ── 인덱스 구축 ──────────────────────────────────────────────────────

def build_index(force: bool = False) -> None:
    """위키 문서를 ChromaDB에 임베딩 후 저장한다."""
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import chromadb

    # 기존 컬렉션 확인
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]
    if _WIKI_COLLECTION in existing:
        if not force:
            existing_col = client.get_collection(_WIKI_COLLECTION)
            count = existing_col.count()
            print(f"[인덱스 존재] {_WIKI_COLLECTION}: {count}개 청크")
            print("  재구축하려면 --build-index --force 를 사용하세요.")
            return
        else:
            print(f"[인덱스 삭제] 기존 {_WIKI_COLLECTION} 컬렉션 제거")
            client.delete_collection(_WIKI_COLLECTION)

    docs = _load_wiki_docs()
    if not docs:
        print(f"[오류] 위키 문서가 없습니다: {_WIKI_DIR}")
        print("먼저 generate_wiki.py를 실행하세요:")
        print("  .venv/bin/python3 -m labs.wiki_lab.generate_wiki --dry-run")
        return

    print(f"[임베딩] {len(docs)}개 청크 처리 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_CACHE),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=_WIKI_COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"[완료] {len(docs)}개 청크 → ChromaDB '{_WIKI_COLLECTION}'")


# ── 검색 ────────────────────────────────────────────────────────────

def get_wiki_retriever(top_k: int = 4):
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_CACHE),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=_WIKI_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


# ── RAG 체인 ────────────────────────────────────────────────────────

def build_wiki_chain():
    """위키 검색 → LLM 답변 LCEL 체인을 반환한다."""
    import httpx
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_openai import ChatOpenAI

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY가 없습니다.")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        http_client=httpx.Client(proxy=_PROXY, verify=False),
        temperature=0.2,
        max_tokens=700,
    )

    retriever = get_wiki_retriever(top_k=4)

    def _format_context(docs) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "?")
            cat = doc.metadata.get("category", "?")
            parts.append(f"[{i}] [{cat}/{title}]\n{doc.page_content}")
        return "\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 한국 인구통계 데이터 전문가입니다. "
         "아래 위키 참고 문서를 바탕으로 질문에 정확하고 간결하게 답하세요.\n\n"
         "위키 참고 문서:\n{context}\n\n"
         "참고 문서에 없는 내용은 '위키에 해당 정보가 없습니다'라고 답하세요."),
        ("human", "{question}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_context(retriever.invoke(x["question"])))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── CLI ─────────────────────────────────────────────────────────────

def _check_index() -> bool:
    """위키 인덱스 존재 여부를 확인한다."""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        cols = [c.name for c in client.list_collections()]
        if _WIKI_COLLECTION not in cols:
            return False
        return client.get_collection(_WIKI_COLLECTION).count() > 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="위키 기반 RAG Q&A")
    parser.add_argument("--build-index", action="store_true", help="위키 인덱스 구축")
    parser.add_argument("--force", action="store_true", help="인덱스 강제 재구축")
    parser.add_argument("--query", type=str, default=None, help="단일 질문 후 종료")
    parser.add_argument("--show-sources", action="store_true", help="참조 소스 출력")
    args = parser.parse_args()

    if args.build_index:
        build_index(force=args.force)
        if not args.query:
            return

    if not _check_index():
        print("[오류] 위키 인덱스가 없습니다.")
        print("먼저 다음을 순서대로 실행하세요:")
        print("  1. .venv/bin/python3 -m labs.wiki_lab.generate_wiki --dry-run")
        print("  2. .venv/bin/python3 -m labs.wiki_lab.wiki_rag --build-index")
        sys.exit(1)

    print("[위키 RAG 초기화 중...]")
    chain, retriever = build_wiki_chain()

    if args.query:
        answer = chain.invoke({"question": args.query})
        print(f"\nQ: {args.query}")
        print(f"A: {answer}")
        if args.show_sources:
            docs = retriever.invoke(args.query)
            print("\n[참조 소스]")
            for doc in docs:
                print(f"  - {doc.metadata['category']}/{doc.metadata['title']}")
        return

    # 대화형 모드
    print("\n한국 인구통계 위키 Q&A를 시작합니다. (종료: q)\n" + "─" * 55)
    print("예시 질문:")
    print("  - 서울 청년층의 주요 직업은?")
    print("  - 고령층 페르소나의 혼인 상태 분포는?")
    print("  - 부산과 경남의 직업 특성 차이는?")
    print("─" * 55)

    try:
        while True:
            question = input("\nQ> ").strip()
            if not question or question.lower() in ("q", "quit", "exit"):
                break

            answer = chain.invoke({"question": question})
            print(f"\nA> {answer}")

            if args.show_sources:
                docs = retriever.invoke(question)
                print("\n  [참조]", ", ".join(
                    f"{d.metadata['category']}/{d.metadata['title']}"
                    for d in docs
                ))
    except (KeyboardInterrupt, EOFError):
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
