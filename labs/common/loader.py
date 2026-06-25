"""JSONL → LangChain Document 로딩 유틸 + LangChain Retriever 래퍼."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

# rag 패키지 import를 위해 프로젝트 루트를 경로에 추가
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from labs.common.config import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL_CACHE, JSONL_PATH


def load_jsonl(
    path: Path = JSONL_PATH,
    max_docs: int | None = None,
) -> list[Document]:
    """JSONL → LangChain Document 리스트 변환."""
    docs: list[Document] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
            obj = json.loads(line)
            docs.append(Document(page_content=obj["text"], metadata={
                "id": obj["id"],
                **{k: v for k, v in obj.get("metadata", {}).items()
                   if isinstance(v, (str, int, float, bool))},
            }))
    return docs


def get_chroma_retriever(
    top_k: int = 5,
    where: dict[str, Any] | None = None,
):
    """기존 ChromaDB 인덱스를 LangChain Retriever로 래핑해 반환."""
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_CACHE),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    search_kwargs: dict[str, Any] = {"k": top_k}
    if where:
        search_kwargs["filter"] = where
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
