from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent  # nemotron-persona-korea/

# .env 자동 로드
try:
    from dotenv import load_dotenv
    _env = _BASE.parent / ".env"  # /home/minkih/.env
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

# SKT 기업 프록시 인증서 경로 (SSL 오류 방지)
_DEFAULT_CERT = Path.home() / "SKtelecom_Pxy.cer"


def suppress_hf_warnings() -> None:
    """HuggingFace Hub의 불필요한 경고/진행 메시지를 숨긴다."""
    import logging
    import warnings

    # env var로 HF Hub 자체 출력 억제
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # "unauthenticated requests" 경고 억제
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    # logging 레벨 설정
    for name in ("huggingface_hub", "sentence_transformers", "transformers", "torch"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # warnings 모듈 필터
    warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")

    # 루트 로거에 필터 추가 (logging 경로로 오는 HF 메시지 차단)
    class _HFFilter(logging.Filter):
        _SKIP = ("unauthenticated", "HF_TOKEN", "rate limit")

        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not any(kw in msg for kw in self._SKIP)

    logging.getLogger().addFilter(_HFFilter())


def setup_ssl(cert_path: Path | None = None) -> None:
    """
    HuggingFace / requests / httpx SSL 설정.

    SKT 프록시 인증서는 BasicConstraints가 non-critical로 마킹되어 있어
    Python 3.12+ (OpenSSL)의 엄격한 RFC 5280 검증을 통과하지 못한다.
    개발 환경에서는 ssl._create_unverified_context 로 우회한다.
    (운영 배포 시에는 IT팀에 RFC 5280 준수 인증서 발급 요청 필요)
    """
    import ssl

    cert = cert_path or _DEFAULT_CERT

    # requests / urllib 계열
    if cert.exists():
        cert_str = str(cert)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_str)
        os.environ.setdefault("CURL_CA_BUNDLE", cert_str)

    # httpx (huggingface_hub >=0.23 사용) — BasicConstraints 문제 우회
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

    # httpx 클라이언트를 verify=False 로 강제
    try:
        import httpx

        _orig_init = httpx.Client.__init__

        def _patched_init(self, *args, verify=True, **kwargs):  # type: ignore[override]
            _orig_init(self, *args, verify=False, **kwargs)

        httpx.Client.__init__ = _patched_init  # type: ignore[method-assign]

        _orig_async_init = httpx.AsyncClient.__init__

        def _patched_async_init(self, *args, verify=True, **kwargs):  # type: ignore[override]
            _orig_async_init(self, *args, verify=False, **kwargs)

        httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore[method-assign]
    except ImportError:
        pass


@dataclass
class RAGConfig:
    # ── 데이터 경로 ────────────────────────────────────────────
    jsonl_path: Path = field(
        default_factory=lambda: _BASE / "Nemotron-Personas-Korea" / "data" / "jsonl" / "personas.jsonl"
    )
    index_dir: Path = field(default_factory=lambda: _BASE / "rag" / ".index")

    # ── Dense 검색 (ChromaDB + sentence-transformers) ──────────
    embedding_model: str = "jhgan/ko-sroberta-multitask"
    chroma_collection: str = "personas"

    # ── Sparse 검색 (BM25) ────────────────────────────────────
    # 476K 전체를 올리면 ~4GB 메모리 필요; 기본 100K로 제한
    max_bm25_docs: int = 100_000
    bm25_tokenizer: str = "whitespace"  # "whitespace" | "char"

    # ── Retrieval top-k ───────────────────────────────────────
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60   # RRF 상수 (표준값)

    # ── Cohere Rerank v3 ──────────────────────────────────────
    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    final_top_k: int = 10

    # ── OpenRouter LLM ────────────────────────────────────────
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    # 무료 모델 기본값: Nemotron-Personas-Korea 데이터셋 생성에 사용된 동일 모델
    llm_model: str = "google/gemma-4-31b-it:free"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.3

    # ── 속성 ─────────────────────────────────────────────────
    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"
