from __future__ import annotations

import os
import ssl
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env = Path.home() / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

# SKT 프록시 SSL 우회
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
os.environ["PYTHONHTTPSVERIFY"] = "0"
try:
    import httpx
    _orig = httpx.Client.__init__
    def _patched(self, *a, verify=True, **kw): _orig(self, *a, verify=False, **kw)
    httpx.Client.__init__ = _patched  # type: ignore[method-assign]
    _orig_a = httpx.AsyncClient.__init__
    def _patched_a(self, *a, verify=True, **kw): _orig_a(self, *a, verify=False, **kw)
    httpx.AsyncClient.__init__ = _patched_a  # type: ignore[method-assign]
except ImportError:
    pass

_BASE = Path(__file__).resolve().parent.parent.parent  # nemotron-persona-korea/

# ── 경로 ────────────────────────────────────────────────────────────
JSONL_PATH = _BASE / "Nemotron-Personas-Korea" / "data" / "jsonl" / "personas.jsonl"
CHROMA_DIR = _BASE / "rag" / ".index" / "chroma"

# ── 모델 ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = _BASE / ".venv" / "lib"  # 로컬 캐시 경로 (아래에서 resolve)
EMBEDDING_MODEL_ID = "jhgan/ko-sroberta-multitask"
EMBEDDING_MODEL_CACHE = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--jhgan--ko-sroberta-multitask"
    / "snapshots"
    / "8fca7c9c98c26599be0e14b9916b11a756a26f19"
)

CHROMA_COLLECTION = "personas"

# ── NVIDIA NIM LLM ──────────────────────────────────────────────────
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# 하위 호환성 별칭 — 모든 labs 모듈이 OpenRouter 변수명으로 NIM을 참조
OPENROUTER_API_KEY: str = NVIDIA_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = NIM_BASE_URL
LLM_MODEL = NIM_MODEL

_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or "http://150.2.127.249:9090"
