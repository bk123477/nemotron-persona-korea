"""
OpenRouter LLM 클라이언트 (OpenAI 호환 API)

무료 모델 목록 (2026-06 기준):
  google/gemma-4-31b-it:free          ctx=262K  ← 기본값, Nemotron 데이터셋 생성 모델
  nvidia/nemotron-3-super-120b-a12b:free  ctx=1M
  meta-llama/llama-3.3-70b-instruct:free ctx=131K
  qwen/qwen3-coder:free               ctx=1M
  openrouter/free                     ctx=200K  (자동 라우팅)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generator

import httpx
import os

import httpx
from openai import OpenAI

from ..schema import SearchResult

if TYPE_CHECKING:
    from ..config import RAGConfig

logger = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# 페르소나 RAG용 시스템 프롬프트
_SYSTEM_PROMPT = """당신은 한국 인구통계 페르소나 전문가입니다.
아래 [검색된 페르소나] 정보를 바탕으로 사용자의 질문에 한국어로 정확하게 답변하세요.

답변 시 다음을 지켜주세요:
- 검색된 페르소나 데이터에 근거해서 답변하세요
- 없는 정보를 추측하지 마세요
- 페르소나의 인구통계 특성(나이, 성별, 지역, 직업 등)을 구체적으로 활용하세요"""


def _format_context(results: list[SearchResult], max_chars: int = 3000) -> str:
    """검색 결과를 LLM 컨텍스트 텍스트로 변환."""
    parts = []
    total = 0
    for i, r in enumerate(results):
        meta = r.metadata
        header = (
            f"[페르소나 {i+1}] "
            f"{meta.get('sex','?')} / {meta.get('age','?')}세 / "
            f"{meta.get('province','?')} / {meta.get('occupation','?')} / "
            f"학력={meta.get('education_level','?')} / 혼인={meta.get('marital_status','?')}"
        )
        snippet = r.text[:500]
        block = f"{header}\n{snippet}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


class OpenRouterLLM:
    """OpenRouter를 통한 LLM 추론 클라이언트."""

    def __init__(self, config: RAGConfig) -> None:
        if not config.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY가 없습니다. /home/minkih/.env 파일을 확인하세요."
            )
        # SKT 프록시 환경: httpx 클라이언트에 프록시 + SSL 비검증 설정
        _proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY") or "http://150.2.127.249:9090"
        _http_client = httpx.Client(proxy=_proxy, verify=False)

        self._client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=_OPENROUTER_BASE,
            http_client=_http_client,
        )
        self._model = config.llm_model
        self._max_tokens = config.llm_max_tokens
        self._temperature = config.llm_temperature
        logger.info("OpenRouter LLM 초기화 (model=%s)", self._model)

    def answer(
        self,
        query: str,
        context: list[SearchResult],
        *,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        RAG 답변 생성.

        Args:
            query  : 사용자 질문
            context: 검색된 페르소나 결과 리스트
            stream : True면 토큰 스트리밍 Generator 반환

        Returns:
            stream=False → 완성된 답변 문자열
            stream=True  → 토큰 청크 Generator
        """
        ctx_text = _format_context(context)
        user_msg = f"[검색된 페르소나]\n{ctx_text}\n\n[질문]\n{query}"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        if stream:
            return self._stream(messages)
        return self._complete(messages)

    def _complete(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            extra_headers={"HTTP-Referer": "https://github.com/nemotron-persona-korea"},
        )
        return resp.choices[0].message.content or ""

    def _stream(self, messages: list[dict]) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
            extra_headers={"HTTP-Referer": "https://github.com/nemotron-persona-korea"},
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
