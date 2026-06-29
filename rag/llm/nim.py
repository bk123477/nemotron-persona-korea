"""
NVIDIA NIM LLM 클라이언트 (OpenAI 호환 API)

모델: nvidia/nemotron-3-super-120b-a12b
엔드포인트: https://integrate.api.nvidia.com/v1
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Generator

import httpx
from openai import OpenAI

from ..schema import SearchResult

if TYPE_CHECKING:
    from ..config import RAGConfig

logger = logging.getLogger(__name__)

_NIM_BASE = "https://integrate.api.nvidia.com/v1"

_SYSTEM_PROMPT = """당신은 한국 인구통계 페르소나 전문가입니다.
아래 [검색된 페르소나] 정보를 바탕으로 사용자의 질문에 한국어로 정확하게 답변하세요.

답변 시 다음을 지켜주세요:
- 검색된 페르소나 데이터에 근거해서 답변하세요
- 없는 정보를 추측하지 마세요
- 페르소나의 인구통계 특성(나이, 성별, 지역, 직업 등)을 구체적으로 활용하세요"""

_SECTION_HEADERS = [
    "Persona",
    "Professional persona",
    "Sports persona",
    "Arts persona",
    "Travel persona",
    "Culinary persona",
    "Family persona",
    "Cultural background",
    "Skills and expertise",
    "Hobbies and interests",
    "Career goals and ambitions",
]

_KEYWORD_SECTION_MAP = {
    "여행": "Travel persona",
    "travel": "Travel persona",
    "운동": "Sports persona",
    "스포츠": "Sports persona",
    "sport": "Sports persona",
    "음식": "Culinary persona",
    "요리": "Culinary persona",
    "culinary": "Culinary persona",
    "가족": "Family persona",
    "family": "Family persona",
    "예술": "Arts persona",
    "art": "Arts persona",
    "직업": "Professional persona",
    "전문": "Professional persona",
    "취미": "Hobbies and interests",
    "hobby": "Hobbies and interests",
}


def _parse_text_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for i, header in enumerate(_SECTION_HEADERS):
        prefix = f"{header}: "
        if prefix not in text:
            continue
        start = text.index(prefix) + len(prefix)
        end = len(text)
        for next_header in _SECTION_HEADERS[i + 1:]:
            marker = f"\n{next_header}: "
            if marker in text[start:]:
                end = text.index(marker, start)
                break
        sections[header] = text[start:end].strip()
    return sections


def _select_sections(sections: dict[str, str], query: str, n: int = 3) -> list[tuple[str, str]]:
    query_lower = query.lower()
    priority: list[str] = []
    for kw, section in _KEYWORD_SECTION_MAP.items():
        if kw in query_lower and section in sections and section not in priority:
            priority.append(section)
    rest = [h for h in _SECTION_HEADERS if h in sections and h not in priority]
    ordered = priority + rest
    return [(h, sections[h]) for h in ordered[:n]]


def _format_context(results: list[SearchResult], query: str = "", max_chars: int = 4000) -> str:
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
        sections = _parse_text_sections(r.text)
        selected = _select_sections(sections, query, n=3)
        body = "\n".join(f"  [{label}] {content[:400]}" for label, content in selected)
        block = f"{header}\n{body}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


class NIMLM:
    """NVIDIA NIM LLM 추론 클라이언트 (OpenAI 호환)."""

    def __init__(self, config: RAGConfig) -> None:
        api_key = config.nvidia_api_key
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY가 없습니다. /home/minkih/.env 파일을 확인하세요."
            )
        _proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY") or "http://150.2.127.249:9090"
        _http_client = httpx.Client(proxy=_proxy, verify=False)

        self._client = OpenAI(
            api_key=api_key,
            base_url=config.nim_base_url,
            http_client=_http_client,
        )
        self._model = config.llm_model
        self._max_tokens = config.llm_max_tokens
        self._temperature = config.llm_temperature
        logger.info("NVIDIA NIM LLM 초기화 (model=%s)", self._model)

    def answer(
        self,
        query: str,
        context: list[SearchResult],
        *,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        ctx_text = _format_context(context, query=query)
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
        )
        return resp.choices[0].message.content or ""

    def _stream(self, messages: list[dict]) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
