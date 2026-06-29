"""
단위 테스트: NVIDIA NIM LLM 클라이언트
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_config(api_key: str = "nvapi-test") -> "RAGConfig":
    from rag.config import RAGConfig
    config = RAGConfig()
    config.nvidia_api_key = api_key
    config.nim_base_url = "https://integrate.api.nvidia.com/v1"
    config.llm_model = "nvidia/nemotron-3-super-120b-a12b"
    config.llm_max_tokens = 512
    config.llm_temperature = 0.3
    return config


class TestNIMLMInit:
    def test_raises_without_api_key(self):
        from rag.llm.nim import NIMLM
        from rag.config import RAGConfig
        config = RAGConfig()
        config.nvidia_api_key = ""
        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            NIMLM(config)

    def test_initializes_with_valid_key(self):
        from rag.llm.nim import NIMLM
        with patch("rag.llm.nim.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            nim = NIMLM(_make_config("nvapi-test-key"))
            assert nim._model == "nvidia/nemotron-3-super-120b-a12b"
            assert nim._max_tokens == 512
            assert nim._temperature == 0.3

    def test_uses_nim_base_url(self):
        from rag.llm.nim import NIMLM
        with patch("rag.llm.nim.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            NIMLM(_make_config("nvapi-test-key"))
            call_kwargs = mock_openai.call_args[1]
            assert "integrate.api.nvidia.com" in call_kwargs.get("base_url", "")


class TestNIMLMAnswer:
    def _make_nim(self):
        from rag.llm.nim import NIMLM
        with patch("rag.llm.nim.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            nim = NIMLM(_make_config())
            nim._client = mock_client
        return nim

    def test_answer_returns_string(self, sample_search_results):
        from rag.llm.nim import NIMLM
        with patch("rag.llm.nim.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "테스트 답변입니다."
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            nim = NIMLM(_make_config())
            nim._client = mock_client
            result = nim.answer("서울 청년층 페르소나는?", sample_search_results)

        assert isinstance(result, str)
        assert result == "테스트 답변입니다."

    def test_answer_stream_yields_tokens(self, sample_search_results):
        from rag.llm.nim import NIMLM
        with patch("rag.llm.nim.OpenAI") as mock_openai:
            mock_client = MagicMock()
            tokens = ["안녕", "하세", "요"]
            chunks = []
            for t in tokens:
                chunk = MagicMock()
                chunk.choices[0].delta.content = t
                chunks.append(chunk)
            mock_client.chat.completions.create.return_value = iter(chunks)
            mock_openai.return_value = mock_client

            nim = NIMLM(_make_config())
            nim._client = mock_client
            result = list(nim.answer("테스트", sample_search_results, stream=True))

        assert result == tokens

    def test_format_context_uses_metadata(self, sample_search_results):
        from rag.llm.nim import _format_context
        ctx = _format_context(sample_search_results[:2], query="서울")
        assert "서울" in ctx or "부산" in ctx  # 메타데이터 province 포함
        assert "페르소나 1" in ctx


class TestNIMLMSectionParsing:
    def test_parse_text_sections(self):
        from rag.llm.nim import _parse_text_sections
        text = "Persona: 기본 설명입니다.\nProfessional persona: 직업 설명입니다.\n"
        sections = _parse_text_sections(text)
        assert "Persona" in sections
        assert "Professional persona" in sections
        assert "기본 설명입니다." in sections["Persona"]

    def test_select_sections_with_keyword(self):
        from rag.llm.nim import _select_sections
        sections = {
            "Persona": "기본",
            "Travel persona": "여행 좋아함",
            "Sports persona": "운동함",
        }
        selected = _select_sections(sections, query="여행 관련 질문", n=2)
        headers = [h for h, _ in selected]
        assert "Travel persona" in headers
        assert headers[0] == "Travel persona"  # 키워드 매칭 섹션이 우선
