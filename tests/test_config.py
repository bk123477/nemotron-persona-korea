"""
단위 테스트: RAGConfig — NIM 설정 검증
"""
from __future__ import annotations

import pytest


def test_ragconfig_nim_base_url():
    from rag.config import RAGConfig
    config = RAGConfig()
    assert config.nim_base_url == "https://integrate.api.nvidia.com/v1"


def test_ragconfig_llm_model_is_nim():
    from rag.config import RAGConfig
    config = RAGConfig()
    assert config.llm_model == "nvidia/nemotron-3-super-120b-a12b"


def test_ragconfig_nvidia_api_key_from_env(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvapi-key")
    from importlib import reload
    import rag.config as cfg_module
    reload(cfg_module)
    config = cfg_module.RAGConfig()
    assert config.nvidia_api_key == "test-nvapi-key"


def test_ragconfig_openrouter_compat_falls_back_to_nvidia(monkeypatch):
    """openrouter_api_key는 NVIDIA_API_KEY를 우선 사용해야 한다."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from importlib import reload
    import rag.config as cfg_module
    reload(cfg_module)
    config = cfg_module.RAGConfig()
    assert config.openrouter_api_key == "nvapi-test"


def test_ragconfig_default_top_k():
    from rag.config import RAGConfig
    config = RAGConfig()
    assert config.dense_top_k == 20
    assert config.sparse_top_k == 20
    assert config.final_top_k == 10
    assert config.rrf_k == 60


def test_ragconfig_chroma_dir_property():
    from rag.config import RAGConfig
    config = RAGConfig()
    assert config.chroma_dir == config.index_dir / "chroma"


def test_ragconfig_bm25_path_property():
    from rag.config import RAGConfig
    config = RAGConfig()
    assert config.bm25_path == config.index_dir / "bm25.pkl"


def test_labs_common_config_nim_vars():
    from labs.common.config import (
        NIM_BASE_URL, NIM_MODEL, NVIDIA_API_KEY,
        OPENROUTER_BASE_URL, LLM_MODEL,
    )
    assert NIM_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert NIM_MODEL == "nvidia/nemotron-3-super-120b-a12b"
    # 하위 호환성 별칭이 NIM을 가리켜야 함
    assert OPENROUTER_BASE_URL == NIM_BASE_URL
    assert LLM_MODEL == NIM_MODEL
