# 테스트 실행 결과 상세 보고서

**실행일**: 2026-06-30  
**환경**: Python 3.14.4 / pytest 9.1.1 / Linux (WSL2)  
**결과**: **63 passed, 1 warning, 0 failed** (1.78초)

---

## 목차

1. [실행 명령어 및 전체 결과](#1-실행-명령어-및-전체-결과)
2. [테스트 파일별 상세 분석](#2-테스트-파일별-상세-분석)
   - [test_config.py — NIM 설정 검증 (8개)](#21-test_configpy--nim-설정-검증-8개)
   - [test_nim_llm.py — NIM LLM 클라이언트 (8개)](#22-test_nim_llmpy--nim-llm-클라이언트-8개)
   - [test_ontology.py — 온톨로지 스키마·인구사·SPARQL (18개)](#23-test_ontologypy--온톨로지-스키마인구사sparql-18개)
   - [test_parquet_loader.py — JSONL 변환 (5개)](#24-test_parquet_loaderpy--jsonl-변환-5개)
   - [test_pipeline.py — RAG 파이프라인 내부 로직 (10개)](#25-test_pipelinepy--rag-파이프라인-내부-로직-10개)
   - [test_retrieval.py — RRF·메타데이터 필터 (14개)](#26-test_retrievalpy--rrf메타데이터-필터-14개)
3. [공통 픽스처(conftest.py) 설명](#3-공통-픽스처conftestpy-설명)
4. [경고 1건 분석](#4-경고-1건-분석)
5. [테스트 설계 원칙](#5-테스트-설계-원칙)

---

## 1. 실행 명령어 및 전체 결과

```bash
$ cd /home/minkih/nemotron-persona-korea
$ .venv/bin/python3 -m pytest tests/ -v --tb=short
```

| 옵션 | 의미 |
|------|------|
| `tests/` | `tests/` 디렉토리 안의 `test_*.py` 파일을 모두 수집 |
| `-v` | 테스트별 PASSED/FAILED 결과를 한 줄씩 출력 (verbose) |
| `--tb=short` | 실패 시 트레이스백을 간결하게 출력 (전체 스택 아닌 핵심만) |

**전체 출력 (요약)**

```
platform linux -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
plugins: langsmith-0.9.1, anyio-4.14.1, asyncio-1.4.0
asyncio: mode=Mode.STRICT

tests/test_config.py::test_ragconfig_nim_base_url                       PASSED
tests/test_config.py::test_ragconfig_llm_model_is_nim                   PASSED
tests/test_config.py::test_ragconfig_nvidia_api_key_from_env            PASSED
tests/test_config.py::test_ragconfig_openrouter_compat_falls_back_to_nvidia PASSED
tests/test_config.py::test_ragconfig_default_top_k                      PASSED
tests/test_config.py::test_ragconfig_chroma_dir_property                PASSED
tests/test_config.py::test_ragconfig_bm25_path_property                 PASSED
tests/test_config.py::test_labs_common_config_nim_vars                  PASSED
tests/test_nim_llm.py::TestNIMLMInit::test_raises_without_api_key       PASSED
tests/test_nim_llm.py::TestNIMLMInit::test_initializes_with_valid_key   PASSED
tests/test_nim_llm.py::TestNIMLMInit::test_uses_nim_base_url            PASSED
tests/test_nim_llm.py::TestNIMLMAnswer::test_answer_returns_string      PASSED
tests/test_nim_llm.py::TestNIMLMAnswer::test_answer_stream_yields_tokens PASSED
tests/test_nim_llm.py::TestNIMLMAnswer::test_format_context_uses_metadata PASSED
tests/test_nim_llm.py::TestNIMLMSectionParsing::test_parse_text_sections PASSED
tests/test_nim_llm.py::TestNIMLMSectionParsing::test_select_sections_with_keyword PASSED
tests/test_ontology.py::TestSchema::test_build_schema_returns_graph     PASSED
tests/test_ontology.py::TestSchema::test_persona_class_defined          PASSED
tests/test_ontology.py::TestSchema::test_province_subclass_of_geographic PASSED
tests/test_ontology.py::TestSchema::test_district_subclass_of_geographic PASSED
tests/test_ontology.py::TestSchema::test_sex_instances_created          PASSED
tests/test_ontology.py::TestSchema::test_age_group_instances_created    PASSED
tests/test_ontology.py::TestSchema::test_object_properties_have_domain_range PASSED
tests/test_ontology.py::TestSchema::test_has_age_is_datatype_property   PASSED
tests/test_ontology.py::TestPopulate::test_populate_small_sample        PASSED
tests/test_ontology.py::TestPopulate::test_persona_has_age              PASSED
tests/test_ontology.py::TestPopulate::test_persona_has_sex              PASSED
tests/test_ontology.py::TestPopulate::test_province_instances_created   PASSED
tests/test_ontology.py::TestPopulate::test_output_file_is_valid_turtle  PASSED
tests/test_ontology.py::TestPopulate::test_max_docs_limit               PASSED
tests/test_ontology.py::TestSPARQLQueries::test_query_1_province_count  PASSED
tests/test_ontology.py::TestSPARQLQueries::test_query_2_occupation_avg_age PASSED
tests/test_ontology.py::TestSPARQLQueries::test_query_4_sex_age_distribution PASSED
tests/test_ontology.py::TestSPARQLQueries::test_load_graph_raises_on_missing_file PASSED
tests/test_ontology.py::TestSPARQLQueries::test_run_query_returns_list  PASSED
tests/test_ontology.py::TestSPARQLQueries::test_run_query_invalid_number_raises PASSED
tests/test_parquet_loader.py::test_export_jsonl_creates_file            PASSED
tests/test_parquet_loader.py::test_jsonl_record_structure               PASSED
tests/test_parquet_loader.py::test_jsonl_metadata_types                 PASSED
tests/test_parquet_loader.py::test_jsonl_all_records                    PASSED
tests/test_parquet_loader.py::test_persona_document_fields              PASSED
tests/test_pipeline.py::TestToChromaWhere::test_simple_eq               PASSED
tests/test_pipeline.py::TestToChromaWhere::test_removes_contains        PASSED
tests/test_pipeline.py::TestToChromaWhere::test_mixed_keeps_supported   PASSED
tests/test_pipeline.py::TestToChromaWhere::test_gte_lte_preserved       PASSED
tests/test_pipeline.py::TestToChromaWhere::test_and_with_two_conditions PASSED
tests/test_pipeline.py::TestToChromaWhere::test_empty_condition_returns_none PASSED
tests/test_pipeline.py::TestApplyMetadataFilter::test_basic_filtering   PASSED
tests/test_pipeline.py::TestApplyMetadataFilter::test_nested_and_or     PASSED
tests/test_pipeline.py::TestApplyMetadataFilter::test_lte_filter        PASSED
tests/test_pipeline.py::TestApplyMetadataFilter::test_ne_filter         PASSED
tests/test_retrieval.py::TestRRF::test_basic_merge                      PASSED
tests/test_retrieval.py::TestRRF::test_overlap_boosts_rank              PASSED
tests/test_retrieval.py::TestRRF::test_top_k_limit                      PASSED
tests/test_retrieval.py::TestRRF::test_empty_sparse                     PASSED
tests/test_retrieval.py::TestRRF::test_rank_assigned_sequentially       PASSED
tests/test_retrieval.py::TestMetadataFilter::test_eq_filter             PASSED
tests/test_retrieval.py::TestMetadataFilter::test_gte_filter            PASSED
tests/test_retrieval.py::TestMetadataFilter::test_contains_filter       PASSED
tests/test_retrieval.py::TestMetadataFilter::test_and_filter            PASSED
tests/test_retrieval.py::TestMetadataFilter::test_or_filter             PASSED
tests/test_retrieval.py::TestMetadataFilter::test_to_chroma_where_strips_contains PASSED
tests/test_retrieval.py::TestMetadataFilter::test_to_chroma_where_keeps_eq PASSED

======================== 63 passed, 1 warning in 1.78s =========================
```

**전체 63개 통과, 실패 0건.**

---

## 2. 테스트 파일별 상세 분석

---

### 2.1 `test_config.py` — NIM 설정 검증 (8개)

**목적**: `rag/config.py`의 `RAGConfig`와 `labs/common/config.py`가 NVIDIA NIM 엔드포인트로 올바르게 설정되어 있는지 확인한다.

#### 테스트 목록과 동작 원리

| # | 테스트 이름 | 검증 내용 | 동작 방식 |
|---|------------|----------|----------|
| 1 | `test_ragconfig_nim_base_url` | `config.nim_base_url == "https://integrate.api.nvidia.com/v1"` | `RAGConfig()` 기본 인스턴스 생성 후 필드 확인 |
| 2 | `test_ragconfig_llm_model_is_nim` | `config.llm_model == "nvidia/nemotron-3-super-120b-a12b"` | 모델 이름이 NIM 모델로 고정되어 있는지 확인 |
| 3 | `test_ragconfig_nvidia_api_key_from_env` | 환경변수 `NVIDIA_API_KEY`가 config에 반영되는지 | `monkeypatch.setenv()`로 환경변수 주입 후 모듈 `reload()` |
| 4 | `test_ragconfig_openrouter_compat_falls_back_to_nvidia` | `openrouter_api_key`가 NVIDIA 키를 우선 사용하는지 | OPENROUTER_API_KEY 없을 때 NVIDIA_API_KEY로 폴백 |
| 5 | `test_ragconfig_default_top_k` | dense/sparse/final top_k, RRF k 기본값 확인 | `dense_top_k=20`, `sparse_top_k=20`, `final_top_k=10`, `rrf_k=60` |
| 6 | `test_ragconfig_chroma_dir_property` | `config.chroma_dir == config.index_dir / "chroma"` | `@property`가 `index_dir` 기반 경로를 반환하는지 확인 |
| 7 | `test_ragconfig_bm25_path_property` | `config.bm25_path == config.index_dir / "bm25.pkl"` | BM25 인덱스 경로 프로퍼티 검증 |
| 8 | `test_labs_common_config_nim_vars` | `labs.common.config`의 NIM 상수 및 하위호환 별칭 | `OPENROUTER_BASE_URL == NIM_BASE_URL`, `LLM_MODEL == NIM_MODEL` |

**왜 이런 테스트가 필요한가?**

기존 코드베이스는 OpenRouter API를 사용했다. NIM으로 마이그레이션하면서 기존 labs 파일들이 `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `LLM_MODEL`을 임포트하기 때문에, 이 별칭들이 NIM 값을 가리키는지 자동으로 검증해야 한다. 테스트 3번에서 `monkeypatch` + `reload()`를 쓰는 이유는 Python이 환경변수를 모듈 임포트 시점에 읽어 `field(default_factory=...)` 클로저에 캡처하기 때문이다 — 환경변수만 바꿔도 이미 로드된 모듈은 변경되지 않으므로 강제로 재로드해야 한다.

---

### 2.2 `test_nim_llm.py` — NIM LLM 클라이언트 (8개)

**목적**: `rag/llm/nim.py`의 `NIMLM` 클래스가 OpenAI 호환 NIM 엔드포인트와 올바르게 통신하는지, 모킹(mocking)으로 실제 API 호출 없이 검증한다.

#### 클래스별 테스트 설명

**`TestNIMLMInit` (초기화 3개)**

| # | 테스트 | 검증 | 이유 |
|---|--------|------|------|
| 1 | `test_raises_without_api_key` | `NVIDIA_API_KEY=""` 시 `ValueError` 발생 | API 키 없는 상태로 LLM 호출 시도를 방지 |
| 2 | `test_initializes_with_valid_key` | 유효한 키로 초기화 시 `_model`, `_max_tokens`, `_temperature` 설정 확인 | `patch("rag.llm.nim.OpenAI")`로 실제 HTTP 연결 없이 테스트 |
| 3 | `test_uses_nim_base_url` | `OpenAI` 생성자에 `integrate.api.nvidia.com`이 포함된 `base_url`이 전달되는지 | NIM 엔드포인트 URL이 하드코딩되지 않고 config에서 읽혀야 하기 때문 |

**`TestNIMLMAnswer` (답변 생성 3개)**

| # | 테스트 | 검증 | 이유 |
|---|--------|------|------|
| 4 | `test_answer_returns_string` | `nim.answer(query, results)` → `str` 반환 | `MagicMock`으로 `completions.create()`의 반환값을 설정해 실제 LLM 응답 흐름을 시뮬레이션 |
| 5 | `test_answer_stream_yields_tokens` | `stream=True` 시 토큰 단위 `Generator` 반환 | 스트리밍 시 각 `chunk.choices[0].delta.content`를 yield해야 함을 검증 |
| 6 | `test_format_context_uses_metadata` | `_format_context()` 출력에 province 정보 포함 여부 | 컨텍스트 포매터가 메타데이터(성별, 지역 등)를 LLM 프롬프트에 삽입하는지 확인 |

**`TestNIMLMSectionParsing` (텍스트 파싱 2개)**

| # | 테스트 | 검증 | 이유 |
|---|--------|------|------|
| 7 | `test_parse_text_sections` | `Persona: ...`, `Professional persona: ...` 섹션 파싱 | 데이터셋의 7개 페르소나 필드가 헤더 기준으로 분리되어야 함 |
| 8 | `test_select_sections_with_keyword` | 쿼리 키워드와 매칭되는 섹션이 최상위에 정렬되는지 | "여행" 질의 → `Travel persona` 섹션이 먼저 선택 (관련도 높은 컨텍스트 우선 제공) |

**모킹 전략**

```python
with patch("rag.llm.nim.OpenAI") as mock_openai:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "테스트 답변입니다."
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
```

`patch()`로 `OpenAI` 클래스 자체를 대체하기 때문에 실제 HTTP 요청이 발생하지 않으며, 테스트가 외부 API 의존성 없이 1.78초 내에 완료된다.

---

### 2.3 `test_ontology.py` — 온톨로지 스키마·인구사·SPARQL (18개)

**목적**: Stage 5(온톨로지)의 세 모듈(`schema.py`, `populate.py`, `query.py`)을 독립적으로 검증한다.

#### `TestSchema` — OWL 스키마 (8개)

| # | 테스트 | 검증 내용 | 동작 원리 |
|---|--------|----------|----------|
| 1 | `test_build_schema_returns_graph` | `build_schema()` → 비어있지 않은 `rdflib.Graph` 반환 | OWL 스키마 생성 함수 호출 후 `len(g) > 0` 확인 |
| 2 | `test_persona_class_defined` | `nemo:Persona` 클래스가 그래프에 존재 | `(NEMO.Persona, RDF.type, None)`으로 트리플 조회 |
| 3 | `test_province_subclass_of_geographic` | `Province rdfs:subClassOf Geographic` | 시도(Province)가 지리 상위 클래스의 하위클래스인지 |
| 4 | `test_district_subclass_of_geographic` | `District rdfs:subClassOf Geographic` | 시군구(District)도 동일한 계층 구조 |
| 5 | `test_sex_instances_created` | `data:sex_남자`, `data:sex_여자`가 `SexCategory` 타입 | 스키마에 고정 인스턴스가 사전 정의되어 있는지 |
| 6 | `test_age_group_instances_created` | youth/middle/senior/elderly 4개 AgeGroup 인스턴스 존재 | 연령대가 이산 URI로 미리 정의되어야 SPARQL 필터링 가능 |
| 7 | `test_object_properties_have_domain_range` | `hasSex`, `hasOccupation`, `hasEducation`에 domain 설정 | OWL ObjectProperty의 도메인이 Persona로 제한됨을 검증 |
| 8 | `test_has_age_is_datatype_property` | `nemo:hasAge`가 `owl:DatatypeProperty` 타입 | 나이는 리터럴(xsd:integer) 값이므로 DataProperty여야 함 |

**왜 이 계층 구조인가?**  
`Province`와 `District`를 별도 클래스로 만들고 `Geographic`의 하위클래스로 설계한 이유는 SPARQL에서 `?region a nemo:Geographic` 같은 상위 클래스 필터로 시도·시군구를 동시에 조회하기 위함이다.

#### `TestPopulate` — RDF 트리플 생성 (6개)

| # | 테스트 | 검증 내용 | 동작 원리 |
|---|--------|----------|----------|
| 1 | `test_populate_small_sample` | 10건 → 정확히 10개의 `nemo:Persona` 트리플 생성 | `sample_jsonl` 픽스처(임시 10건 JSONL) + `tmp_path` 사용 |
| 2 | `test_persona_has_age` | 각 Persona에 `nemo:hasAge` 트리플 존재 | `g.objects(p, NEMO.hasAge)` 결과가 비어있지 않아야 함 |
| 3 | `test_persona_has_sex` | 하나 이상의 Persona에 `nemo:hasSex` 연결 | 성별 정보가 SexCategory URI와 연결되었는지 |
| 4 | `test_province_instances_created` | `nemo:Province` 타입의 URI가 하나 이상 생성 | 시도 URI가 카테고리 캐시를 통해 생성·재사용되는지 |
| 5 | `test_output_file_is_valid_turtle` | 출력된 `.ttl` 파일을 다시 파싱해도 오류 없음 | `g2.parse(str(out), format="turtle")` 로 재파싱 |
| 6 | `test_max_docs_limit` | `max_docs=3` 시 정확히 3개의 Persona만 생성 | 스트리밍 변환 루프의 중단 조건 검증 |

**`tmp_path`와 `sample_jsonl`을 쓰는 이유**  
실제 데이터셋(1M 레코드)을 테스트에 사용하면 수분 이상 소요된다. `sample_jsonl` 픽스처는 10건의 가짜 JSONL을 `tmp_path`(pytest 임시 디렉토리)에 생성하므로, 테스트가 격리되고 빠르게 실행된다.

#### `TestSPARQLQueries` — SPARQL 실행 (6개)

| # | 테스트 | 검증 내용 | 동작 원리 |
|---|--------|----------|----------|
| 1 | `test_query_1_province_count` | 시도별 페르소나 수의 합 == 10 | `QUERIES[1]` 실행 후 count 컬럼 합산 |
| 2 | `test_query_2_occupation_avg_age` | 직업별 평균 나이 쿼리가 결과 반환 | 10건 중 직업 데이터가 있어야 하므로 `len(rows) > 0` |
| 3 | `test_query_4_sex_age_distribution` | 성별×연령대 분포 쿼리 결과 존재 | GROUP BY 쿼리의 기본 동작 확인 |
| 4 | `test_load_graph_raises_on_missing_file` | 없는 파일 경로 → `FileNotFoundError` | 에러 처리 경로 검증 |
| 5 | `test_run_query_returns_list` | `run_query(g, 1)` → `list` 타입 반환 | SPARQL 결과가 파이썬 리스트로 변환되는지 |
| 6 | `test_run_query_invalid_number_raises` | 없는 쿼리 번호 → `KeyError` | `QUERIES[9999]` 딕셔너리 키 누락 시 명확한 에러 |

---

### 2.4 `test_parquet_loader.py` — JSONL 변환 (5개)

**목적**: `scripts/parquet_loader.py`가 생성하는 JSONL 포맷이 다운스트림 모듈(RAG, 온톨로지)에서 기대하는 스키마를 충족하는지 확인한다.

| # | 테스트 | 검증 내용 | 동작 원리 |
|---|--------|----------|----------|
| 1 | `test_export_jsonl_creates_file` | JSONL 파일 파싱 시 10건 모두 읽힘 | `sample_jsonl` 픽스처를 직접 파싱해 레코드 수 확인 |
| 2 | `test_jsonl_record_structure` | 각 레코드에 `text`, `metadata` 필드 존재 | 첫 번째 레코드만 읽어 필드 이름 확인 |
| 3 | `test_jsonl_metadata_types` | `metadata.age`가 `int` 또는 `float` | ChromaDB 메타데이터 필터는 숫자형을 요구하므로 타입 검증 필요 |
| 4 | `test_jsonl_all_records` | 10건 전체가 유효한 JSON으로 파싱됨 | `json.loads(line)` 호출이 모두 성공해야 함 |
| 5 | `test_persona_document_fields` | 필수 5개 필드(`age`, `sex`, `province`, `occupation`, `education_level`) 존재 | 모든 레코드 순회하며 필드 누락 확인 |

**왜 `age`의 타입을 별도로 검증하는가?**  
ChromaDB의 `where` 필터(`$gte`, `$lte`)는 숫자형 메타데이터에만 작동한다. Parquet에서 변환 시 age가 문자열로 변환되면 `{"age": {"$gte": 60}}` 같은 필터가 조용히 실패한다. 이 테스트가 타입 계약을 명시적으로 보호한다.

---

### 2.5 `test_pipeline.py` — RAG 파이프라인 내부 로직 (10개)

**목적**: `rag/pipeline.py`의 두 핵심 함수 `_to_chroma_where()`와 `_apply_metadata_filter()`의 동작을 검증한다.

#### `TestToChromaWhere` — ChromaDB 필터 변환 (6개)

`_to_chroma_where()`는 범용 메타데이터 필터 딕셔너리를 ChromaDB가 이해하는 `where` 형식으로 변환하는 함수다. **ChromaDB는 `$contains` 연산자를 지원하지 않으므로** 이 함수가 필터를 정제한다.

| # | 입력 | 기대 출력 | 이유 |
|---|------|----------|------|
| 1 | `{"sex": "여자"}` | `{"sex": "여자"}` | 단순 동등 비교는 그대로 통과 |
| 2 | `{"occupation": {"$contains": "전문"}}` | `None` | `$contains`만 있으면 ChromaDB에 전달할 필터 없음 |
| 3 | `{"$and": [{"sex": "여자"}, {"occupation": {"$contains": "전문"}}]}` | `{"sex": "여자"}` | `$contains` 항목 제거 후 단일 항목이면 `$and` 래퍼도 unwrap |
| 4 | `{"age": {"$gte": 60, "$lte": 80}}` | 동일 그대로 | `$gte`, `$lte`는 ChromaDB가 지원하는 연산자 |
| 5 | `{"$and": [{"sex": "여자"}, {"province": "서울"}]}` | 동일 그대로 | 두 조건 모두 유효하면 `$and` 유지 |
| 6 | `{"occupation": {"$contains": "x"}}` | `None` | 지원 연산자 없으면 None |

**왜 `$contains`를 별도 처리하는가?**  
Hybrid RAG 파이프라인은 Dense 검색(ChromaDB)과 Sparse 검색(BM25)를 병렬로 실행한다. Dense 검색에서는 ChromaDB `where` 필터로 사전 필터링하고, Sparse 결과를 포함한 최종 결과에는 `_apply_metadata_filter()`가 `$contains`를 처리한다. ChromaDB에 `$contains`를 전달하면 에러가 발생하므로 이 변환이 필수다.

#### `TestApplyMetadataFilter` — 후처리 필터 (4개)

`_apply_metadata_filter()`는 RRF 퓨전 이후 검색 결과를 파이썬 레벨에서 필터링한다. ChromaDB 제약 없이 모든 연산자를 처리한다.

| # | 테스트 | 검증 내용 |
|---|--------|----------|
| 1 | `test_basic_filtering` | 단순 `{"province": "서울"}` → 서울 결과만 반환 |
| 2 | `test_nested_and_or` | `{"$and": [{"province": "서울"}, {"$or": [{"sex": "여자"}, {"age": {"$gte": 60}}]}]}` | 중첩 논리 연산자 |
| 3 | `test_lte_filter` | `{"age": {"$lte": 40}}` → age ≤ 40인 결과만 | 비교 연산자 |
| 4 | `test_ne_filter` | `{"sex": {"$ne": "남자"}}` → 여자만 반환 | 부정 연산자 |

---

### 2.6 `test_retrieval.py` — RRF·메타데이터 필터 (14개)

**목적**: `rag/retrieval/hybrid.py`의 핵심 알고리즘인 Reciprocal Rank Fusion(RRF)과 메타데이터 필터를 독립적으로 검증한다.

#### `TestRRF` — RRF 알고리즘 (5개)

RRF는 Dense(의미 검색)와 Sparse(BM25 키워드) 두 결과 리스트를 병합한다. 각 문서의 RRF 점수 = Σ `1 / (k + rank_i)`, 여기서 k=60은 순위 차이를 완화하는 상수다.

| # | 테스트 | 검증 내용 | 알고리즘적 의미 |
|---|--------|----------|--------------|
| 1 | `test_basic_merge` | Dense 5개 + Sparse 5개 → 최대 10개 결과 | 두 리스트가 겹치지 않으면 합집합 |
| 2 | `test_overlap_boosts_rank` | 양쪽 리스트에 모두 있는 `"shared"` 문서가 1위 | 두 검색기 모두 높게 평가한 문서가 가장 높은 RRF 점수 획득 |
| 3 | `test_top_k_limit` | `top_k=3` 시 3개 이하만 반환 | 결과 수 제한 |
| 4 | `test_empty_sparse` | Sparse 결과 = `[]` 이어도 Dense 결과 5개 반환 | 한쪽 검색기 실패 시 다른 쪽으로 폴백 가능 |
| 5 | `test_rank_assigned_sequentially` | 반환된 결과의 `rank` 속성이 0, 1, 2, ... 순서 | 다운스트림 코드가 rank 속성에 의존하므로 재번호 필요 |

**왜 RRF k=60인가?**  
k=60은 RRF 논문(Cormack 2009)에서 제안된 기본값이다. 낮은 k는 1위 문서에 극단적 가중치를 부여하고, 높은 k는 순위 차이를 희석한다. k=60이 실험적으로 가장 안정적인 성능을 보인다.

#### `TestMetadataFilter` (7개)

`test_retrieval.py`의 `TestMetadataFilter`는 `test_pipeline.py`와 겹치는 것처럼 보이지만, retrieval 레이어(`rag/retrieval/`)의 공개 인터페이스를 테스트한다는 점에서 다르다.

| # | 테스트 | 검증 연산자 |
|---|--------|-----------|
| 1 | `test_eq_filter` | `{"sex": "여자"}` — 동등 비교 |
| 2 | `test_gte_filter` | `{"age": {"$gte": 65}}` — 이상 비교 |
| 3 | `test_contains_filter` | `{"occupation": {"$contains": "전문"}}` — 문자열 포함 |
| 4 | `test_and_filter` | `{"$and": [...]}` — AND 논리 |
| 5 | `test_or_filter` | `{"$or": [...]}` — OR 논리 |
| 6 | `test_to_chroma_where_strips_contains` | `$contains` → ChromaDB 전달 시 `None` |
| 7 | `test_to_chroma_where_keeps_eq` | `{"sex": "여자"}` → 그대로 통과 |

---

## 3. 공통 픽스처(`conftest.py`) 설명

pytest는 `conftest.py`의 픽스처를 자동으로 모든 테스트 파일에 주입한다.

### `sample_jsonl` 픽스처

```python
@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
```

- **반환**: 10건의 가짜 페르소나 레코드가 담긴 임시 JSONL 파일 경로
- **생성 규칙**: 성별(남/여 교대), 나이(20, 25, 30, ..., 65), 시도(5개 순환), 직업(5개 순환)
- **격리**: `tmp_path`는 pytest가 테스트별로 생성하는 고유 임시 디렉토리 → 테스트 간 파일 충돌 없음
- **사용처**: `test_parquet_loader.py`, `test_ontology.py`

### `sample_search_results` 픽스처

```python
@pytest.fixture
def sample_search_results():
    return [SearchResult(id=f"id-{i}", ...) for i in range(5)]
```

- **반환**: 5개의 `SearchResult` 객체 리스트 (RAG 파이프라인 중간 결과 형식)
- **사용처**: `test_nim_llm.py`의 답변 생성 테스트

---

## 4. 경고 1건 분석

```
tests/test_pipeline.py::TestToChromaWhere::test_simple_eq
  /home/minkih/nemotron-persona-korea/.venv/lib/python3.14/site-packages/chromadb/telemetry/opentelemetry/__init__.py:128:
  DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16;
  use inspect.iscoroutinefunction() instead
```

**원인**: ChromaDB 내부 텔레메트리 코드가 Python 3.12에서 deprecated된 `asyncio.iscoroutinefunction()`을 사용한다.

**영향**: 테스트 결과에 영향 없음. ChromaDB 라이브러리의 내부 코드 문제이므로 이 프로젝트에서 직접 수정 불가.

**해결 방법**: ChromaDB 버전 업데이트 시 자연스럽게 해소될 예정. Python 3.16 이전까지는 경고 수준으로만 발생.

---

## 5. 테스트 설계 원칙

### 외부 의존성 격리

모든 테스트는 실제 API 호출(NIM, Cohere), 실제 데이터셋(1M 레코드 Parquet), ChromaDB 인덱스 빌드 없이 실행된다.

| 의존성 | 격리 방법 |
|--------|----------|
| NIM LLM API | `unittest.mock.patch` + `MagicMock` |
| Parquet 데이터셋 | `sample_jsonl` 픽스처 (10건 인메모리) |
| 파일 시스템 | `tmp_path` 픽스처 (pytest 임시 디렉토리) |
| 환경변수 | `monkeypatch.setenv()` / `monkeypatch.delenv()` |

### 테스트 실행 속도

전체 63개가 **1.78초** 내에 완료되는 이유:
- HTTP 요청 없음 (완전 모킹)
- 파일 I/O 최소화 (10건 픽스처만)
- 온톨로지 테스트도 10건 샘플만 사용

### 계층별 커버리지

```
┌─────────────────────────────────────────────────┐
│  통합 수준  │  test_ontology (SPARQL 실행)        │
├─────────────────────────────────────────────────┤
│  컴포넌트   │  test_nim_llm (LLM 클라이언트)      │
│  수준       │  test_ontology (populate)           │
├─────────────────────────────────────────────────┤
│  단위 수준  │  test_config, test_pipeline,        │
│             │  test_retrieval, test_parquet_loader│
└─────────────────────────────────────────────────┘
```
