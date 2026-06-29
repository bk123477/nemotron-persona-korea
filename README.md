# Nemotron-Personas-Korea Toolkit

NVIDIA의 [Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) 데이터셋(100만 레코드, CC BY 4.0)을 활용한 **한국어 인구통계 페르소나 AI 플랫폼**.

LLM 백엔드로 **NVIDIA NIM** (`nvidia/nemotron-3-super-120b-a12b`)을 사용하며, Hybrid RAG부터 온톨로지 SPARQL 질의까지 6개 실험 트랙을 제공합니다.

---

## 할 수 있는 것들

### 1. 페르소나 검색 (Hybrid RAG)
자연어로 한국인 페르소나를 검색하고, LLM이 검색 결과를 바탕으로 답변을 생성합니다.

```
"광주에 사는 70대 하역 노동자"  →  유사 페르소나 10건 + LLM 분석
"서울 미혼 30대 여성 IT 종사자"  →  메타데이터 필터 + 의미론적 검색
```

### 2. 페르소나 캐릭터 챗봇 (LangChain)
검색된 페르소나를 시스템 프롬프트에 주입해 그 사람처럼 대화하는 챗봇을 즉시 실행합니다.

```
"부산 60대 어부"  →  LLM이 해당 페르소나로 역할극
```

### 3. 구조화 데이터 추출 (LangChain)
비정형 페르소나 텍스트에서 성격·취미·여행스타일 등을 JSON으로 추출합니다.

### 4. 멀티스텝 분석 워크플로우 (LangGraph)
페르소나 하나를 입력하면 직업·지역·가족 분석을 병렬로 실행하고, 이상값을 탐지하며, 사람이 승인/거부한 뒤 최종 인사이트를 생성하는 에이전트 그래프를 실행합니다.

### 5. MCP 서버 연동 (Claude Desktop / Claude Code)
페르소나 데이터를 MCP Tool로 노출해 Claude Desktop에서 직접 페르소나를 검색하고 롤플레이를 시작합니다.

### 6. 인구통계 위키 자동 생성 (Wiki Lab)
476K 페르소나의 통계를 집계해 시도별·연령대별·직업별 위키 문서를 자동 생성하고, 그 위키로 Q&A를 합니다.

### 7. 온톨로지 구축 및 SPARQL 질의 (Stage 5)
페르소나 데이터를 OWL/RDF 온톨로지로 변환하고, SPARQL로 복잡한 인구통계 관계를 질의합니다. 자연어를 SPARQL로 자동 변환(Text2SPARQL)하는 기능도 포함합니다.

---

## 아키텍처

```
[Nemotron-Personas-Korea 데이터셋]  (476K 레코드 / HuggingFace 자동 다운로드)
             │
             ▼
   scripts/parquet_loader.py  ──→  JSONL 변환
             │
    ┌────────┴──────────────────────────────────────┐
    │           RAG Core (rag/)                     │
    │                                               │
    │   JSONL ─→ ChromaDB (Dense)                  │
    │         ─→ BM25 Index (Sparse)                │
    │                │                              │
    │         RRF Fusion                            │
    │                │                              │
    │         Cohere Rerank v3                      │
    │                │                              │
    │         NVIDIA NIM LLM                        │
    │    nvidia/nemotron-3-super-120b-a12b          │
    └───────────────────────────────────────────────┘
             │
    ┌────────┴────────────────────────────────────────────────────┐
    │  Labs                                                       │
    │                                                             │
    │  02_langchain/   LangChain LCEL 체인                        │
    │  langgraph_lab/  LangGraph 멀티스텝 에이전트 그래프          │
    │  mcp_lab/        MCP 서버 (Claude Desktop 연동)              │
    │  wiki_lab/       인구통계 위키 자동생성 + RAG               │
    │  ontology/       OWL/RDF 온톨로지 + SPARQL + Text2SPARQL   │
    └─────────────────────────────────────────────────────────────┘
```

---

## 설치

Python 3.10+ 필요.

```bash
git clone https://github.com/bk123477/nemotron-persona-korea.git
cd nemotron-persona-korea
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 환경 변수

`~/.env` 파일에 설정합니다:

```bash
NVIDIA_API_KEY=nvapi-...          # NVIDIA NIM LLM (필수)
COHERE_API_KEY=your_cohere_key    # Reranker (선택 — 없으면 RRF 결과 그대로 사용)
```

---

## 데이터셋 준비

```bash
# 테스트용 1,000건
.venv/bin/python3 scripts/convert_personas_jsonl.py --sample 1000

# 전체 변환 (~476K 레코드, 약 5분 소요)
.venv/bin/python3 scripts/convert_personas_jsonl.py
```

출력: `Nemotron-Personas-Korea/data/jsonl/personas.jsonl`

---

## 빠른 시작

### RAG 인덱스 구축 → 검색

```bash
# 1. 인덱스 구축 (최초 1회)
.venv/bin/python3 -m rag.ingest --max-docs 10000

# 2. 검색
.venv/bin/python3 -m rag.query "광주에 사는 70대 하역 노동자"

# 3. LLM 답변 생성
.venv/bin/python3 -m rag.query "미혼 30대 남성 IT 종사자" --answer
```

### 페르소나 챗봇

```bash
.venv/bin/python3 -m labs.02_langchain.persona_chat "부산 60대 어부"
```

### LangGraph 분석 에이전트

```bash
.venv/bin/python3 -m labs.langgraph_lab.run "서울 40대 직장인 여성"
```

### 온톨로지 구축 및 SPARQL 질의

```bash
# RDF 트리플 생성 (10,000건 기본)
.venv/bin/python3 -m labs.ontology.populate

# SPARQL 질의 예시 실행
.venv/bin/python3 -m labs.ontology.query

# 자연어 → SPARQL 변환
.venv/bin/python3 -m labs.ontology.text2sparql "서울 고령층의 직업 분포는?"
```

---

## 프로젝트 구조

```
nemotron-persona-korea/
├── scripts/
│   ├── parquet_loader.py           # 데이터 로딩 · JSONL 변환 라이브러리
│   └── convert_personas_jsonl.py   # JSONL 변환 CLI
│
├── rag/                            # Hybrid RAG 파이프라인 (핵심)
│   ├── config.py                   # RAGConfig (경로, 모델, API 키)
│   ├── schema.py                   # Document, SearchResult 데이터클래스
│   ├── ingest.py                   # 인덱스 적재 CLI
│   ├── pipeline.py                 # HybridRAGPipeline (Dense+Sparse+RRF)
│   ├── query.py                    # 검색 CLI
│   ├── retrieval/
│   │   ├── dense.py                # ChromaDB + ko-sroberta-multitask
│   │   ├── sparse.py               # BM25 (rank-bm25)
│   │   └── hybrid.py               # Reciprocal Rank Fusion
│   ├── reranking/
│   │   └── cohere.py               # Cohere Rerank v3
│   └── llm/
│       ├── nim.py                  # NVIDIA NIM 클라이언트 (기본)
│       └── openrouter.py           # OpenRouter 클라이언트 (레거시)
│
├── labs/
│   ├── common/
│   │   ├── config.py               # 공통 환경 설정 (NIM API, 경로)
│   │   └── loader.py               # JSONL → LangChain Document 변환
│   │
│   ├── 02_langchain/               # Stage 2: LangChain LCEL 체인
│   │   ├── persona_chat.py         # 페르소나 주입 챗봇
│   │   ├── extractor.py            # 텍스트 → 구조화 속성 추출 (Pydantic)
│   │   └── few_shot.py             # Few-shot 페르소나 자동 생성
│   │
│   ├── langgraph_lab/              # Stage 3: LangGraph 에이전트
│   │   ├── graph_definition.py     # StateGraph + 조건 라우팅
│   │   ├── nodes.py                # 7개 분석 노드
│   │   └── run.py                  # 그래프 실행 + Human-in-the-loop
│   │
│   ├── mcp_lab/                    # Stage 4: MCP 서버
│   │   ├── persona_server.py       # FastMCP 서버 (Tool 3개 + Resource + Prompt)
│   │   └── test_client.py          # 비동기 테스트 클라이언트
│   │
│   ├── wiki_lab/                   # Stage 6: LLM Wiki
│   │   ├── generate_wiki.py        # 통계 집계 → LLM 위키 문서 생성
│   │   └── wiki_rag.py             # 위키 기반 Q&A
│   │
│   └── ontology/                   # Stage 5: OWL/RDF 온톨로지
│       ├── schema.py               # OWL 클래스 13개 + 프로퍼티 19개
│       ├── populate.py             # JSONL → RDF 트리플 변환 (.ttl)
│       ├── query.py                # SPARQL 질의 예시 8종
│       └── text2sparql.py          # 자연어 → SPARQL (NIM LLM)
│
├── tests/                          # 단위/통합 테스트 (63개, 전체 통과)
│   ├── conftest.py                 # 공통 픽스처
│   ├── test_config.py
│   ├── test_nim_llm.py
│   ├── test_pipeline.py
│   ├── test_retrieval.py
│   ├── test_ontology.py
│   └── test_parquet_loader.py
│
└── requirements.txt                # 통합 의존성
```

---

## 기능별 상세 가이드

### RAG 검색 옵션

```bash
# 검색 모드 선택
.venv/bin/python3 -m rag.query "질의" --mode dense    # 의미론적 검색
.venv/bin/python3 -m rag.query "질의" --mode sparse   # BM25 키워드 검색
.venv/bin/python3 -m rag.query "질의" --mode hybrid   # RRF 융합 (기본)

# 메타데이터 필터 조합
.venv/bin/python3 -m rag.query "전문직" \
  --filter-sex 여자 \
  --filter-province 서울 \
  --filter-age-gte 30 \
  --filter-age-lte 45

# 직업 키워드 필터 (부분 일치)
.venv/bin/python3 -m rag.query "생활 패턴" --filter-occupation 교사

# JSON 출력
.venv/bin/python3 -m rag.query "부산 노인" --json --top-k 3

# LLM 답변 스트리밍
.venv/bin/python3 -m rag.query "미혼 청년 직업 특성" --answer
```

| 모드 | 특징 | 적합한 케이스 |
|------|------|--------------|
| `dense` | 의미론적 유사도 (임베딩) | 감성·개념 기반 질의, 메타데이터 필터 |
| `sparse` | 키워드 매칭 (BM25) | 직업명·지명 등 정확한 단어 검색 |
| `hybrid` | RRF 융합 → Cohere Rerank | 일반 질의 (기본값, 권장) |

### LangChain 체인

```bash
# 페르소나 챗봇 — 그 사람처럼 대화
.venv/bin/python3 -m labs.02_langchain.persona_chat "40대 여성 교사 경기도"

# 구조화 속성 추출
.venv/bin/python3 -m labs.02_langchain.extractor --query "부산 60대 어부"
# → PersonaTraits JSON: 연령대, 직업, 성격 키워드, 취미, 여행스타일, 삶의 목표 등

# Few-shot 페르소나 생성
.venv/bin/python3 -m labs.02_langchain.few_shot "20대 남성 개발자 서울" --n-shots 5
```

### LangGraph 에이전트 그래프

```
입력 페르소나 쿼리
    ↓
[분류] 인구통계 파악
    ↓
age >= 65? ─→ [고령층 특화 분석]
    ↓
[유사 페르소나 검색]
    ↓
┌─────────┬────────────┬───────────┐
▼         ▼            ▼           
[직업분석] [지역분석] [가족분석]  ← 병렬 실행
└─────────┴────────────┴───────────┘
    ↓
[이상값 탐지]  (나이-학력 불일치, 성별-병역 모순 등)
    ↓
[Human-in-the-loop] 사용자 승인/거부
    ↓
[최종 인사이트 요약]
```

### MCP 서버 (Claude Desktop 연동)

`.mcp.json`이 이미 설정되어 있어 Claude Code에서 바로 사용 가능합니다.

노출된 Tool:
- `search_personas(query, sex, province, age_min, age_max, top_k)` — 자연어 검색
- `get_persona_by_id(uuid)` — ID로 직접 조회
- `get_demographic_stats(province, age_group)` — 통계 집계

### 온톨로지 SPARQL 질의 예시

| 번호 | 질의 내용 |
|------|----------|
| 1 | 시도별 페르소나 수 집계 |
| 2 | 직업별 평균 나이 (상위 20개) |
| 3 | 학력 × 혼인상태 교차 분석 |
| 4 | 성별 × 연령대 분포 |
| 5 | 서울 거주 고령층 직업 분포 |
| 6 | 가구형태별 분포 |
| 7 | 고학력 미혼 20~35세 페르소나 목록 |
| 8 | 시도별 1인 가구 비율 |

---

## 데이터셋 스키마

레코드당 26개 필드:

| 분류 | 필드 |
|------|------|
| 페르소나 텍스트 (7) | `persona`, `professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona`, `family_persona` |
| 속성 (6) | `cultural_background`, `skills_and_expertise`, `hobbies_and_interests`, `career_goals_and_ambitions`, `skills_and_expertise_list`, `hobbies_and_interests_list` |
| 인구통계 (12) | `sex`, `age`, `marital_status`, `military_status`, `family_type`, `housing_type`, `education_level`, `bachelors_field`, `occupation`, `district`, `province`, `country` |
| 식별자 (1) | `uuid` |

---

## 테스트

```bash
.venv/bin/python3 -m pytest tests/ -v
# 63개 테스트 전체 통과
```

| 테스트 파일 | 대상 |
|------------|------|
| `test_config.py` | NIM 설정, 하위 호환성 alias |
| `test_nim_llm.py` | NIMLM 초기화, 답변 생성, 스트리밍 |
| `test_pipeline.py` | ChromaDB where 변환, 메타데이터 필터 |
| `test_retrieval.py` | RRF 알고리즘, 필터 연산자 |
| `test_ontology.py` | OWL 스키마, RDF 변환, SPARQL 질의 |
| `test_parquet_loader.py` | JSONL 구조 및 타입 검증 |

---

## 인덱스 구축 규모 가이드

| 규모 | 명령 | Dense | BM25 | RAM |
|------|------|-------|------|-----|
| 테스트 | `--max-docs 1000` | 1K | 1K | ~500MB |
| 개발 | `--max-docs 10000` | 10K | 10K | ~1GB |
| 실용 | `--max-docs 100000` | 100K | 100K | ~4GB |
| 전체 | (옵션 없음) | 476K | 476K | ~8GB |

---

## 클라우드 확장

로컬 자원 한계 없이 운영하려면:

- **임베딩**: Cohere/OpenAI Embedding API로 교체 (`rag/retrieval/dense.py`)
- **Vector DB**: Pinecone / Qdrant Cloud / Weaviate Cloud
- **Sparse 검색**: AWS OpenSearch / Elasticsearch
- **LLM**: `rag/config.py`의 `nim_base_url`, `llm_model`만 변경

`rag/config.py`의 엔드포인트만 교체하면 동일한 파이프라인이 클라우드에서 동작합니다.

---

## 라이선스

- 코드: MIT
- 데이터셋: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — NVIDIA Nemotron-Personas-Korea
