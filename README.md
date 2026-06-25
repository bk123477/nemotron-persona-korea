# Nemotron-Personas-Korea RAG

NVIDIA의 [Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) 데이터셋(1M 레코드, CC BY 4.0)을 활용한 Hybrid RAG 파이프라인.

한국어 인구통계 페르소나를 Dense + Sparse 검색으로 조회하고, LLM으로 답변을 생성합니다.

## 아키텍처

```
[Nemotron-Personas-Korea 데이터셋]
         │
         ▼
  scripts/parquet_loader.py  ──→  JSONL 변환
         │
         ▼
    rag/ingest.py
    ┌────┴────────────┐
    ▼                 ▼
ChromaDB          BM25 Index
(Dense)           (Sparse)
    │                 │
    └────────┬─────────┘
             ▼
         RRF Fusion
             │
             ▼
     Cohere Rerank v3
             │
             ▼
     OpenRouter LLM
  (google/gemma-4-31b-it:free)
```

## 설치

Python 3.10+ 필요.

```bash
git clone https://github.com/bk123477/nemotron-persona-korea.git
cd nemotron-persona-korea
python3 -m venv .venv
source .venv/bin/activate
pip install datasets pyarrow pandas
pip install -r rag/requirements.txt
```

### 환경 변수

```bash
# ~/.env 또는 .env 파일 생성
COHERE_API_KEY=your_cohere_key       # Reranker (선택)
OPENROUTER_API_KEY=your_openrouter_key  # LLM 답변 생성 (선택)
```

## 데이터셋 준비

데이터셋은 용량 문제로 저장소에 포함되지 않습니다. HuggingFace에서 자동으로 다운로드됩니다.

```bash
# 테스트용 1,000건 변환
python3 scripts/convert_personas_jsonl.py --sample 1000

# 전체 변환 (~476K 레코드)
python3 scripts/convert_personas_jsonl.py
```

출력: `Nemotron-Personas-Korea/data/jsonl/personas.jsonl`

## 인덱스 구축

```bash
# 테스트 (1,000건 Dense + BM25)
python3 -m rag.ingest --max-docs 1000

# 실용적 구성 (Dense 10만 건 + BM25 10만 건)
python3 -m rag.ingest --max-docs 100000

# BM25만 빠르게 구축 (Dense 생략)
python3 -m rag.ingest --max-bm25-docs 100000 --skip-dense

# 중단 후 이어서 적재
python3 -m rag.ingest --resume
```

> **메모리 주의**: BM25 전체(476K) 적재 시 약 4GB RAM 필요. 기본값 100K.

## 검색

```bash
# 기본 Hybrid 검색
python3 -m rag.query "광주에 사는 70대 하역 노동자"

# Dense 검색 + 메타데이터 필터
python3 -m rag.query "전문직 종사자" --mode dense --filter-sex 여자 --filter-province 서울

# 나이 범위 필터
python3 -m rag.query "은퇴 후 취미 활동" --filter-age-gte 60 --filter-age-lte 75

# LLM 답변 생성 (OPENROUTER_API_KEY 필요)
python3 -m rag.query "미혼 30대 남성 IT 종사자의 특징" --answer

# JSON 출력
python3 -m rag.query "서울 거주 대졸 여성" --json --top-k 3

# 대화형 모드
python3 -m rag.query
```

## 프로젝트 구조

```
nemotron-persona-korea/
├── scripts/
│   ├── parquet_loader.py           # 데이터 로딩 · JSONL 변환 라이브러리
│   └── convert_personas_jsonl.py   # JSONL 변환 CLI
├── rag/
│   ├── config.py                   # RAGConfig (경로, 모델, API 키)
│   ├── schema.py                   # Document, SearchResult 데이터클래스
│   ├── ingest.py                   # 인덱스 적재 CLI
│   ├── pipeline.py                 # HybridRAGPipeline (Dense+Sparse+RRF)
│   ├── query.py                    # 검색 데모 CLI
│   ├── retrieval/
│   │   ├── dense.py                # ChromaDB + ko-sroberta-multitask
│   │   ├── sparse.py               # BM25 (rank-bm25)
│   │   └── hybrid.py               # Reciprocal Rank Fusion
│   ├── reranking/
│   │   └── cohere.py               # Cohere Rerank v3
│   ├── llm/
│   │   └── openrouter.py           # OpenRouter LLM 클라이언트
│   └── requirements.txt
└── Nemotron-Personas-Korea/        # Git 미포함 (gitignore), HF에서 다운로드
```

## 데이터셋 스키마

레코드당 26개 필드:

| 분류 | 필드 |
|------|------|
| 페르소나 텍스트 (7) | `persona`, `professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona`, `family_persona` |
| 속성 (6) | `cultural_background`, `skills_and_expertise`, `hobbies_and_interests`, `career_goals_and_ambitions`, `skills_and_expertise_list`, `hobbies_and_interests_list` |
| 인구통계 (12) | `sex`, `age`, `marital_status`, `military_status`, `family_type`, `housing_type`, `education_level`, `bachelors_field`, `occupation`, `district`, `province`, `country` |
| 식별자 (1) | `uuid` |

## 검색 모드 비교

| 모드 | 특징 | 적합한 케이스 |
|------|------|--------------|
| `dense` | 의미론적 유사도 (임베딩) | 개념·감성 기반 질의, 메타데이터 필터 |
| `sparse` | 키워드 매칭 (BM25) | 직업명·지명 등 정확한 단어 검색 |
| `hybrid` | RRF 융합 → Cohere Rerank | 일반 질의 (기본값, 권장) |

## 클라우드 확장

로컬 자원 제한 없이 운영하려면:

- **임베딩**: GPU 인스턴스(AWS g4dn, GCP A100) 또는 Cohere/OpenAI Embedding API
- **Vector DB**: Pinecone / Qdrant Cloud / Weaviate Cloud (ChromaDB 교체)
- **Sparse 검색**: AWS OpenSearch / Elasticsearch (BM25 기본 지원)
- **데이터**: S3 / GCS에 JSONL 적재 후 스트리밍 인덱싱

`rag/config.py`의 엔드포인트만 교체하면 동일한 파이프라인 코드가 클라우드에서 동작합니다.

## 라이선스

- 코드: MIT
- 데이터셋: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — NVIDIA Nemotron-Personas-Korea
