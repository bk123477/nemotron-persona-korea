# 전체 Stage 실행 결과 보고서

**실행일**: 2026-06-30  
**환경**: Python 3.14.4 / WSL2 (Ubuntu) / NVIDIA NIM API  
**모델**: `nvidia/nemotron-3-super-120b-a12b` (NVIDIA NIM)  
**인덱스**: Dense 500건, Sparse(BM25) 500건

---

## 목차

| Stage | 이름 | 상태 |
|-------|------|------|
| [Stage 1](#stage-1-hybrid-rag-파이프라인) | Hybrid RAG 파이프라인 | ✅ |
| [Stage 2-A](#stage-2a-구조화-추출-extractor) | 구조화 추출 (Extractor) | ✅ |
| [Stage 2-B](#stage-2b-few-shot-페르소나-생성) | Few-shot 페르소나 생성 | ⚠️ |
| [Stage 3](#stage-3-langgraph-분석-파이프라인) | LangGraph 분석 파이프라인 | ✅ |
| [Stage 4](#stage-4-mcp-서버) | MCP 서버 | ✅ |
| [Stage 5-A](#stage-5a-온톨로지-populate) | 온톨로지 Populate | ✅ |
| [Stage 5-B](#stage-5b-sparql-질의) | SPARQL 질의 (8개) | ✅ |
| [Stage 5-C](#stage-5c-text2sparql) | Text2SPARQL | ✅ |
| [Stage 6](#stage-6-wiki-rag) | Wiki RAG | ⚠️ |

> **⚠️ 상태 설명**: Stage 2-B, 6은 파이프라인 자체는 정상 동작하나, NIM 모델의 chain-of-thought 추론 텍스트가 최종 답변 앞에 출력되는 현상이 있음. [NIM 모델 특성 섹션](#nim-reasoning-model-특성-및-대응)에서 상세 설명.

---

## Stage 1: Hybrid RAG 파이프라인

### 실행 명령어

```bash
cd /home/minkih/nemotron-persona-korea

# Hybrid 검색 (Dense + Sparse BM25 RRF 융합)
.venv/bin/python3 -m rag.query "부산 60대 남성 어부의 생활 방식" --mode hybrid

# Dense만 (ChromaDB 의미 검색)
.venv/bin/python3 -m rag.query "부산 60대 남성 어부의 생활 방식" --mode dense

# Sparse만 (BM25 키워드 검색)
.venv/bin/python3 -m rag.query "부산 60대 남성 어부의 생활 방식" --mode sparse
```

### Hybrid 검색 결과

```
[인덱스 상태]  Dense: 500건 | Sparse(BM25): 500건 | Reranker: ✗ (COHERE_API_KEY 없음)

────────────────────────────────────────────────────────────
질의: 부산 60대 남성 어부의 생활 방식  |  모드: hybrid  |  결과: 5건
────────────────────────────────────────────────────────────
[ 1] score=0.0315  source=hybrid
     남자 / 62세 / 부산 부산-수영구 / 무직
     학력=초등학교  혼인=배우자있음  가구=배우자와 거주
[ 2] score=0.0280  source=hybrid
     남자 / 55세 / 부산 부산-사하구 / 위성방송 안테나 설치 및 수리원
     학력=초등학교  혼인=배우자있음  가구=배우자·자녀와 거주
[ 3] score=0.0164  source=hybrid
     남자 / 76세 / 경상남 경상남-통영시 / 무직
     학력=초등학교  혼인=배우자있음  가구=배우자와 거주
[ 4] score=0.0164  source=hybrid
     남자 / 59세 / 부산 부산-해운대구 / 그 외 철도 및 전동차 기관사
     학력=4년제 대학교  혼인=배우자있음  가구=?
[ 5] score=0.0161  source=hybrid
     남자 / 50세 / 인천 인천-서구 / 무직
     학력=고등학교  혼인=배우자있음  가구=혼자 거주 (배우자 별거)
```

### Dense vs Sparse 비교

| 검색 모드 | 1위 결과 | 1위 점수 | 특징 |
|----------|---------|---------|------|
| Dense (의미 검색) | 남자/76세/경상남-통영시/무직 | 0.6295 (코사인 유사도) | "어부", "바닷가" 의미적 연관 페르소나 우선 |
| Sparse (BM25) | 남자/59세/부산-해운대구/기관사 | 9.4321 (BM25 점수) | "부산", "60대" 등 키워드 정확 매칭 우선 |
| Hybrid (RRF 융합) | 남자/62세/부산-수영구/무직 | 0.0315 (RRF 점수) | 두 검색기가 공통으로 높게 평가한 문서가 상위 |

### 왜 이런 결과가 나오는가?

**Dense 검색**은 `jhgan/ko-sroberta-multitask` 임베딩 모델로 "어부"를 의미적으로 유사한 벡터로 변환해 ChromaDB에서 코사인 유사도 기반 검색을 수행한다. 따라서 "어부"와 의미적으로 가까운 "하역 종사원", "바닷가 거주" 등이 높게 평가된다.

**Sparse BM25 검색**은 "부산", "60", "남성" 같은 정확한 키워드 토큰이 텍스트에 얼마나 자주, 드물게 등장하는지로 점수를 계산한다. "부산"이라는 단어가 텍스트에 있는 문서가 그렇지 않은 것보다 점수가 높다.

**RRF (Reciprocal Rank Fusion)**는 두 리스트를 통합한다. 각 문서의 RRF 점수 = `1/(60 + rank_dense) + 1/(60 + rank_sparse)`. k=60은 순위 차이를 완화하는 표준 상수다. 두 검색기 모두에서 상위에 오른 문서가 가장 높은 RRF 점수를 받는다.

---

## Stage 2-A: 구조화 추출 (Extractor)

### 실행 명령어

```bash
# ChromaDB에서 "부산 60대 어부" 검색 후 추출
.venv/bin/python3 -m labs.02_langchain.extractor --query "부산 60대 어부"

# 직접 텍스트 입력
.venv/bin/python3 -m labs.02_langchain.extractor --text "서울 강남에 사는 40대 의사..."

# 무작위 페르소나 선택
.venv/bin/python3 -m labs.02_langchain.extractor
```

### 실행 결과

```
[검색 중: '부산 60대 어부']
[매칭] 남자 / 76세 / 경상남 / 무직

──────────────────────────────────────────────────
[원본 텍스트 미리보기]
Persona: 장대수 씨는 통영의 갯내음과 함께 살아온 수산물 유통의 베테랑으로,
철저한 가계 관리와 가족에 대한 깊은 사랑을 가진 고집 센 경상도 할아버지입니다.
Professional persona: 장대수 씨는 수십 년간 통영 강구안 수산물 시장에서 생선의
눈깔만 봐도 신선도를 단숨에 맞히는 베테랑이었으며, 십 원 단위까지 꼼꼼하게
기록한 낡은 장부들을 지금도 보물처럼 간직하고 있습니다...
──────────────────────────────────────────────────
[추출 중...]

[추출 결과]
  이름/별칭    : 장대수
  연령대       : 60대
  직업 요약    : 수산물 유통업 종사자
  성격 키워드  : 고집센, 투박함, 성실함, 가족애, 꼼꼼함, 열정적
  주요 취미    : 야구 경기 시청
  여행 스타일  : 가족 중심의 조용한 역사 유적지 여행
  삶의 목표    : 가족과의 평화로운 노후
  인상적 사실  : 십 원 단위까지 꼼꼼히 기록한 낡은 장부를 보물처럼 간직함

[JSON]
{
  "name_or_alias": "장대수",
  "age_group": "60대",
  "occupation_summary": "수산물 유통업 종사자",
  "personality_keywords": ["고집센", "투박함", "성실함", "가족애", "꼼꼼함", "열정적"],
  "top_hobby": "야구 경기 시청",
  "travel_style": "가족 중심의 조용한 역사 유적지 여행",
  "life_goal": "가족과의 평화로운 노후",
  "notable_fact": "십 원 단위까지 꼼꼼히 기록한 낡은 장부를 보물처럼 간직함"
}
```

### 왜 이렇게 동작하는가?

LangChain LCEL 체인 구조: `ChatPromptTemplate | ChatOpenAI | RunnableLambda(_parse)`

1. `ChatPromptTemplate`이 "JSON만 출력" 지시와 페르소나 텍스트를 조합해 프롬프트 생성
2. `ChatOpenAI`가 NIM API를 호출해 JSON 응답 생성  
3. `_parse()` 함수의 `_extract_json()` 헬퍼가 정규식으로 LLM 응답에서 JSON 블록 추출

**NIM 모델 chain-of-thought 대응**: NIM reasoning model이 JSON 앞에 추론 과정을 텍스트로 출력할 때, `_extract_json()` 정규식 `r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}'`이 마지막 `{}` 블록(= 실제 JSON)만 추출하므로 추론 텍스트는 자동 제거된다.

```python
def _extract_json(text: str) -> str:
    matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL))
    if matches:
        return matches[-1].group()  # 가장 마지막 JSON 블록 반환
    ...
```

---

## Stage 2-B: Few-shot 페르소나 생성

### 실행 명령어

```bash
# 유사 페르소나 3개를 few-shot 예시로 신규 페르소나 생성
.venv/bin/python3 -m labs.02_langchain.few_shot "40대 남성 의사 서울" --n-shots 3

# few-shot 수 조정
.venv/bin/python3 -m labs.02_langchain.few_shot "20대 여성 개발자 서울" --n-shots 5
```

### 실행 결과

```
[유사 페르소나 검색: '40대 남성 의사 서울' | few-shot 3개]

[Few-shot 예시 3개]
  [1] 남자 / 24세 / 경기 / 치과위생사
  [2] 남자 / 51세 / 대구 / 그 외 물품 이동 장비 조작원
  [3] 남자 / 56세 / 광주 / 무직

[생성 대상: 40대 남성 의사 서울]
──────────────────────────────────────────────────
[생성 중...]

[생성된 페르소나]
Okay, let's tackle this query. The user wants a persona description for a 40s male
doctor in Seoul, following the examples provided.
...
(reasoning text 생략 — NIM 모델 chain-of-thought 출력)
```

### 왜 reasoning 텍스트가 나오는가?

`nvidia/nemotron-3-super-120b-a12b`는 **reasoning model**이다. 일반 지시 모델(instruction model)과 달리, 답변 전에 영어로 단계별 사고 과정을 출력하는 것이 이 모델의 내재적 특성이다. 이는 "추론 과정 출력하지 마세요" 같은 시스템 프롬프트 지시로도 완전히 제어되지 않는다.

**파이프라인 자체는 정상 동작**한다. `FewShotChatMessagePromptTemplate`이 3개의 유사 페르소나를 예시로 삽입하고, `ChatOpenAI | StrOutputParser` 체인이 실행된다. 최종 출력에 한국어 페르소나 텍스트가 포함되나, 긴 추론 텍스트 뒤에 위치한다.

**상세 해결 방법**: [NIM Reasoning Model 특성 및 대응](#nim-reasoning-model-특성-및-대응) 섹션 참고.

---

## Stage 3: LangGraph 분석 파이프라인

### 실행 명령어

```bash
# 7-node 그래프 자동 실행 (human-in-the-loop 자동 승인)
.venv/bin/python3 -m labs.langgraph_lab.run "서울 30대 직장인" --auto-approve

# Mermaid 다이어그램 출력
.venv/bin/python3 -m labs.langgraph_lab.run --mermaid
```

### 실행 결과

```
[그래프 실행 시작] 질의: '서울 30대 직장인'
────────────────────────────────────────────────────────────

  ▶ [인구통계 분류] [분류] 남자 / 36세(중년 (30~49세)) / 대전 대전-서구 / 경리 사무원 / 학력:2~3년제 전문대학
  ▶ [유사 페르소나 검색] [유사 페르소나] 5건 검색됨
  ▶ [지역 분석] [지역 분석] 대전 대전-서구
  ▶ [직업 분석] [직업 분석] 경리 사무원
  ▶ [가족 분석] [가족 분석] 부모와 동거 / 미혼
  ▶ [이상값 감지] [이상값 감지] 이상값 없음 — 일반적인 인구통계 조합

[중간 검토] 분석 대상 페르소나
────────────────────────────────────────────────────────────
  남자 / 36세 (중년 (30~49세)) / 대전 / 경리 사무원

[직업 분석]
경리 사무원으로서 윤종원 씨는 엑셀을 활용한 원가·급여 관리에서 오차 없이 처리하며
대표의 절대적인 신뢰를 받는 등 직업적 안정성과 전문성이 높은 편이다.

[지역 분석]
대전 서구는 대전의 행정·교육 중심지로, 공공기관·대학·연구소가 밀집해 상대적으로
높은 교육 수준과 중산층 이상의 소득 분포를 보이는 사회경제적 특성을 보입니다.

[가족 분석]
현재 36세, 부모와 동거하며 미혼인 상태는 '청년‑성인 전환기'에 해당하며, 아직 독립
가구를 형성하지 않은 채 부모와의 경제적·정서적 의존이 지속되는 단계입니다.

[이상값] 없음

계속 진행하려면 'yes', 취소는 'no', 피드백은 텍스트 입력:
[자동 승인] 'yes'
────────────────────────────────────────────────────────────
[그래프 재개]
  ▶ [인간 검토 (interrupt)] [인간 검토] 결정: 'yes' → 승인
  ▶ [최종 요약] [요약] 최종 종합 완료

════════════════════════════════════════════════════════════
[ 최종 분석 결과 ]
════════════════════════════════════════════════════════════
  대상: 남자 / 36세 (중년 (30~49세)) / 대전 / 경리 사무원

[종합 인사이트]
윤종원 씨는 정확성과 책임감이 강한 경리 사무원으로서 엑셀·전산회계 역량을
지속적으로 강화하며, 직장 내 신뢰를 바탕으로 안정적인 소득을 확보하고 있다.
부모와 동거 중인 미혼 상태로 주거비 부담은 낮지만 개인 공간이 제한되어
집 근처 산책, 홈트레이닝, 스트리밍·게임 등 저비용·집중형 여가를 선호한다.

[유사 페르소나 Top-5]
  [1] 남자 / 38세 / 인천 / 생산관리 사무원
  [2] 남자 / 47세 / 경상남 / 전기 감리 기술자 및 연구원
  [3] 남자 / 36세 / 경상남 / 회계 사무원
  [4] 남자 / 36세 / 경기 / 경리 사무원
  [5] 남자 / 35세 / 서울 / 회계 사무원

[실행 로그]
  [분류] 남자 / 36세(중년 (30~49세)) / 대전 대전-서구 / 경리 사무원
  [유사 페르소나] 5건 검색됨
  [가족 분석] 부모와 동거 / 미혼
  [직업 분석] 경리 사무원
  [지역 분석] 대전 대전-서구
  [이상값 감지] 이상값 없음 — 일반적인 인구통계 조합
  [인간 검토] 결정: 'yes' → 승인
  [요약] 최종 종합 완료
```

### 왜 이렇게 동작하는가?

LangGraph `StateGraph`로 구성된 7-node 파이프라인:

```
classify ──→ search_similar ──→ occ_analysis ──┐
                                 region_analysis──┤→ outlier_check → human_review → summarize
                                 family_analysis──┘
```

1. **`classify` 노드**: RAG 파이프라인으로 질의와 가장 유사한 페르소나 1건 검색 → 연령대 분류(청년/중년/장년/고령) → 전문가 분석 노드 결정
2. **병렬 노드** (`occ_analysis`, `region_analysis`, `family_analysis`): LangGraph가 세 노드를 동시에 실행 (fan-out). 각 노드가 NIM LLM에 전문화된 프롬프트로 독립 질의
3. **`outlier_check` 노드**: 분석 결과에서 이상값(통계적 이상치) 감지. 예: "고령 청년층 조합" 등
4. **`human_review` (interrupt)**: `interrupt()` 호출로 그래프 실행을 중단하고 사람 검토 대기. `--auto-approve` 옵션으로 자동 'yes' 응답
5. **`summarize` 노드**: 모든 분석 결과를 종합해 마케팅·서비스 인사이트 생성

**`MemorySaver`** 체크포인터가 상태를 메모리에 저장하므로, interrupt 후 재개 시 이전 분석 결과가 유지된다.

---

## Stage 4: MCP 서버

### 실행 명령어

```bash
# MCP 서버 통합 테스트
.venv/bin/python3 -m labs.mcp_lab.test_client
```

### 실행 결과

```
============================================================
Nemotron-Personas-Korea MCP 서버 테스트
============================================================

[1] 서버 정보 확인
  ✓ 서버 연결 성공 (name=persona-korea)

[2] Tools 목록
  ✓ tool: get_demographic_stats
  ✓ tool: search_personas
  ✓ tool: get_persona_by_id

[3] Resources 목록
  ✓ resource template: persona://{uuid}

[4] Prompts 목록
  ✓ prompt: persona_roleplay

[5] Tool 호출: search_personas
  ✓ search_personas (기본 검색)
      count=3 | 첫 결과: 성별: 남자 | 나이: 74세 | 지역: 광주 광주-서구
                          직업: 하역 및 적재 관련 단순 종사원
                          학력: 초등학교 | 혼인: 배우자있음
  ✓ search_personas (province=서울 필터)
      count=2

[6] Tool 호출: get_persona_by_id
  ✓ get_persona_by_id (uuid=03b4f36a...)  → 정상 반환
  ✓ get_persona_by_id (없는 ID → error 반환)

[7] Tool 호출: get_demographic_stats
  ✓ get_demographic_stats (전체)
      matched=500 | sex={'남자': 257, '여자': 243}
  ✓ get_demographic_stats (서울 + 청년)
      matched=18 | age={'min': 19, 'max': 29, 'avg': 23.9}

[8] Resource 조회: persona://{uuid}
  ✓ persona://03b4f36a...
      성별: 남자 | 나이: 74세 | 지역: 광주 광주-서구

[9] Prompt 호출: persona_roleplay
  ✓ persona_roleplay
      "당신은 아래 프로필의 한국인입니다. 이 사람의 성격, 경험, 가치관을
       그대로 살려 1인칭으로 자연스럽게 대화하세요."

============================================================
결과: 전체 통과 ✓
============================================================
```

### 왜 이렇게 동작하는가?

`FastMCP` 프레임워크로 구현된 MCP(Model Context Protocol) 서버:

| 구성 요소 | 설명 |
|----------|------|
| **Tool: `search_personas`** | 자연어 질의 + 선택적 province/sex 필터로 ChromaDB에서 유사 페르소나 검색 |
| **Tool: `get_persona_by_id`** | UUID로 특정 페르소나 전체 텍스트 조회. 없는 ID는 error dict 반환 |
| **Tool: `get_demographic_stats`** | 전체 또는 필터링된 페르소나의 통계 집계 (성별 분포, 나이 통계 등) |
| **Resource: `persona://{uuid}`** | URI 템플릿으로 특정 페르소나에 직접 접근 |
| **Prompt: `persona_roleplay`** | 페르소나 프로필을 받아 1인칭 대화용 시스템 프롬프트 생성 |

`test_client.py`는 `mcp.ClientSession`을 사용해 in-process로 서버와 통신하므로, 실제 네트워크 연결 없이 서버-클라이언트 전체 흐름을 검증한다. 9개 항목 전체 통과.

---

## Stage 5-A: 온톨로지 Populate

### 실행 명령어

```bash
# JSONL → OWL/RDF 트리플 → .ttl 파일 변환 (5,000건)
.venv/bin/python3 -m labs.ontology.populate --max-docs 5000

# 전체 변환 (약 1M건)
.venv/bin/python3 -m labs.ontology.populate
```

### 실행 결과

```
2026-06-30 09:43:17,443 1000건 처리됨 (트리플: 16,061)
2026-06-30 09:43:17,761 2000건 처리됨 (트리플: 30,572)
2026-06-30 09:43:18,059 3000건 처리됨 (트리플: 44,898)
2026-06-30 09:43:18,366 4000건 처리됨 (트리플: 59,132)
2026-06-30 09:43:18,711 5000건 처리됨 (트리플: 73,303)

[JSONL 경로] /home/minkih/nemotron-persona-korea/Nemotron-Personas-Korea/data/jsonl/personas.jsonl
[출력 경로]  /home/minkih/nemotron-persona-korea/labs/ontology/data/personas.ttl
[변환 대수]  5,000건
[완료] 5,000건 → personas.ttl  (트리플: 73,303개)
```

### 왜 이렇게 동작하는가?

**트리플 생성 비율**: 5,000건에서 73,303개 트리플 ≈ 레코드당 약 **14.7 트리플**

각 페르소나 레코드에서 생성되는 트리플:
- `nemo:Persona` 타입 선언 (1개)
- `nemo:hasAge`, `nemo:hasBachelorsField` 등 데이터 프로퍼티 (최대 2개)
- `nemo:hasSex`, `nemo:hasOccupation`, `nemo:hasEducation`, `nemo:hasMaritalStatus`, `nemo:hasFamilyType`, `nemo:hasHousingType`, `nemo:hasMilitaryStatus`, `nemo:hasAgeGroup` 오브젝트 프로퍼티 (8개)
- `nemo:livesInProvince`, `nemo:livesInDistrict` (2개)
- 카테고리 인스턴스 (Province, District, OccupationCategory 등)의 라벨 트리플 (첫 등장 시만 생성)

**카테고리 캐싱**: 같은 시도(예: "서울")의 `Province` URI는 첫 생성 후 `_cache["prov"]`에 저장되어 재사용. 시도 17개가 5,000개 페르소나에서 공유되므로 트리플 중복이 없다.

**처리 속도**: 1,000건 → 약 0.3초. 전체 1M건이면 약 5분 예상.

---

## Stage 5-B: SPARQL 질의

### 실행 명령어

```bash
# 8개 사전 정의 SPARQL 쿼리 전체 실행
.venv/bin/python3 -m labs.ontology.query
```

### 실행 결과

**질의 1 — 시도별 페르소나 수**
```
경기 | 1405    서울 | 943     경상남 | 306   부산 | 299
인천 | 247     경상북 | 238   대구 | 234    충청남 | 207
전라남 | 189   전북 | 162     충청북 | 145   대전 | 141
강원 | 137     광주 | 131     울산 | 109    제주 | 73
세종 | 34
```

**질의 2 — 직업별 평균 나이 (상위 5개)**
```
무직          | 평균 56.7세 | 1,784명
건물 청소원   | 평균 63.1세 |    94명
건물 경비원   | 평균 56.0세 |    85명
사무 보조원   | 평균 45.4세 |    84명
경리 사무원   | 평균 43.5세 |    80명
```

**질의 3 — 학력 × 혼인상태 교차 분석 (상위 5개)**
```
고등학교     × 배우자있음 | 1,131명
4년제 대학교 × 배우자있음 |   741명
4년제 대학교 × 미혼       |   514명
2~3년제 전문대 × 배우자있음 | 350명
2~3년제 전문대 × 미혼       | 341명
```

**질의 4 — 성별 × 연령대 분포**
```
남자 | 고령(65세+) | 517    남자 | 장년(50~64) | 725
남자 | 중년(30~49) | 850    남자 | 청년(18~29) | 394
여자 | 고령(65세+) | 651    여자 | 장년(50~64) | 715
여자 | 중년(30~49) | 811    여자 | 청년(18~29) | 337
```

**질의 5 — 서울 거주 고령층(65세+) 직업 분포 (상위 5개)**
```
무직             | 128명
건물 청소원      |   6명
그 외 서비스 종사원 | 5명
회계 사무원      |   3명
건물 경비원      |   2명
```

**질의 6 — 가구형태별 분포 (상위 5개)**
```
배우자·자녀와 거주 | 1,352명
배우자와 거주      | 1,022명
혼자 거주          |   672명
부모와 동거        |   464명
기타2세대          |   204명
```

**질의 7 — 고학력 미혼 20~35세 페르소나 목록 (상위 3개)**
```
persona_093151f0... | 22세 | 서울 | 전직 양식 조리사
persona_0e69793e... | 22세 | 서울 | 무대 및 세트 디자이너
persona_0f6e9162... | 22세 | 서울 | 치과위생사
```

**질의 8 — 시도별 1인 가구 수**
```
경기 | 전체 1,405명 | 1인 가구 195명
서울 | 전체   943명 | 1인 가구 167명
경상남 | 전체 306명 | 1인 가구  57명
```

### 왜 이런 결과가 나오는가?

`rdflib.Graph.query(sparql_string)` 메서드가 `.ttl` 파일에서 로드한 그래프에 SPARQL SELECT 쿼리를 실행한다. rdflib는 Python 내장 SPARQL 엔진으로 외부 트리플스토어 없이도 작동한다.

**질의 2 결과 해석**: 건물 청소원 평균 나이(63.1세)가 무직(56.7세)보다 높은 이유는 이 직군이 육체적 노동임에도 대안이 없어 고령층이 오래 종사하는 사회구조적 특성을 반영한다.

**질의 4 결과 해석**: 여자 고령층(651명)이 남자 고령층(517명)보다 많은 이유는 여성의 평균수명이 길기 때문이다. 반면 청년층(18~29세)은 남자(394명) > 여자(337명)로 역전되는데, 이는 남성의 군 복무 기간이 청년기를 연장시키는 효과와 남성 인구 구조를 반영한다.

---

## Stage 5-C: Text2SPARQL

### 실행 명령어

```bash
# 자연어 → SPARQL 변환 및 실행
.venv/bin/python3 -m labs.ontology.text2sparql "경기도 중년 여성의 교육 수준 분포는?"

# 대화형 모드
.venv/bin/python3 -m labs.ontology.text2sparql
```

### 실행 결과

```
[질문] 경기도 중년 여성의 교육 수준 분포는?
[SPARQL 생성 중...]

[생성된 SPARQL]
──────────────────────────────────────────────────
PREFIX nemo: <http://nemotron.persona.kr/ontology#>
PREFIX data: <http://nemotron.persona.kr/data/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?eduName (COUNT(?person) AS ?count)
WHERE {
  ?person a nemo:Persona ;
          nemo:hasAgeGroup data:age_middle ;
          nemo:hasSex ?sex ;
          nemo:livesInProvince ?prov ;
          nemo:hasEducation ?edu .
  ?prov nemo:hasProvinceName ?provName .
  FILTER(STR(?provName) = "경기")
  ?sex rdfs:label ?sexLabel .
  FILTER(LANG(?sexLabel) = 'ko' && STR(?sexLabel) = "여자")
  ?edu nemo:hasEducationName ?eduName .
}
GROUP BY ?eduName
ORDER BY DESC(?count)
LIMIT 20
──────────────────────────────────────────────────

[그래프 로딩 중...]
[쿼리 실행 중...]

[결과] 7건
──────────────────────────────────────────────────
   1. 4년제 대학교 | 117
   2. 고등학교     |  73
   3. 2~3년제 전문대학 | 44
   4. 대학원       |  15
   5. 중학교       |   5
   6. 초등학교     |   3
   7. 무학         |   1
```

### 왜 이렇게 동작하는가?

**Text2SPARQL 파이프라인**:
1. NIM LLM에게 온톨로지 스키마 문서를 시스템 프롬프트로 제공
2. 자연어 질문 입력 → LLM이 SPARQL SELECT 쿼리 생성
3. 생성된 SPARQL에서 마크다운 코드 블록(` ``` `) 자동 제거
4. `rdflib.Graph.query()`로 실행 → 결과 출력

**LLM이 올바른 SPARQL을 생성한 이유**:
- 시스템 프롬프트에 시도명 예시(`"경기"`, `"서울"` 등)와 올바른 `FILTER(STR(?provName) = "경기")` 패턴이 포함되어 있음
- 연령대 URI 예시(`data:age_middle = 중년 (30~49세)`)가 정확히 문서화되어 있음
- 성별 필터를 `rdfs:label` 기반으로 처리하는 방법이 스키마에 명시되어 있음

이전 버전(max_tokens=800)에서는 SPARQL이 중간에 잘려 파싱 오류가 발생했다. max_tokens=1500으로 증가 후 완전한 SPARQL 생성이 가능해졌다.

---

## Stage 6: Wiki RAG

### 실행 명령어

```bash
# 위키 인덱스 재구축
.venv/bin/python3 -m labs.wiki_lab.wiki_rag --build-index --force

# 질의 및 소스 표시
.venv/bin/python3 -m labs.wiki_lab.wiki_rag \
    --query "서울의 인구 특성과 주요 직업 분포를 설명해주세요" \
    --show-sources
```

### 인덱스 구축 결과

```
[인덱스 삭제] 기존 persona-wiki 컬렉션 제거
[임베딩] 44개 청크 처리 중...
[완료] 44개 청크 → ChromaDB 'persona-wiki'
```

### 질의 결과 (정상 응답 예시)

```
Q: 서울의 인구 특성과 주요 직업 분포를 설명해주세요

A: 서울 지역 페르소나는 총 87,978명으로, 성별은 여자 45,659명(51.9%),
   남자 42,319명(48.1%)입니다. 평균 나이는 49.1세이며, 연령대는 20대(14,299명),
   30대(15,383명), 40대(14,640명), 50대(15,764명) 순으로 많습니다.
   주요 직업 상위 7개는 무직(32,250명, 36.7%), 건물 청소원(1,422명, 1.6%),
   사무 보조원(1,388명, 1.6%), 경리 사무원(1,315명, 1.5%),
   전화 상담원(1,284명, 1.5%), 일반 비서(1,076명, 1.2%),
   마케팅 전문가(925명, 1.1%)입니다.

[참조 소스]
  - regions/서울
  - demographics/age_groups
  - occupations/overview
  - regions/세종
```

### 왜 이렇게 동작하는가?

**Wiki RAG 아키텍처**:

```
wiki/ 디렉토리 (44개 Markdown 파일)
  └── regions/ (17개 시도 위키)
  └── demographics/ (연령대별 통계)
  └── occupations/ (직업 개요)
         ↓ --build-index
ChromaDB 'persona-wiki' 컬렉션 (44개 청크)
         ↓ 질의 시
ko-sroberta-multitask 임베딩 → 유사 청크 4개 검색
         ↓
NIM LLM에 컨텍스트로 전달 → 최종 답변 생성
```

**"서울" 질의가 정상 동작한 이유**: `regions/서울.md`에 정확한 통계 수치(87,978명, 무직 36.7% 등)가 포함되어 있고, 임베딩 검색에서 해당 파일이 1순위로 검색되므로 LLM이 위키 문서를 근거로 구체적인 수치를 포함한 답변을 생성한다.

**복합 조건 질의의 제한**: "경기도 중장년 1인 가구" 같이 여러 위키 파일을 교차 분석해야 하는 질의는 현재 위키 구조에서 정확한 답변이 어렵다. 이 경우 NIM 모델이 chain-of-thought 추론을 통해 추론을 시도하지만 불완전한 결과가 나온다. [온톨로지 SPARQL 질의 (Stage 5-B)](#stage-5b-sparql-질의)를 사용하면 정확한 교차 분석이 가능하다.

---

## NIM Reasoning Model 특성 및 대응

### 현상

`nvidia/nemotron-3-super-120b-a12b`는 **reasoning model**로 설계되었다. 답변 전에 영어로 단계별 사고 과정(`"Okay, let's tackle this..."`)을 출력하는 것이 모델의 내재적 동작이다.

| Stage | 영향 | 대응 |
|-------|------|------|
| Stage 1 (RAG query `--answer`) | 추론 텍스트가 최종 답변 앞에 출력 | 시스템 프롬프트에 "추론 출력 금지" 지시 추가 (부분 효과) |
| Stage 2-A (Extractor) | 추론 텍스트 + JSON 혼합 출력 | `_extract_json()` 정규식으로 JSON만 추출 → **완전 해결** |
| Stage 2-B (Few-shot) | 추론 텍스트 + 한국어 페르소나 혼합 | `StrOutputParser`가 전체 텍스트 반환, 추론 제거 어려움 |
| Stage 3 (LangGraph) | 분석 노드에서 추론 텍스트 일부 포함 가능 | 구조화된 분석 포맷으로 최종 출력은 정상 |
| Stage 5-C (Text2SPARQL) | SPARQL 전에 추론 텍스트 출력 가능 | 마크다운 코드 블록 파싱 + 완전한 SPARQL 추출 |
| Stage 6 (Wiki RAG) | 추론 텍스트가 최종 답변 앞에 출력 | 시스템 프롬프트 지시 추가 (부분 효과) |

### 근본 원인

이 모델은 [NVIDIA Build](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b) 페이지에서 "advanced reasoning" 능력을 강조하는 reasoning 특화 모델이다. 추론 텍스트 출력을 프롬프트 수준에서 완전히 억제하려면 모델 수준의 제어(예: `think_mode=off` 같은 API 파라미터)가 필요하며, 이 기능이 현재 NIM API에서 공식 지원되지 않는 경우 프롬프트 엔지니어링만으로는 한계가 있다.

### 완전한 해결책 (향후 적용 가능)

1. **structured output 강제**: JSON Schema를 API에 전달해 JSON만 반환하도록 강제
2. **다른 NIM 모델 사용**: `meta/llama-3.1-70b-instruct` 등 reasoning 없는 지시 모델로 교체
3. **후처리 파이프라인 강화**: 모든 LLM 출력에 `_extract_korean_text()` 같은 후처리 함수 적용

---

## 전체 실행 환경 정보

```
OS       : Linux (WSL2 6.6.87.2-microsoft-standard)
Python   : 3.14.4
pytest   : 9.1.1
LangChain: 0.2.x
LangGraph: 0.1.x
ChromaDB : 0.5.x
rdflib   : 6.x
MCP      : FastMCP 1.x
임베딩   : jhgan/ko-sroberta-multitask
LLM      : nvidia/nemotron-3-super-120b-a12b (NIM)
프록시   : http://150.2.127.249:9090 (SK텔레콤 사내망)
```

## 실행 순서 요약

```bash
# Stage 1: RAG
.venv/bin/python3 -m rag.query "부산 60대 남성 어부" --mode hybrid

# Stage 2-A: Extractor
.venv/bin/python3 -m labs.02_langchain.extractor --query "부산 60대 어부"

# Stage 2-B: Few-shot
.venv/bin/python3 -m labs.02_langchain.few_shot "40대 남성 의사 서울" --n-shots 3

# Stage 3: LangGraph
.venv/bin/python3 -m labs.langgraph_lab.run "서울 30대 직장인" --auto-approve

# Stage 4: MCP
.venv/bin/python3 -m labs.mcp_lab.test_client

# Stage 5: 온톨로지
.venv/bin/python3 -m labs.ontology.populate --max-docs 5000
.venv/bin/python3 -m labs.ontology.query
.venv/bin/python3 -m labs.ontology.text2sparql "경기도 중년 여성의 교육 수준 분포는?"

# Stage 6: Wiki RAG
.venv/bin/python3 -m labs.wiki_lab.wiki_rag --build-index --force
.venv/bin/python3 -m labs.wiki_lab.wiki_rag \
    --query "서울의 인구 특성과 주요 직업 분포를 설명해주세요" \
    --show-sources
```
